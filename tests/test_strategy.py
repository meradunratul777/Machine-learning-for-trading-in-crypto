# =============================================================================
# SOL/USDT 4H - Next-candle colour (v4)
#
# WHAT v3's RESULT TOLD US
#   * price-only features had a real-looking edge in 2023-25 (~54%) that vanished
#     in 2026 (~46-50%)  -> the problem is NON-STATIONARITY, not missing features
#   * adding flow / derivative / order-book features did NOT help a linear model
#   * the single holdout is now "burned" (we looked at it), so v4 uses a fully
#     causal evaluation instead of another holdout
#
# WHAT v4 DOES
#   1. NEW FEATURES: rolling & anchored (daily/weekly) VWAP/TWAP distances,
#      intra-candle VWAP/TWAP, volume-clock and trade-count-clock ("tick bar"
#      style) sub-bars, VPIN-style flow toxicity, Kyle-lambda / Amihud price impact
#   2. ADAPTATION: many configs (short windows, exponential-decay weighting,
#      strong L2 shrinkage) walk-forward trained on the whole history
#   3. ONLINE MODEL SELECTION: at every refit the top-K configs by TRAILING
#      log-loss are averaged (uses only information available at that time)
#   4. TRUST GATE: optional - act only when recent accuracy is healthy
#   5. FORWARD-TEST LOG: every live prediction is saved & scored later (the only
#      truly clean test from now on)
#
# Paste into a Google Colab cell and run (roughly 10-20 minutes).
# =============================================================================
import io, os, time, itertools, warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import binomtest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")

# ---- Settings ---------------------------------------------------------------
SYMBOL, PEERS   = "SOLUSDT", ["BTCUSDT", "ETHUSDT"]
YEARS_BACK      = 4
MIN_TRAIN       = 1800      # first walk-forward fit uses at least this many candles
STEP            = 90        # refit every 90 candles (~15 days)
SEL_LOOKBACK    = 1000      # online selection: judge configs on their last 1000 predictions
TOPK            = 7         # average the K best configs
GATE_WINDOW     = 500       # trust gate: trailing accuracy window
RECENT          = 1500      # "recent regime" report (~8 months)
LOG_PATH        = "/content/sol_forward_log.csv"   # mount Google Drive for a persistent path
USE_FUTURES     = True      # futures flow / basis / premium / funding
USE_METRICS     = True      # open interest + long/short ratios (daily archive files)
USE_BOOKDEPTH   = True      # order-book depth imbalance (daily archive files, 2023+)
SPOT_HOSTS      = ["https://api.binance.com", "https://api.binance.us"]
FAPI            = "https://fapi.binance.com"
VISION          = "https://data.binance.vision/data/futures/um/daily"
TF              = pd.Timedelta("4h")

SESSION = requests.Session()
NOW = pd.Timestamp.now(tz="UTC")
START = NOW - pd.Timedelta(days=int(365.25 * YEARS_BACK))
START_MS = int(START.timestamp() * 1000)


# ---- 1. HTTP / download helpers --------------------------------------------
def http_get(url, params=None, retries=4):
    for i in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 451:
                raise RuntimeError("HTTP 451 (region blocked)")
            time.sleep(2 * (i + 1))           # 418 / 429 / 5xx -> back off
        except requests.RequestException:
            time.sleep(1 + i)
    return None


KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_vol", "trades", "tb_base", "tb_quote", "ignore"]
NUM = ["open", "high", "low", "close", "volume", "quote_vol", "trades", "tb_base", "tb_quote"]


def fetch_klines(url, symbol, interval, step_ms, limit, workers=8):
    """Parallel chunked download of fixed-step klines."""
    end_ms = int(time.time() * 1000)
    starts = list(range(START_MS, end_ms, limit * step_ms))

    def job(s):
        r = http_get(url, dict(symbol=symbol, interval=interval, startTime=s, limit=limit))
        return r.json() if r is not None else []

    with ThreadPoolExecutor(workers) as ex:
        chunks = list(ex.map(job, starts))
    rows = [x for c in chunks for x in c]
    if len(rows) < 100:
        raise RuntimeError(f"no data from {url} for {symbol}")
    d = pd.DataFrame(rows, columns=KL_COLS)
    d["ts"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    d[NUM] = d[NUM].astype(float)
    return d.drop_duplicates("ts").set_index("ts").sort_index()[NUM]


def fetch_funding(symbol):
    rows, cur = [], START_MS
    while True:
        r = http_get(FAPI + "/fapi/v1/fundingRate", dict(symbol=symbol, startTime=cur, limit=1000))
        if r is None:
            break
        b = r.json()
        if not b:
            break
        rows += b
        if len(b) < 1000:
            break
        cur = b[-1]["fundingTime"] + 1
    d = pd.DataFrame(rows)
    d["ts"] = pd.to_datetime(d["fundingTime"], unit="ms", utc=True).dt.floor("min")
    s = d.drop_duplicates("ts").set_index("ts").sort_index()["fundingRate"].astype(float)
    if len(s) < 50:
        raise RuntimeError("funding history empty")
    return s


def parse_time(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ms", utc=True)
    return pd.to_datetime(s, utc=True)


def download_daily(kind, symbol, first_day, parser, workers=16):
    days = pd.date_range(first_day.normalize().tz_localize(None), NOW.normalize().tz_localize(None), freq="D")

    def job(day):
        try:
            url = f"{VISION}/{kind}/{symbol}/{symbol}-{kind}-{day:%Y-%m-%d}.zip"
            r = http_get(url)
            if r is None:
                return None
            return parser(pd.read_csv(io.BytesIO(r.content), compression="zip"))
        except Exception:
            return None

    with ThreadPoolExecutor(workers) as ex:
        out = [o for o in ex.map(job, days) if o is not None]
    if not out:
        return None, 0
    d = pd.concat(out).sort_index()
    return d[~d.index.duplicated(keep="last")], len(out)


def parse_metrics(df):
    df = df.copy()
    df["ts"] = parse_time(df["create_time"])
    df = df.set_index("ts")
    cols = [c for c in ["sum_open_interest", "sum_open_interest_value",
                        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                        "count_long_short_ratio", "sum_taker_long_short_vol_ratio"] if c in df.columns]
    return df[cols].astype(float).resample("4h").last()


def parse_depth(df):
    df = df.copy()
    df["ts"] = parse_time(df["timestamp"])
    p = df.pivot_table(index="ts", columns="percentage", values="notional", aggfunc="last")
    out = pd.DataFrame(index=p.index)
    for k in (1, 2, 5):
        if (-k in p.columns) and (k in p.columns):
            b, a = p[-k], p[k]
            out[f"ob_imb{k}"] = (b - a) / (b + a)
            if k == 1:
                out["ob_depth1"] = b + a
    imb = [c for c in out.columns if c.startswith("ob_imb")]
    g = out.resample("4h")
    return pd.concat([g[imb].mean().add_suffix("_mean"),
                      g[imb].last().add_suffix("_last"),
                      g[["ob_depth1"]].last().add_suffix("_last")], axis=1)


# ---- 2. Load everything -----------------------------------------------------
print("Downloading spot candles...")
SPOT_HOST, sol = None, None
for host in SPOT_HOSTS:
    try:
        sol = fetch_klines(host + "/api/v3/klines", SYMBOL, "4h", 4 * 3600_000, 1000)
        SPOT_HOST = host
        break
    except Exception as e:
        print(f"[{host}] failed: {str(e)[:80]}")
if sol is None:
    raise RuntimeError("Could not download spot data (try a different Colab runtime region).")
print("Spot host:", SPOT_HOST)

btc = fetch_klines(SPOT_HOST + "/api/v3/klines", PEERS[0], "4h", 4 * 3600_000, 1000)
eth = fetch_klines(SPOT_HOST + "/api/v3/klines", PEERS[1], "4h", 4 * 3600_000, 1000)

print("Downloading 5-minute candles (intra-candle microstructure)...")
m5 = fetch_klines(SPOT_HOST + "/api/v3/klines", SYMBOL, "5m", 5 * 60_000, 1000)

fut = prem = fund = metrics = depth = None
if USE_FUTURES:
    print("Downloading futures data...")
    try:
        fut = fetch_klines(FAPI + "/fapi/v1/klines", SYMBOL, "4h", 4 * 3600_000, 1500, workers=4)
    except Exception as e:
        print("  futures klines unavailable:", str(e)[:80])
    try:
        prem = fetch_klines(FAPI + "/fapi/v1/premiumIndexKlines", SYMBOL, "4h", 4 * 3600_000, 1500, workers=4)
    except Exception as e:
        print("  premium index unavailable:", str(e)[:80])
    try:
        fund = fetch_funding(SYMBOL)
    except Exception as e:
        print("  funding unavailable:", str(e)[:80])

if USE_METRICS:
    print("Downloading open-interest / long-short archive (daily files)...")
    metrics, nd = download_daily("metrics", SYMBOL, START, parse_metrics)
    print(f"  metrics days: {nd}")
    if metrics is None or nd < 200:
        metrics = None
        print("  not enough metrics data -> skipped")

if USE_BOOKDEPTH:
    print("Downloading order-book depth archive (daily files, ~1-4 min)...")
    depth, nd = download_daily("bookDepth", SYMBOL, max(START, pd.Timestamp("2023-01-01", tz="UTC")), parse_depth)
    print(f"  bookDepth days: {nd}")
    if depth is None or nd < 200:
        depth = None
        print("  not enough bookDepth data -> skipped")

# keep only CLOSED 4h candles & align peers
sol = sol[sol.index + TF <= NOW]
common = sol.index.intersection(btc.index).intersection(eth.index)
sol, btc, eth = sol.loc[common], btc.loc[common], eth.loc[common]
expected = pd.date_range(sol.index[0], sol.index[-1], freq="4h")
print(f"4H candles: {len(sol)}  ({sol.index[0]} -> {sol.index[-1]}),  missing: {len(expected) - len(sol)}")


# ---- 3. Features ------------------------------------------------------------
def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def zs(s, n=30):
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


def price_features(d, btc, eth):
    f = pd.DataFrame(index=d.index)
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    ret = c.pct_change()
    f["body"] = (c - o) / o
    f["upper_wick"] = (h - np.maximum(o, c)) / o
    f["lower_wick"] = (np.minimum(o, c) - l) / o
    f["range"] = (h - l) / o
    f["close_pos"] = (c - l) / (h - l).replace(0, np.nan)
    for k in range(1, 6):
        f[f"ret_lag{k}"] = ret.shift(k - 1)
        f[f"green_lag{k}"] = (c > o).astype(int).shift(k - 1)
    f["rsi14"] = rsi(c)
    f["ema10_30"] = c.ewm(span=10, adjust=False).mean() / c.ewm(span=30, adjust=False).mean() - 1
    f["close_ema50"] = c / c.ewm(span=50, adjust=False).mean() - 1
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    for k in (3, 6, 12, 18, 42):
        f[f"ret_{k}"] = c.pct_change(k)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    f["vol_std20"] = ret.rolling(20).std()
    f["vol_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    f["bb_z"] = zs(c, 20)
    f["ret_z"] = ret / ret.rolling(20).std()
    hi, lo = h.rolling(24).max(), l.rolling(24).min()
    f["donchian_pos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    f["vol_z"] = zs(v, 20)
    hr, dow = d.index.hour, d.index.dayofweek
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    for name, p in (("btc", btc), ("eth", eth)):
        f[f"{name}_body"] = (p["close"] - p["open"]) / p["open"]
        f[f"{name}_ret1"] = p["close"].pct_change()
        f[f"{name}_ret6"] = p["close"].pct_change(6)
    f["sol_rel_btc1"] = ret - btc["close"].pct_change()
    f["btc_rsi14"] = rsi(btc["close"])
    return f


def clock_features(d, wcol, prefix, nb=4):
    """Split each 4H candle into nb sub-bars of EQUAL volume (or equal trade count) - a
    volume-clock / tick-clock view built from 5m bars - and compare early vs late behaviour."""
    cum = d.groupby("bin")[wcol].cumsum()
    tot = d.groupby("bin")[wcol].transform("sum").replace(0, np.nan)
    b = np.floor(cum / tot * nb - 1e-9).clip(0, nb - 1)
    tmp = pd.DataFrame({"bin": d["bin"], "b": b, "ret": d["ret"], "net": d["net"], "vol": d["volume"]})
    agg = tmp.groupby(["bin", "b"]).agg(ret=("ret", "sum"), net=("net", "sum"), vol=("vol", "sum"))
    agg["flow"] = agg["net"] / agg["vol"].replace(0, np.nan)
    ret, flow = agg["ret"].unstack("b"), agg["flow"].unstack("b")
    out = pd.DataFrame(index=ret.index)
    out[f"{prefix}_late_early"] = (ret[nb - 1] + ret[nb - 2]) - (ret[0] + ret[1])
    out[f"{prefix}_last_ret"] = ret[nb - 1]
    out[f"{prefix}_last_flow"] = flow[nb - 1]
    out[f"{prefix}_first_flow"] = flow[0]
    return out


def micro_features(m5, idx):
    """Intra-candle microstructure from 5-minute bars (48 bars per 4H candle)."""
    d = m5[["close", "volume", "quote_vol", "trades", "tb_base"]].copy()
    d["ret"] = np.log(d["close"]).diff()
    d["net"] = 2 * d["tb_base"] - d["volume"]               # taker buy - taker sell
    d["sq"], d["cube"] = d["ret"] ** 2, d["ret"] ** 3
    d["up"] = (d["ret"] > 0).astype(float)
    bins = d.index.floor("4h")
    d["pos"] = np.asarray((d.index - bins) // pd.Timedelta("5min"))
    d["bin"] = bins
    g = d.groupby("bin")
    tv = g["volume"].sum().replace(0, np.nan)
    f = pd.DataFrame({
        "cnt": g["ret"].count(),
        "mi_rv": np.sqrt(g["sq"].sum()),
        "mi_up_frac": g["up"].mean(),
        "mi_avg_trade": tv / g["trades"].sum().replace(0, np.nan),
        "mi_max_bar_share": g["volume"].max() / tv,
    })
    f["mi_rskew"] = np.sqrt(f["cnt"]) * g["cube"].sum() / (g["sq"].sum() ** 1.5).replace(0, np.nan)
    for name, lo, hi in (("last1h", 36, 48), ("last30m", 42, 48), ("first1h", 0, 12)):
        s = d[(d["pos"] >= lo) & (d["pos"] < hi)].groupby("bin")
        f[f"mi_{name}_ret"] = s["ret"].sum()
        f[f"mi_{name}_flow"] = s["net"].sum() / s["volume"].sum().replace(0, np.nan)
        f[f"mi_{name}_volshare"] = s["volume"].sum() / tv
    # --- VWAP / TWAP inside the candle ---
    d["abs_net"] = d["net"].abs()
    d["abs_ret"] = d["ret"].abs()
    d["imb"] = d["net"] / d["volume"].replace(0, np.nan)
    d["ret_imb"] = d["ret"] * d["imb"]
    d["imb_sq"] = d["imb"] ** 2
    g2 = d.groupby("bin")
    vw = g2["quote_vol"].sum() / tv
    tw = g2["close"].mean()
    last_close = g2["close"].last()
    f["mi_vwap_twap_gap"] = vw / tw - 1
    f["mi_close_vs_twap"] = last_close / tw - 1
    s_ = d[d["pos"] >= 36].groupby("bin")
    lh_vwap = s_["quote_vol"].sum() / s_["volume"].sum().replace(0, np.nan)
    f["mi_last1h_vwap_vs_candle"] = lh_vwap / vw - 1
    f["mi_close_vs_last1h_vwap"] = last_close / lh_vwap - 1
    # --- order-flow toxicity (VPIN-style) and price impact ---
    f["mi_vpin"] = g2["abs_net"].sum() / tv
    f["mi_flow_persist"] = g2["net"].sum().abs() / g2["abs_net"].sum().replace(0, np.nan)
    f["mi_amihud"] = g2["abs_ret"].sum() / g2["quote_vol"].sum().replace(0, np.nan)
    sx, sy = g2["imb"].sum(), g2["ret"].sum()
    cov = g2["ret_imb"].sum() - sx * sy / f["cnt"]
    var = g2["imb_sq"].sum() - sx ** 2 / f["cnt"]
    f["mi_kyle_lambda"] = cov / var.replace(0, np.nan)
    # --- volume-clock and tick(trade-count)-clock sub-bars ---
    f = f.join(pd.concat([clock_features(d, "quote_vol", "vc"), clock_features(d, "trades", "tc")], axis=1))
    f.loc[f["cnt"] != 48, :] = np.nan                        # require complete candles
    return f.drop(columns="cnt").reindex(idx)


def flow_features(sol, m5):
    f = pd.DataFrame(index=sol.index)
    v, c, o = sol["volume"], sol["close"], sol["open"]
    net = 2 * sol["tb_base"] - v
    f["fl_net_ratio"] = net / v.replace(0, np.nan)
    for k in (1, 2, 3):
        f[f"fl_net_lag{k}"] = f["fl_net_ratio"].shift(k)
    for k in (6, 18, 42):
        f[f"fl_cum{k}"] = net.rolling(k).sum() / v.rolling(k).sum()
    tr = sol["trades"].replace(0, np.nan)
    f["fl_trades_z"] = zs(tr)
    f["fl_avg_trade_z"] = zs(v / tr)
    f["fl_qvol_z"] = zs(sol["quote_vol"])
    f["fl_close_vs_vwap"] = c / (sol["quote_vol"] / v.replace(0, np.nan)) - 1
    f["fl_absorb"] = np.where(np.sign(c - o) != np.sign(net), np.sign(c - o), 0.0)   # price vs flow disagreement
    return pd.concat([f, micro_features(m5, sol.index)], axis=1)


def deriv_features(sol, fut, prem, fund):
    idx, f = sol.index, pd.DataFrame(index=sol.index)
    if fut is not None:
        fu = fut.reindex(idx)
        fnet = 2 * fu["tb_base"] - fu["volume"]
        f["dv_fut_net"] = fnet / fu["volume"].replace(0, np.nan)
        f["dv_fut_net_lag1"] = f["dv_fut_net"].shift(1)
        f["dv_fut_cum6"] = fnet.rolling(6).sum() / fu["volume"].rolling(6).sum()
        f["dv_flow_gap"] = f["dv_fut_net"] - (2 * sol["tb_base"] - sol["volume"]) / sol["volume"].replace(0, np.nan)
        f["dv_basis"] = fu["close"] / sol["close"] - 1
        f["dv_basis_chg"] = f["dv_basis"].diff()
        ratio = np.log(fu["quote_vol"] / sol["quote_vol"])
        f["dv_futspot_vol_z"] = zs(ratio, 42)
    if prem is not None:
        p = prem.reindex(idx)["close"]
        f["dv_prem"], f["dv_prem_chg"], f["dv_prem_z"] = p, p.diff(), zs(p, 42)
    if fund is not None:
        fr = fund.reindex(idx, method="ffill")
        f["dv_funding"], f["dv_funding_chg"] = fr, fr.diff()
        f["dv_funding_mean6"] = fr.rolling(6).mean()
    return f


def metrics_features(sol, met):
    m = met.reindex(sol.index).ffill(limit=12)
    f, ret = pd.DataFrame(index=sol.index), sol["close"].pct_change()
    if "sum_open_interest" in m:
        oi = m["sum_open_interest"]
        f["me_oi_chg1"], f["me_oi_chg6"] = oi.pct_change(), oi.pct_change(6)
        f["me_oi_x_dir"] = f["me_oi_chg1"] * np.sign(ret)      # OI up with price up/down
    if "sum_toptrader_long_short_ratio" in m:
        top = np.log(m["sum_toptrader_long_short_ratio"])
        f["me_top_ls"], f["me_top_ls_chg"] = top, top.diff()
    if "count_long_short_ratio" in m:
        crowd = np.log(m["count_long_short_ratio"])
        f["me_crowd_ls"], f["me_crowd_ls_chg"] = crowd, crowd.diff()
        if "me_top_ls" in f:
            f["me_smart_vs_crowd"] = f["me_top_ls"] - crowd
    if "sum_taker_long_short_vol_ratio" in m:
        f["me_taker_ls"] = np.log(m["sum_taker_long_short_vol_ratio"])
    return f


def depth_features(sol, dep):
    f = dep.reindex(sol.index).ffill(limit=12).copy()
    if "ob_depth1_last" in f:
        f["ob_depth1_z"] = zs(np.log(f["ob_depth1_last"]), 30)
        f = f.drop(columns="ob_depth1_last")
    if "ob_imb1_mean" in f:
        f["ob_imb1_chg"] = f["ob_imb1_mean"].diff()
    return f


def vwap_features(sol):
    """Rolling & anchored VWAP / TWAP distances (known at candle close)."""
    f = pd.DataFrame(index=sol.index)
    h, l, c, v, q = sol["high"], sol["low"], sol["close"], sol["volume"], sol["quote_vol"]
    typ = (h + l + c) / 3
    sd = c.pct_change().rolling(20).std()
    for k in (6, 18, 42, 126):
        vwap = q.rolling(k).sum() / v.rolling(k).sum()
        twap = typ.rolling(k).mean()
        f[f"vw_c_vwap{k}"] = c / vwap - 1
        f[f"vw_c_twap{k}"] = c / twap - 1
        f[f"vw_vwap_twap{k}"] = vwap / twap - 1            # >0: more volume traded at higher prices
        if k <= 42:
            f[f"vw_z_vwap{k}"] = (c / vwap - 1) / (sd * np.sqrt(k))
    idx = sol.index
    week = (idx - pd.to_timedelta(idx.dayofweek, unit="D")).floor("D")
    for name, key in (("day", idx.floor("D")), ("week", week)):
        av = q.groupby(key.values).cumsum() / v.groupby(key.values).cumsum()
        f[f"vw_c_avwap_{name}"] = c / av - 1                # anchored VWAP (UTC day / week)
    return f


price_f = price_features(sol, btc, eth)
flow_f = flow_features(sol, m5)
parts = {"price": price_f, "vwap": vwap_features(sol), "flow": flow_f}
if any(x is not None for x in (fut, prem, fund)):
    parts["deriv"] = deriv_features(sol, fut, prem, fund)
full_parts = []
if metrics is not None:
    full_parts.append(metrics_features(sol, metrics))
if depth is not None:
    full_parts.append(depth_features(sol, depth))
if full_parts:
    parts["full"] = pd.concat(full_parts, axis=1)

X_all = pd.concat(parts.values(), axis=1).replace([np.inf, -np.inf], np.nan)
X_all = X_all.loc[:, ~X_all.columns.duplicated()]

# nested feature sets: price < flow < deriv < full
FEATSETS, acc_cols = {}, []
for name, part in parts.items():
    acc_cols = acc_cols + [c for c in part.columns if c not in acc_cols]
    FEATSETS[name] = list(acc_cols)
print({k: len(v) for k, v in FEATSETS.items()}, "features per set")

# Target: is the NEXT candle green?
y_all = (sol["close"].shift(-1) > sol["open"].shift(-1)).astype(float)
y_all.iloc[-1] = np.nan
X_latest = X_all.iloc[[-1]]

need = FEATSETS[list(FEATSETS)[-1]]
data = X_all[need].assign(target=y_all).dropna()          # same rows for every feature set -> fair comparison
X, y = X_all.loc[data.index], data["target"].astype(int)
n = len(X)
print(f"Samples: {n} ({X.index[0].date()} -> {X.index[-1].date()})")


# ---- 4. Walk-forward engine -------------------------------------------------
def make_model(cfg):
    if cfg["model"] == "lr":
        return Pipeline([("sc", StandardScaler()),
                         ("clip", FunctionTransformer(lambda z: np.clip(z, -5, 5))),
                         ("lr", LogisticRegression(C=cfg["C"], penalty="l2",
                                                   solver="liblinear", max_iter=1000))])
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100,
                                          min_samples_leaf=100, l2_regularization=5.0, random_state=0)


def fit_model(cfg, Xtr, ytr):
    m = make_model(cfg)
    if cfg["model"] == "lr" and cfg["hl"]:                   # exponential recency weighting
        age = np.arange(len(Xtr))[::-1]
        m.fit(Xtr, ytr, lr__sample_weight=0.5 ** (age / cfg["hl"]))
    else:
        m.fit(Xtr, ytr)
    return m


def walk_forward(cfg, start, end):
    """Each block of STEP candles is predicted by a model trained ONLY on earlier candles."""
    Xc, p = X[FEATSETS[cfg["feats"]]], np.full(n, np.nan)
    for s in range(start, end, STEP):
        e = min(s + STEP, end)
        lo = 0 if cfg["window"] is None else max(0, s - cfg["window"])
        p[s:e] = fit_model(cfg, Xc.iloc[lo:s], y.iloc[lo:s]).predict_proba(Xc.iloc[s:e])[:, 1]
    return p


RECENCY = [dict(window=1000, hl=None), dict(window=2000, hl=None), dict(window=None, hl=None),
           dict(window=None, hl=500), dict(window=None, hl=1500)]
configs = []
for feats in FEATSETS:
    for C in (0.003, 0.03):                                  # strong shrinkage (v3 was over-confident)
        for rec in RECENCY:
            configs.append(dict(model="lr", feats=feats, C=C, **rec))
for feats in list(FEATSETS)[-2:]:
    configs.append(dict(model="hgb", feats=feats, C=None, window=None, hl=None))


# ---- 5. Walk-forward predictions for every config (whole history) -----------
print(f"\nWalk-forward training of {len(configs)} configs over {n - MIN_TRAIN} candles...")
PM = np.full((len(configs), n), np.nan)
for i, cfg in enumerate(configs):
    PM[i] = walk_forward(cfg, MIN_TRAIN, n)
    if (i + 1) % 10 == 0 or i + 1 == len(configs):
        print(f"  {i + 1}/{len(configs)} done")
PM = np.clip(PM, 1e-4, 1 - 1e-4)
yv = y.values.astype(float)
first = MIN_TRAIN + SEL_LOOKBACK
recent_start = max(first, n - RECENT)


def report(name, p, a, b, mask=None):
    idx = np.arange(a, b)
    if mask is not None:
        idx = idx[mask[a:b]]
    pp, yt = p[idx], yv[idx]
    m = len(idx)
    k = int(((pp >= 0.5) == (yt == 1)).sum())
    ci = binomtest(k, m, 0.5).proportion_ci(confidence_level=0.95, method="wilson")
    pv = binomtest(k, m, 0.5, alternative="greater").pvalue
    auc = roc_auc_score(yt, pp) if len(np.unique(yt)) > 1 else float("nan")
    print(f"{name:<44} n={m:<5} acc={k / m:.4f}  95%CI[{ci.low:.3f},{ci.high:.3f}]  p={pv:.3f}  AUC={auc:.4f}")
    return k / m, ci


def acc_auc(p, a, b):
    yt, pp = yv[a:b], p[a:b]
    return ((pp >= 0.5) == (yt == 1)).mean(), roc_auc_score(yt, pp)


rows = []
for cfg, p in zip(configs, PM):
    a1, u1 = acc_auc(p, first, n)
    a2, u2 = acc_auc(p, recent_start, n)
    rec = (f"win{int(cfg['window'])}" if cfg["window"] else f"hl{int(cfg['hl'])}" if cfg["hl"] else "expanding")
    rows.append({**cfg, "recency": rec, "acc_all": a1, "auc_all": u1, "acc_recent": a2, "auc_recent": u2})
table = pd.DataFrame(rows)

print("\n--- Diagnostics: average over ALL configs in each group (no 'best-of' bias) ---")
cols_show = ["acc_all", "auc_all", "acc_recent", "auc_recent"]
print("\nBy feature set (does the extra data help?):")
print(table.groupby("feats", sort=False)[cols_show].mean().round(4).to_string())
print("\nBy recency handling (does adapting faster help?):")
print(table[table.model == "lr"].groupby("recency", sort=False)[cols_show].mean().round(4).to_string())


# ---- 6. Online model selection (causal) -------------------------------------
ll = -(yv * np.log(PM) + (1 - yv) * np.log(1 - PM))          # per-row log-loss of every config
p_sel = np.full(n, np.nan)
picked = []
for s in range(first, n, STEP):
    e = min(s + STEP, n)
    trailing = np.nanmean(ll[:, s - SEL_LOOKBACK:s], axis=1)  # only outcomes already known at row s
    top = np.argsort(trailing)[:TOPK]
    p_sel[s:e] = PM[top, s:e].mean(axis=0)
    picked += [(configs[j]["feats"], table.loc[j, "recency"]) for j in top]
p_all = np.nanmean(PM, axis=0)                               # naive equal-weight ensemble (reference)
i_fixed = next(i for i, c in enumerate(configs) if c["model"] == "lr" and c["feats"] == "price"
               and c["C"] == 0.03 and c["window"] == 2000)

print("\n=========== CAUSAL WALK-FORWARD RESULTS (no hindsight selection) ===========")
print(f"Period: {X.index[first].date()} -> {X.index[-1].date()}   |   random = 50%, majority baseline = "
      f"{max(yv[first:].mean(), 1 - yv[first:].mean()):.4f}")
acc_sel, ci_sel = report("Online-selected top-K ensemble  [ALL]", p_sel, first, n)
report("Online-selected top-K ensemble  [RECENT]", p_sel, recent_start, n)
report("Equal-weight ensemble of all configs [ALL]", p_all, first, n)
report("Equal-weight ensemble of all configs [RECENT]", p_all, recent_start, n)
report("Fixed price-only LR (C=.03, win=2000) [ALL]", PM[i_fixed], first, n)
report("Fixed price-only LR (C=.03, win=2000) [RECENT]", PM[i_fixed], recent_start, n)
print(f"Median single config accuracy [ALL]: {table['acc_all'].median():.4f} | [RECENT]: {table['acc_recent'].median():.4f}")

pk = pd.Series(picked)
print("\nWhat the selector kept choosing (share of top-K slots):")
print(pk.apply(lambda t: t[0]).value_counts(normalize=True).round(2).to_string())
print(pk.apply(lambda t: t[1]).value_counts(normalize=True).round(2).to_string())

# trust gate: act only when the selector's recent record is healthy (uses past outcomes only)
correct = np.where(np.isnan(p_sel), np.nan, ((p_sel >= 0.5) == (yv == 1)).astype(float))
trail = pd.Series(correct).rolling(GATE_WINDOW, min_periods=GATE_WINDOW // 2).mean().shift(1).values
print(f"\nTrust gate (exploratory): act only if accuracy over previous {GATE_WINDOW} calls >= threshold")
for thr in (0.0, 0.505, 0.515, 0.525):
    mask = np.isfinite(p_sel) if thr == 0 else (np.isfinite(trail) & (trail >= thr))
    if mask[first:n].sum() >= 100:
        report(f"  threshold {thr:.3f} (coverage {mask[first:n].mean():.0%})", p_sel, first, n, mask)

cs = pd.Series(correct, index=X.index).dropna()
print("\nAccuracy by quarter (online-selected ensemble):")
print(cs.groupby(cs.index.tz_localize(None).to_period("Q")).agg(["mean", "count"]).round(3).to_string())

plt.figure(figsize=(10, 4))
cs.rolling(500, min_periods=300).mean().plot(color="steelblue", label="online-selected ensemble")
pd.Series(((PM[i_fixed] >= 0.5) == (yv == 1)).astype(float), index=X.index)[first:].rolling(500, min_periods=300) \
    .mean().plot(color="gray", alpha=0.7, label="fixed price-only LR")
plt.axhline(0.5, color="k", ls="--"); plt.axhline(0.52, color="green", ls=":", label="52% target")
plt.title("Rolling 500-candle accuracy (causal walk-forward)")
plt.legend(); plt.tight_layout(); plt.show()

if ci_sel.low > 0.5:
    print("\nVERDICT: whole-period accuracy is statistically above 50% (95% CI excludes 50%). "
          "Check the [RECENT] line before trusting it going forward - edges here have come and gone.")
elif acc_sel >= 0.52:
    print("\nVERDICT: point estimate >= 52% but the CI still includes 50% -> not proven.")
else:
    print("\nVERDICT: no demonstrable edge in the causal walk-forward.")


# ---- 7. Live prediction -----------------------------------------------------
final_top = np.argsort(np.nanmean(ll[:, n - SEL_LOOKBACK:n], axis=1))[:TOPK]
ps, warned = [], []
for j in final_top:
    cfg, cols = configs[j], FEATSETS[configs[j]["feats"]]
    lo = 0 if cfg["window"] is None else max(0, n - cfg["window"])
    model = fit_model(cfg, X[cols].iloc[lo:], y.iloc[lo:])
    xl = X_latest[cols].copy()
    if xl.isna().any(axis=None):
        warned += list(xl.columns[xl.isna().iloc[0]])
        xl = xl.fillna(X[cols].median())
    ps.append(model.predict_proba(xl)[0, 1])
p_green = float(np.mean(ps))
cand_open = X_latest.index[0] + TF

print("\n============ NEXT CANDLE PREDICTION ============")
print(f"Last closed candle opened at : {X_latest.index[0]}")
print(f"Predicting candle opening at : {cand_open}")
print(f"Probability GREEN : {p_green:.2%}")
print(f"Probability RED   : {1 - p_green:.2%}")
print(f"Trailing accuracy of the selector (last {GATE_WINDOW} calls): {np.nanmean(correct[-GATE_WINDOW:]):.3f}")
if warned:
    print(f"[warn] {len(set(warned))} live feature(s) missing, filled with median: {sorted(set(warned))[:6]}")
print("(Educational use only - not financial advice.)")

# ---- 8. Forward-test log: the only clean out-of-sample test from here on -----
try:
    log = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame(columns=["candle_open", "p_green"])
    log["candle_open"] = pd.to_datetime(log["candle_open"], utc=True)
    known = log[log["candle_open"].isin(sol.index)].copy()
    if len(known):
        actual = (sol.loc[known["candle_open"], "close"].values > sol.loc[known["candle_open"], "open"].values)
        hit = (known["p_green"].values >= 0.5) == actual
        print(f"\nFORWARD TEST so far: {int(hit.sum())}/{len(hit)} correct = {hit.mean():.1%}  "
              f"(need several hundred calls before this means anything)")
    if cand_open not in set(log["candle_open"]):
        log.loc[len(log)] = [cand_open, p_green]
    log.to_csv(LOG_PATH, index=False)
    print(f"Prediction logged to {LOG_PATH} (re-run this script after future candles close to score it).")
except Exception as e:
    print("[forward log skipped]", str(e)[:100])
