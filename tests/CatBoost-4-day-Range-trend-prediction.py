# %% [markdown]
# # CatBoost "clear trend vs. range" detector: crypto 4H — classification only
#
# **Target (direction-agnostic)**: what will the NEXT `HORIZON` bars look like?
# * **Trend (1)**: large AND efficient move, `|fwd return| >= TREND_Z * sigma*sqrt(H)` and path efficiency >= `TREND_ER`
# * **Range (0)**: price goes nowhere, `|fwd return| < RANGE_Z * sigma*sqrt(H)`
# * Everything in between is *ambiguous*: excluded from training / classification metrics.
#
# The model never predicts direction, only "is a clean trend likely right now?".
#
# **What you get at the end**
# 1. Classification quality (AUC, balanced accuracy, precision, log-loss/Brier) vs. baselines
#    (ADX>25, "trend persists")
# 2. Block-bootstrap confidence intervals on AUC and AUC gains (is the edge real or noise?)
# 3. Precision/lift vs. coverage, and a directional-payoff diagnostic on the flagged bars

# %% [1] Setup
import subprocess, sys, importlib, os, time, json, warnings
warnings.filterwarnings("ignore")

def _ensure(pip_name, import_name):
    try:
        importlib.import_module(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)

for _p, _m in [("catboost", "catboost"), ("ccxt", "ccxt"), ("optuna", "optuna"), ("scikit-learn", "sklearn")]:
    _ensure(_p, _m)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (roc_auc_score, roc_curve, log_loss, brier_score_loss, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score, f1_score)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% [2] Config
SYMBOLS   = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "LINK", "LTC", "AVAX"]
QUOTE     = "USDT"
TIMEFRAME = "4h"
START     = "2019-01-01"
EXCHANGES = ["binance", "bybit", "kucoin", "okx", "gate", "bitget"]   # tried in order, per symbol
CSV_DIR   = None   # optional: folder with BTC.csv, ETH.csv ... (columns: timestamp,open,high,low,close,volume)

# --- regime definition (all in units of sigma*sqrt(H), sigma = trailing EWM vol of 4H log-returns) ---
HORIZON   = 24     # bars ahead (24 x 4H = 4 days)
TREND_Z   = 1.0    # |net move| must be >= 1.0 sigma-sqrt(H) ...
TREND_ER  = 0.30   # ... and path efficiency (|net| / sum|bar moves|) >= 0.30  -> "clear trend"
RANGE_Z   = 0.5    # |net move| < 0.5 sigma-sqrt(H) -> "range"; between the two = ambiguous (dropped from training)

# --- validation / training ---
WARMUP          = 400
MIN_BARS        = WARMUP + 1500
TEST_FRAC       = 0.20     # final hold-out (touched once, at the end)
CV_FOLDS        = 4
N_TRIALS        = 20
TIME_BUDGET_MIN = 45
N_SEEDS         = 5
USE_ASSET_ID    = False    # coin id as a categorical feature (off = better generalisation across coins)
USE_GPU         = False    # True on a T4 runtime (rsm auto-disabled)
SEED            = 42
OUT_DIR         = "/content/trend_range_catboost"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)

# %% [3] Data download (robust: per-symbol exchange fallback, handles pre-listing gaps, caches only complete data)
_EX = {}
def get_ex(name):
    if name in _EX:
        return _EX[name]
    ex = None
    if hasattr(ccxt, name):
        for attempt in range(2):
            try:
                e = getattr(ccxt, name)({"enableRateLimit": True})
                e.load_markets()
                ex = e
                break
            except Exception as err:
                last = type(err).__name__
                time.sleep(2)
        if ex is None:
            print(f"  [{name}] unavailable ({last})")
    _EX[name] = ex
    return ex

def fetch_all(ex, symbol, since_iso, limit=1000):
    if symbol not in ex.markets:
        return None
    tf_ms = ex.parse_timeframe(TIMEFRAME) * 1000
    since = ex.parse8601(since_iso + "T00:00:00Z")
    now = ex.milliseconds()
    rows, hops = [], 0
    while since < now:
        batch = None
        for attempt in range(3):
            try:
                batch = ex.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=limit)
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        if batch is None:                      # persistent error
            break
        if len(batch) == 0:
            if rows:                           # nothing more to download
                break
            # symbol not listed yet at `since` (e.g. KuCoin returns [] before listing) -> hop forward
            since += limit * tf_ms
            hops += 1
            if hops > 40:
                break
            continue
        rows += batch
        nxt = batch[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
        time.sleep(ex.rateLimit / 1000)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df = df[df["ts"] + tf_ms <= now]                         # drop the still-forming candle
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")

def is_complete(df):
    return df.index[-1] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=4 * 6)

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    tcol = "timestamp" if "timestamp" in df.columns else df.columns[0]
    ts = df[tcol]
    df["ts"] = pd.to_datetime(ts, unit="ms", utc=True) if pd.api.types.is_numeric_dtype(ts) \
        else pd.to_datetime(ts, utc=True)
    return df.set_index("ts")[["open", "high", "low", "close", "volume"]].sort_index()

def clean(df):
    df = df[~df.index.duplicated()].astype(float)
    return df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]

def load_symbol(s):
    if CSV_DIR:
        return clean(load_csv(os.path.join(CSV_DIR, f"{s}.csv"))), "csv"
    for name in EXCHANGES:
        ex = get_ex(name)
        if ex is None:
            continue
        cache = os.path.join(OUT_DIR, f"v2_{name}_{s}_{TIMEFRAME}_{START}.csv")
        if os.path.exists(cache):
            df = clean(pd.read_csv(cache, index_col=0, parse_dates=True))
        else:
            df = fetch_all(ex, f"{s}/{QUOTE}", START)
            if df is None:
                continue
            df = clean(df)
            if len(df) >= MIN_BARS and is_complete(df):
                df.to_csv(cache)
        if len(df) >= MIN_BARS:
            return df, name
    return None, None

def load_all():
    raw = {}
    for s in SYMBOLS:
        try:
            df, src = load_symbol(s)
        except Exception as e:
            print(f"  {s:5s} failed: {type(e).__name__}: {e}")
            continue
        if df is None:
            print(f"  {s:5s} skipped (no exchange returned enough history)")
        else:
            raw[s] = df
            print(f"  {s:5s} {len(df):6d} bars  {df.index[0].date()} -> {df.index[-1].date()}  [{src}]")
    assert "BTC" in raw, "BTC is required (market-context features)"
    return raw

print("Downloading data ...")
RAW = load_all()

# %% [4] Feature engineering (every feature uses information up to the bar close only)
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))

def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)

def adx_di(df, n):
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = true_range(df).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi

def make_features(df, btc_lr=None):
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"] + 1e-9
    lc = np.log(c)
    r = lc.diff()
    F = {}

    # --- volatility scale (normalises returns and the label) -------------------------------
    sig = r.ewm(span=84, min_periods=84).std()
    tr = true_range(df)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    F["log_sigma"] = np.log(sig)
    F["atr_pct"] = atr14 / c

    # --- multi-horizon returns --------------------------------------------------------------
    for k in (1, 2, 3, 6, 12, 24, 42, 84, 180):
        rk = lc.diff(k)
        F[f"ret_{k}"] = rk
        if k >= 3:
            F[f"z_{k}"] = rk / (sig * np.sqrt(k))
    for n in (12, 24, 42, 84):
        F[f"abs_z_{n}"] = F[f"z_{n}"].abs()                 # direction-agnostic trend size

    # --- volatility term structure & estimators ---------------------------------------------
    rv = {n: r.rolling(n).std() for n in (6, 12, 24, 42, 84, 180, 360)}
    F["rv_ratio_6_42"] = np.log(rv[6] / rv[42])
    F["rv_ratio_12_84"] = np.log(rv[12] / rv[84])
    F["rv_ratio_24_180"] = np.log(rv[24] / rv[180])
    F["rv_ratio_84_360"] = np.log(rv[84] / rv[360])
    F["rv_z_24"] = (rv[24] - rv[24].rolling(360).mean()) / rv[24].rolling(360).std()
    F["volvol_42"] = rv[12].rolling(42).std() / rv[12].rolling(42).mean()
    hl = np.log(h / l)
    park = hl ** 2 / (4 * np.log(2))
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    for n in (12, 42):
        F[f"park_{n}"] = np.sqrt(park.rolling(n).mean())
        F[f"gk_{n}"] = np.sqrt(gk.rolling(n).mean().clip(lower=0))
    F["park_vs_rv_42"] = F["park_42"] / rv[42]
    F["skew_84"] = r.rolling(84).skew()
    F["kurt_84"] = r.rolling(84).kurt()
    F["downvol_ratio_42"] = r.clip(upper=0).pow(2).rolling(42).mean() / (r ** 2).rolling(42).mean()

    # --- trend structure: EMAs, MACD, ADX, regression, efficiency, choppiness, variance ratio ---
    E = {n: ema(c, n) for n in (10, 20, 50, 100, 200)}
    for n, e in E.items():
        F[f"dist_ema_{n}"] = (c - e) / atr14
        lag = max(3, n // 5)
        F[f"ema_slope_{n}"] = np.log(e / e.shift(lag)) / (sig * np.sqrt(lag))
    F["ema_20_50"] = (E[20] - E[50]) / atr14
    F["ema_50_200"] = (E[50] - E[200]) / atr14
    F["ema_align"] = np.sign(E[20] - E[50]) * np.sign(E[50] - E[200])      # +1 = stacked EMAs
    macd = ema(c, 12) - ema(c, 26)
    F["macd"] = macd / atr14
    F["macd_hist"] = (macd - ema(macd, 9)) / atr14
    for n in (14, 42):
        adx, pdi, mdi = adx_di(df, n)
        F[f"adx_{n}"], F[f"di_diff_{n}"] = adx, pdi - mdi
    F["adx_slope_6"] = F["adx_14"].diff(6)
    F["adx_rel_180"] = F["adx_14"] / F["adx_14"].rolling(180).mean()
    t = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    for n in (24, 84, 180):
        rho = lc.rolling(n).corr(t).clip(-0.9999, 0.9999)
        F[f"reg_r2_{n}"] = rho ** 2
        F[f"reg_t_{n}"] = rho * np.sqrt(n - 2) / np.sqrt(1 - rho ** 2)
        F[f"reg_slope_{n}"] = rho * lc.rolling(n).std() / np.sqrt((n ** 2 - 1) / 12) / sig
    for n in (12, 24, 42, 60, 120):
        path = r.abs().rolling(n).sum()
        F[f"er_{n}"] = lc.diff(n).abs() / path               # Kaufman efficiency ratio
        F[f"er_signed_{n}"] = lc.diff(n) / path
    for n in (14, 42):                                       # Choppiness Index (high = range, low = trend)
        F[f"chop_{n}"] = 100 * np.log10(tr.rolling(n).sum() / (h.rolling(n).max() - l.rolling(n).min())) / np.log10(n)
    sgn = np.sign(c - c.rolling(20).mean())
    F["cross_rate_42"] = (sgn.diff().abs() > 0).astype(float).rolling(42).mean()   # mean-reversion crossings
    for q in (3, 6, 12):
        F[f"vr_{q}"] = lc.diff(q).rolling(180).var() / (q * r.rolling(180).var())
    F["acf1_84"] = r.rolling(84).corr(r.shift(1))
    F["acf1_180"] = r.rolling(180).corr(r.shift(1))
    vote = np.sign(F["z_12"]) + np.sign(F["z_42"]) + np.sign(F["z_180"]) + np.sign(F["ema_20_50"])
    F["vote_abs"] = vote.abs()                               # agreement across horizons

    # --- regime memory: did the PAST window already meet the 'trend' definition? ---------------
    sigH = sig * np.sqrt(HORIZON)
    past_path = r.abs().rolling(HORIZON).sum()
    past_er = lc.diff(HORIZON).abs() / past_path
    past_trend = ((lc.diff(HORIZON).abs() / sigH >= TREND_Z) & (past_er >= TREND_ER)).astype(float)
    F["past_er"] = past_er
    F["past_trend_flag"] = past_trend
    F["past_trend_frac_60"] = past_trend.rolling(60).mean()
    F["past_trend_frac_180"] = past_trend.rolling(180).mean()

    # --- momentum / oscillators ---------------------------------------------------------------
    for n in (6, 14, 42):
        F[f"rsi_{n}"] = rsi(c, n)
    for n in (14, 42):
        lo, hi = l.rolling(n).min(), h.rolling(n).max()
        F[f"stoch_{n}"] = (c - lo) / (hi - lo)
    tp = (h + l + c) / 3
    F["tp_z_20"] = (tp - tp.rolling(20).mean()) / tp.rolling(20).std()

    # --- bands & range structure ----------------------------------------------------------------
    for n in (20, 60):
        ma, sd = c.rolling(n).mean(), c.rolling(n).std()
        bw = 4 * sd / ma
        F[f"bb_pctb_{n}"] = (c - (ma - 2 * sd)) / (4 * sd)
        F[f"bb_bw_{n}"] = bw
        F[f"bb_squeeze_{n}"] = bw / bw.rolling(180).median()
    F["bb_kc_ratio"] = (4 * c.rolling(20).std() / c.rolling(20).mean()) / (3 * atr14 / E[20])
    for n in (42, 180):
        hh, ll = h.rolling(n).max(), l.rolling(n).min()
        F[f"don_pos_{n}"] = (c - ll) / (hh - ll)
        F[f"dd_{n}"] = c / hh - 1
        F[f"du_{n}"] = c / ll - 1
        F[f"range_w_{n}"] = (hh - ll) / c
    F["bars_since_high_84"] = h.rolling(84).apply(lambda x: 83 - np.argmax(x), raw=True) / 84
    F["bars_since_low_84"] = l.rolling(84).apply(lambda x: 83 - np.argmin(x), raw=True) / 84

    # --- volume / liquidity -------------------------------------------------------------------------
    dv = c * v
    F["vol_ratio_12_84"] = np.log(v.rolling(12).mean() / v.rolling(84).mean())
    F["vol_ratio_42_180"] = np.log(v.rolling(42).mean() / v.rolling(180).mean())
    lv = np.log(v)
    F["vol_z_180"] = (lv - lv.rolling(180).mean()) / lv.rolling(180).std()
    flow = np.sign(r) * v
    for n in (12, 42):
        F[f"netflow_{n}"] = flow.rolling(n).sum() / v.rolling(n).sum()
    F["pv_corr_42"] = r.rolling(42).corr(lv.diff())
    ami = r.abs() / dv
    F["amihud_ratio"] = np.log(ami.rolling(24).mean() / ami.rolling(180).mean())
    vwap = (tp * v).rolling(42).sum() / v.rolling(42).sum()
    F["vwap_dist_42"] = (c - vwap) / atr14

    # --- candle shape ---------------------------------------------------------------------------------
    rng_ = (h - l).replace(0, np.nan)
    F["clv"] = ((c - l) - (h - c)) / rng_
    F["clv_mean_6"] = F["clv"].rolling(6).mean()
    F["body_atr"] = (c - o) / atr14
    F["upper_wick"] = (h - np.maximum(o, c)) / rng_
    F["lower_wick"] = (np.minimum(o, c) - l) / rng_
    F["range_atr"] = (h - l) / atr14
    F["up_frac_12"] = (r > 0).astype(float).rolling(12).mean()
    F["up_frac_42"] = (r > 0).astype(float).rolling(42).mean()

    # --- relation to BTC ---------------------------------------------------------------------------------
    if btc_lr is not None:
        F["corr_btc_84"] = r.rolling(84).corr(btc_lr)
        F["beta_btc_84"] = r.rolling(84).cov(btc_lr) / btc_lr.rolling(84).var()

    # --- calendar ------------------------------------------------------------------------------------------
    hr, dw = df.index.hour.values, df.index.dayofweek.values
    F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    F["dow_sin"], F["dow_cos"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)
    F["is_weekend"] = (dw >= 5).astype(float)

    X = pd.DataFrame(F, index=df.index)

    # =============== LABEL (looks FORWARD): never used as a feature ===============
    fwd = lc.shift(-HORIZON) - lc
    fwd_z = fwd / sigH
    fwd_path = r.abs().rolling(HORIZON).sum().shift(-HORIZON)       # sum |r| over t+1 .. t+H
    fwd_er = fwd.abs() / fwd_path
    is_trend = (fwd_z.abs() >= TREND_Z) & (fwd_er >= TREND_ER)
    is_range = fwd_z.abs() < RANGE_Z
    known = fwd_z.notna() & fwd_er.notna()
    X["lab"] = np.where(~known, np.nan, np.where(is_trend, 1.0, np.where(is_range, 0.0, np.nan)))
    X["fwd_z"] = fwd_z
    X["tf_dir"] = np.sign(F["ema_20_50"]).fillna(0.0)                # EMA20 vs EMA50 direction (diagnostic only)
    return X

def add_cross_asset(P):
    def piv(col):
        return P.pivot(index="ts", columns="asset", values=col)
    T = pd.DataFrame(index=piv("z_12").index)
    T["mkt_z_12"] = piv("z_12").mean(axis=1)
    T["mkt_z_42"] = piv("z_42").mean(axis=1)
    r6, r42 = piv("ret_6"), piv("ret_42")
    T["mkt_breadth_6"] = (r6 > 0).sum(axis=1) / r6.notna().sum(axis=1)
    T["mkt_breadth_42"] = (r42 > 0).sum(axis=1) / r42.notna().sum(axis=1)
    T["mkt_disp_6"] = r6.std(axis=1)
    T["mkt_rv_ratio"] = piv("rv_ratio_6_42").mean(axis=1)
    T["mkt_adx"] = piv("adx_14").mean(axis=1)
    T["mkt_er_24"] = piv("er_24").mean(axis=1)
    T["mkt_trend_frac"] = piv("past_trend_flag").mean(axis=1)       # how many coins are trending right now
    for col in ("z_12", "z_42", "z_180", "adx_14", "er_42", "rv_ratio_6_42", "dd_180", "rsi_14", "chop_14"):
        T[f"btc_{col}"] = piv(col)["BTC"]
    P = P.merge(T, left_on="ts", right_index=True, how="left")
    P["rel_ret_42"] = P["ret_42"] - P["ts"].map(r42.mean(axis=1))
    P["rel_rank_42"] = P.groupby("ts")["ret_42"].rank(pct=True)
    return P

def build_panel(raw):
    btc_lr = np.log(raw["BTC"]["close"]).diff()
    frames = []
    for sym, df in raw.items():
        X = make_features(df, btc_lr.reindex(df.index)).iloc[WARMUP:].copy()
        X["asset"] = sym
        frames.append(X)
    P = pd.concat(frames)
    P.index.name = "ts"
    P = P.reset_index()
    P = add_cross_asset(P)
    P = P.replace([np.inf, -np.inf], np.nan)
    P = P.dropna(subset=["z_180", "rv_ratio_84_360", "adx_42", "past_er"]).copy()   # keep unlabeled rows too
    u = pd.Index(P["ts"].unique()).sort_values()
    P["tpos"] = P["ts"].map(pd.Series(np.arange(len(u)), index=u)).astype(int)
    return P.sort_values(["tpos", "asset"]).reset_index(drop=True)

print("Building features ...")
PANEL = build_panel(RAW)
META = ["ts", "tpos", "lab", "fwd_z", "tf_dir"]
FEATS_ALL = [c for c in PANEL.columns if c not in META and (USE_ASSET_ID or c != "asset")]
_l = PANEL["lab"]
print(f"Panel: {len(PANEL):,} rows | {len(FEATS_ALL)} candidate features | assets: {PANEL['asset'].nunique()}")
print(f"Labels -> Trend: {(_l == 1).mean():.1%} | Range: {(_l == 0).mean():.1%} | "
      f"Ambiguous/unlabeled: {_l.isna().mean():.1%}   (trend share of labeled bars: {_l.dropna().mean():.1%})")

# %% [5] Chronological split: DEV | gap | TEST   (gap = HORIZON, so no training label can see into the test period)
GAP = HORIZON + 1
n_t = PANEL["tpos"].max() + 1
test_start = int(n_t * (1 - TEST_FRAC))
dev_end = test_start - GAP

DEV = PANEL[PANEL["tpos"] < dev_end].reset_index(drop=True)
TEST = PANEL[PANEL["tpos"] >= test_start].reset_index(drop=True)
LAB_D, LAB_T = DEV["lab"].notna().values, TEST["lab"].notna().values
yd = DEV["lab"].fillna(-1).astype(int).values
td = DEV["tpos"].values
print(f"DEV : {len(DEV):,} rows ({LAB_D.sum():,} labeled)  {DEV['ts'].min().date()} -> {DEV['ts'].max().date()}")
print(f"TEST: {len(TEST):,} rows ({LAB_T.sum():,} labeled)  {TEST['ts'].min().date()} -> {TEST['ts'].max().date()}")

def purged_folds(tpos, lab_mask, n_folds, start_frac=0.40):
    """Expanding-window walk-forward. Train on labeled rows only; validate/predict on ALL rows of the block."""
    L = tpos.max() + 1
    first = int(L * start_frac)
    size = (L - first) // n_folds
    folds = []
    for k in range(n_folds):
        vs = first + k * size
        ve = L if k == n_folds - 1 else vs + size
        tr = np.where((tpos < vs - GAP) & lab_mask)[0]
        va = np.where((tpos >= vs) & (tpos < ve))[0]
        folds.append((tr, va))
    return folds

def time_weights(tpos, half_life_years):
    if not half_life_years:
        return None
    return 0.5 ** ((tpos.max() - tpos) / (half_life_years * 365 * 6))

def base_params():
    p = dict(loss_function="Logloss", eval_metric="Logloss", iterations=2000, learning_rate=0.05,
             bootstrap_type="Bayesian", allow_writing_files=False, verbose=False)
    if USE_GPU:
        p["task_type"] = "GPU"
    else:
        p["thread_count"] = -1
    return p

# %% [6] Feature pruning (DEV only): correlation clusters -> permutation importance
def corr_prune(df, feats, thr=0.97):
    num = [f for f in feats if f != "asset"]
    sub = df[num].sample(min(len(df), 30000), random_state=SEED)
    cm = sub.corr().abs()
    upper = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > thr).any()]
    return [f for f in feats if f not in drop], drop

def perm_importance(model, df, feats, n_rep=3):
    rng = np.random.default_rng(SEED)
    Xv, yv = df[feats].copy(), df["lab"].values.astype(int)
    cat = [f for f in feats if f == "asset"]
    pr = lambda: np.clip(model.predict_proba(Pool(Xv, cat_features=cat))[:, 1], 1e-6, 1 - 1e-6)
    base = log_loss(yv, pr())
    imp = {}
    for f in feats:
        orig, d = Xv[f].values.copy(), []
        for _ in range(n_rep):
            Xv[f] = rng.permutation(orig)
            d.append(log_loss(yv, pr()) - base)
        Xv[f] = orig
        imp[f] = float(np.mean(d))
    return pd.Series(imp).sort_values(ascending=False)

print("Pruning features ...")
FEATS_UNC, dropped_corr = corr_prune(DEV[LAB_D], FEATS_ALL)
print(f"  dropped {len(dropped_corr)} highly-correlated features -> {len(FEATS_UNC)} left")
_cut = int(dev_end * 0.80)
_trm, _vam = (DEV["tpos"] < _cut - GAP) & LAB_D, (DEV["tpos"] >= _cut) & LAB_D
_cat0 = [f for f in FEATS_UNC if f == "asset"]
_m = CatBoostClassifier(**base_params(), depth=6, l2_leaf_reg=10, random_seed=SEED)
_m.fit(Pool(DEV.loc[_trm, FEATS_UNC], DEV.loc[_trm, "lab"].astype(int), cat_features=_cat0),
       eval_set=Pool(DEV.loc[_vam, FEATS_UNC], DEV.loc[_vam, "lab"].astype(int), cat_features=_cat0),
       early_stopping_rounds=100, use_best_model=True)
PERM = perm_importance(_m, DEV[_vam], FEATS_UNC)
FEATS = [f for f in FEATS_UNC if PERM[f] > 0]
if len(FEATS) < 30:
    FEATS = list(PERM.index[:30])
CAT = [f for f in FEATS if f == "asset"]
print(f"  permutation importance kept {len(FEATS)} / {len(FEATS_UNC)} features")
print("  top-10:", ", ".join(PERM.index[:10]))

# %% [7] Optuna search with purged walk-forward CV (objective = mean validation log-loss on labeled rows)
FOLDS = purged_folds(td, LAB_D, CV_FOLDS)
Xd = DEV[FEATS].reset_index(drop=True)

def run_cv(params, hl_years, trial=None):
    losses, aucs, iters = [], [], []
    oof = np.full(len(DEV), np.nan)
    for i, (tr, va) in enumerate(FOLDS):
        va_l = va[LAB_D[va]]
        m = CatBoostClassifier(**base_params(), **params, random_seed=SEED)
        m.fit(Pool(Xd.iloc[tr], yd[tr], cat_features=CAT, weight=time_weights(td[tr], hl_years)),
              eval_set=Pool(Xd.iloc[va_l], yd[va_l], cat_features=CAT),
              early_stopping_rounds=100, use_best_model=True)
        oof[va] = m.predict_proba(Pool(Xd.iloc[va], cat_features=CAT))[:, 1]
        pv = np.clip(oof[va_l], 1e-6, 1 - 1e-6)
        losses.append(log_loss(yd[va_l], pv))
        aucs.append(roc_auc_score(yd[va_l], pv))
        iters.append(m.get_best_iteration())
        if trial is not None:
            trial.report(float(np.mean(losses)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(losses)), float(np.mean(aucs)), iters, oof

def objective(trial):
    p = dict(depth=trial.suggest_int("depth", 4, 7),
             l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 60, log=True),
             random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
             bagging_temperature=trial.suggest_float("bagging_temperature", 0, 5),
             border_count=trial.suggest_categorical("border_count", [64, 128]))
    if not USE_GPU:
        p["rsm"] = trial.suggest_float("rsm", 0.3, 1.0)
    hl = trial.suggest_categorical("half_life_years", [0, 1, 2, 4])
    loss, auc, iters, _ = run_cv(p, hl, trial)
    trial.set_user_attr("cv_auc", auc)
    return loss

print(f"Tuning ({N_TRIALS} trials, <= {TIME_BUDGET_MIN} min) ...")
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1))
study.optimize(objective, n_trials=N_TRIALS, timeout=TIME_BUDGET_MIN * 60)
best = study.best_trial
BEST_HL = best.params["half_life_years"]
BEST = {k: v for k, v in best.params.items() if k != "half_life_years"}
print(f"Best CV logloss {best.value:.4f} | CV AUC {best.user_attrs['cv_auc']:.4f}")
print("Best params:", BEST, "| half-life (yrs):", BEST_HL)

_, _, CV_ITERS, OOF = run_cv(BEST, BEST_HL)       # out-of-fold P(trend) for every dev row from fold 1 onwards
N_FINAL = max(int(np.median(CV_ITERS) * 1.15) + 1, 100)
print(f"Best iterations per fold: {CV_ITERS} -> final model uses {N_FINAL}")

# %% [8] Helpers: smoothing, bootstrap, classification metrics
def smooth(p, assets, tpos, span):
    """Causal per-asset EMA of P(trend): trend regimes persist, so this denoises the signal."""
    if span <= 1:
        return p
    df = pd.DataFrame({"p": p, "a": assets, "t": tpos}).sort_values(["a", "t"])
    df["s"] = df.groupby("a")["p"].transform(lambda x: x.ewm(span=span, adjust=False).mean())
    return df["s"].sort_index().values

def block_idx(T, block, B, rng):
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(B, nb))
    return (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :T]

def boot_auc(y, scores, tpos, B=300, block=2 * HORIZON, seed=SEED):
    o = np.argsort(tpos, kind="stable")
    y, tpos = y[o], tpos[o]
    scores = {k: v[o] for k, v in scores.items()}
    ut, start = np.unique(tpos, return_index=True)
    end = np.r_[start[1:], len(tpos)]
    bidx = block_idx(len(ut), block, B, np.random.default_rng(seed))
    out = {k: [] for k in scores}
    for b in range(B):
        rows = np.concatenate([np.arange(start[i], end[i]) for i in bidx[b]])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        for k, v in scores.items():
            out[k].append(roc_auc_score(yy, v[rows]))
    return {k: np.array(v) for k, v in out.items()}

def hard_metrics(y, pred, score):
    return dict(auc=roc_auc_score(y, score), accuracy=accuracy_score(y, pred), balanced_acc=balanced_accuracy_score(y, pred),
                precision_trend=precision_score(y, pred, zero_division=0), recall_trend=recall_score(y, pred, zero_division=0),
                f1_trend=f1_score(y, pred, zero_division=0))

def metric_row(y, pred, score):
    m = hard_metrics(y, pred, score)
    m["log_loss"] = log_loss(y, np.clip(score, 1e-6, 1 - 1e-6))
    m["brier"] = brier_score_loss(y, np.clip(score, 0, 1))
    return m

# --- choose smoothing span and classification threshold on OOF only ---
mask = ~np.isnan(OOF)
O = DEV[mask].reset_index(drop=True)
oof_p, o_lab, o_y = OOF[mask], LAB_D[mask], yd[mask]
rows = []
for sp in (1, 2, 3, 4, 6, 8, 12):
    ps = smooth(oof_p, O["asset"].values, O["tpos"].values, sp)
    rows.append((sp, roc_auc_score(o_y[o_lab], ps[o_lab]), log_loss(o_y[o_lab], np.clip(ps[o_lab], 1e-6, 1 - 1e-6))))
SM = pd.DataFrame(rows, columns=["span", "oof_auc", "oof_logloss"]).set_index("span")
print("\nOOF smoothing study:\n", SM.round(4))
BEST_SPAN = int(SM["oof_auc"].idxmax())
oof_s = smooth(oof_p, O["asset"].values, O["tpos"].values, BEST_SPAN)

grid = np.arange(0.20, 0.801, 0.01)
bal = [balanced_accuracy_score(o_y[o_lab], (oof_s[o_lab] >= t).astype(int)) for t in grid]
TH_CLS = float(grid[int(np.argmax(bal))])
print(f"\nChosen on OOF -> smoothing span {BEST_SPAN} | classification threshold {TH_CLS:.2f}")
print("\nOOF classification @ chosen threshold:")
print(pd.Series(metric_row(o_y[o_lab], (oof_s[o_lab] >= TH_CLS).astype(int), oof_s[o_lab])).round(4).to_string())

# %% [9] Final model: seed ensemble on ALL dev (labeled) data, evaluated once on TEST
print("\nTraining final ensemble ...")
Xtr, ytr = DEV.loc[LAB_D, FEATS], yd[LAB_D]
w_final = time_weights(td[LAB_D], BEST_HL)
models, probs = [], []
for s in range(N_SEEDS):
    m = CatBoostClassifier(**{**base_params(), **BEST, "iterations": N_FINAL, "random_seed": SEED + s})
    m.fit(Pool(Xtr, ytr, cat_features=CAT, weight=w_final))
    m.save_model(os.path.join(OUT_DIR, f"trend_range_seed{s}.cbm"))
    models.append(m)
    probs.append(m.predict_proba(Pool(TEST[FEATS], cat_features=CAT))[:, 1])
P_raw = np.mean(probs, axis=0)
P_fin = smooth(P_raw, TEST["asset"].values, TEST["tpos"].values, BEST_SPAN)

# %% [10] Classification results on labeled TEST bars (clear trend vs clear range)
yT = TEST["lab"].values[LAB_T].astype(int)
pT, pTraw = P_fin[LAB_T], P_raw[LAB_T]
prior_trend = yd[LAB_D].mean()
maj = int(prior_trend >= 0.5)
base_rate = yT.mean()

R = pd.DataFrame([
    {"model": f"CatBoost (smoothed, th={TH_CLS:.2f})", **metric_row(yT, (pT >= TH_CLS).astype(int), pT)},
    {"model": f"CatBoost (raw, th={TH_CLS:.2f})", **metric_row(yT, (pTraw >= TH_CLS).astype(int), pTraw)},
    {"model": "Baseline: always majority class", **metric_row(yT, np.full(len(yT), maj), np.full(len(yT), 0.5))},
    {"model": "Baseline: ADX(14) > 25", **metric_row(yT, (TEST["adx_14"].values[LAB_T] > 25).astype(int), TEST["adx_14"].values[LAB_T])},
    {"model": "Baseline: 'trend persists' (past window was a trend)", **metric_row(yT, TEST["past_trend_flag"].values[LAB_T].astype(int), TEST["past_er"].values[LAB_T])},
]).set_index("model")
print("\n================ TEST: CLEAR TREND vs CLEAR RANGE (labeled bars only) ================")
print(f"labeled test bars: {len(yT):,} | trend share: {base_rate:.1%}")
print(R.round(4).to_string())
print(f"model log-loss {log_loss(yT, np.clip(pT, 1e-6, 1 - 1e-6)):.4f} vs. constant-prior {log_loss(yT, np.full(len(yT), prior_trend)):.4f} | "
      f"Brier {brier_score_loss(yT, pT):.4f} vs. {brier_score_loss(yT, np.full(len(yT), prior_trend)):.4f}")

bs = boot_auc(yT, {"model": pT, "adx": TEST["adx_14"].values[LAB_T], "persist": TEST["past_er"].values[LAB_T]},
              TEST["tpos"].values[LAB_T])
ci = lambda a: f"[{np.percentile(a, 2.5):.4f}, {np.percentile(a, 97.5):.4f}]"
print(f"\nAUC 95% CI (block bootstrap)        : model {ci(bs['model'])}")
print(f"AUC gain over ADX baseline          : {np.mean(bs['model'] - bs['adx']):+.4f}  CI {ci(bs['model'] - bs['adx'])}")
print(f"AUC gain over 'trend persists' rule : {np.mean(bs['model'] - bs['persist']):+.4f}  CI {ci(bs['model'] - bs['persist'])}")

print("\nHow clean are the bars the model flags as 'trend'? (TEST)")
rows = []
fz = TEST["fwd_z"].values
for th in (0.40, 0.50, 0.60, 0.70, 0.80):
    f_all = P_fin >= th
    f_lab = f_all[LAB_T]
    if f_lab.sum() < 30:
        continue
    ok = f_all & ~np.isnan(fz)
    rows.append(dict(threshold=th, coverage_all_bars=f_all.mean(), precision_trend=yT[f_lab].mean(), base_rate=base_rate,
                     lift=yT[f_lab].mean() / base_rate,
                     fwd_move_in_TF_dir_flagged=np.nanmean(TEST["tf_dir"].values[ok] * fz[ok]),
                     fwd_move_in_TF_dir_unflagged=np.nanmean((TEST["tf_dir"].values * fz)[~f_all & ~np.isnan(fz)])))
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("(fwd_move_in_TF_dir = mean of tf_dir * forward z-move: > 0 means following the EMA direction pays on average)")

# %% [11] Plots
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

for name, sc in (("CatBoost", pT), ("ADX(14)", TEST["adx_14"].values[LAB_T]), ("Past-trend persistence", TEST["past_er"].values[LAB_T])):
    fpr, tpr, _ = roc_curve(yT, sc)
    ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(yT, sc):.3f})")
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].legend(); ax[0].set_title("ROC: clear trend vs clear range (TEST)")

for name, sc in (("CatBoost", pT), ("ADX(14)", TEST["adx_14"].values[LAB_T]), ("Past-trend persistence", TEST["past_er"].values[LAB_T])):
    o = np.argsort(-sc)
    prec = np.cumsum(yT[o]) / np.arange(1, len(yT) + 1)
    cov = np.arange(1, len(yT) + 1) / len(yT)
    ax[1].plot(cov[200:], prec[200:], label=name)
ax[1].axhline(base_rate, ls="--", c="grey"); ax[1].set_xlabel("share of bars flagged 'trend'")
ax[1].set_ylabel("precision (flagged bars that were real trends)")
ax[1].set_title("Precision vs coverage (TEST)"); ax[1].legend()

fi = pd.Series(np.mean([m.get_feature_importance() for m in models], axis=0), index=FEATS).sort_values().tail(25)
ax[2].barh(fi.index, fi.values, color="#3b6fb6"); ax[2].set_title("Top-25 feature importance (seed-avg)")

plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "evaluation.png"), dpi=130); plt.show()

# %% [12] Save artefacts
out = TEST[["ts", "asset", "lab", "tf_dir"]].copy()
out["p_trend_raw"], out["p_trend"] = P_raw, P_fin
out.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
json.dump(dict(features=FEATS, cat_features=CAT, horizon=HORIZON, trend_z=TREND_Z, trend_er=TREND_ER, range_z=RANGE_Z,
               best_params=BEST, half_life_years=BEST_HL, n_iterations=N_FINAL, smoothing_span=BEST_SPAN,
               threshold_classification=TH_CLS,
               test_metrics=R.round(5).to_dict()),
          open(os.path.join(OUT_DIR, "config.json"), "w"), indent=2)
print(f"\nSaved models, predictions and config to {OUT_DIR}")
