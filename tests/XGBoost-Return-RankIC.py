# %% [markdown]
# # Cross-sectional RankIC model for crypto (4H bars) with XGBoost
#
# **Goal:** at every 4H bar, rank ~50-60 liquid USDT coins by their expected forward return
# (default: next 24h, entered one bar later) and measure quality with **RankIC**
# (per-bar Spearman correlation between prediction and realised forward return).
#
# **Pipeline**
# 1. Download 4H klines from Binance public market-data endpoints (no API key needed), cache to disk.
# 2. Build a point-in-time liquid universe (no look-ahead in who is "in" at each bar).
# 3. Engineer ~40 per-coin features (momentum/reversal, vol, liquidity, order-flow, market-relative),
#    cross-sectionally rank-normalise them each bar, and rank-normalise the target each bar.
# 4. Unit-test the features for look-ahead leakage.
# 5. Tune XGBoost with **purged, embargoed, expanding-window walk-forward CV** on the development period only.
# 6. Evaluate once on a **held-out final period** with periodic refits (as you'd run it live),
#    against baselines (Ridge, best single factor) and a **shuffled-label placebo**.
# 7. Report RankIC stats, Newey-West t-stat, quintile spreads, long/short backtest net of costs, importances.
#
# Runtime: expect roughly 20-40 min on a CPU Colab runtime with the defaults (data download is a few minutes of
# that, and is cached). Set `QUICK_RUN = True` for a much shorter smoke test.

# %%
import os, time, shutil, warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:,.4f}".format
pd.options.display.width = 200

# ------------------------------- CONFIG -------------------------------
QUICK_RUN       = False          # True = fewer coins/trials/rounds for a fast smoke test

# data
START_DATE      = "2021-01-01"
INTERVAL        = "4h"
BARS_PER_DAY    = 6
N_CANDIDATES    = 90             # download top-N USDT pairs by current 24h volume (stablecoins/wrapped removed)
MIN_BARS        = 1000           # drop coins with < ~170 days of history
UNIVERSE_TOP_K  = 60             # per bar: keep the K most liquid coins (trailing 30d dollar volume)
MIN_ADV_USDT    = 250_000        # per bar: min trailing mean quote volume per 4H bar
MIN_COINS_PER_TS = 20            # bars with fewer eligible coins are dropped (cross-section too thin)

# target
HORIZON         = 6              # forward return horizon in bars (6 x 4H = 24h)
EXEC_DELAY      = 1              # trade one bar after the signal: target = ret from t+1 to t+1+HORIZON
EMBARGO         = HORIZON + EXEC_DELAY + 2   # bars purged between train and validation/test

# validation
TEST_FRAC       = 0.20           # last 20% of time is the untouched final test
N_FOLDS         = 4              # expanding-window walk-forward folds on the development period
CV_START_FRAC   = 0.40           # first 40% of dev data is the minimum training window
N_TRIALS        = 20             # random-search trials (trial 0 is a conservative baseline)
MAX_ROUNDS      = 500
CKPT_EVERY      = 25             # number of boosting rounds is chosen from these checkpoints
LEARNING_RATE   = 0.05
STD_PENALTY     = 0.5            # selection score = mean fold IC - STD_PENALTY * std across folds
STRIDE          = 3              # train on every 3rd bar (adjacent bars + overlapping targets are near-duplicates)
N_BAGS_FINAL    = 3              # final model = average of STRIDE-offset models, so all data gets used
REFIT_EVERY_DAYS = 60            # in the test period, refit the model every 60 days (expanding window)

# features
USE_MARKET_STATE = True          # add 4 time-only regime features (market return, vol ratio, dispersion, breadth)

# backtest
QUANTILE        = 0.20           # long top 20%, short bottom 20%
COST_BPS        = 5.0            # cost per unit traded notional (fee + slippage), in basis points

USE_GPU         = False          # set True on a GPU runtime (helps only for much larger data)
SEED            = 42
CACHE_DIR       = "/content/crypto_cache"

if QUICK_RUN:
    N_CANDIDATES, N_TRIALS, MAX_ROUNDS, N_FOLDS, REFIT_EVERY_DAYS = 70, 6, 300, 3, 120

NW_LAGS = 2 * HORIZON
os.makedirs(CACHE_DIR, exist_ok=True)
rng_global = np.random.default_rng(SEED)
print("xgboost", xgb.__version__, "| pandas", pd.__version__)

# %% [markdown]
# ## 1. Data download (Binance public market data)
# `data-api.binance.vision` serves public market data and is normally reachable from Colab (the main
# `api.binance.com` host returns HTTP 451 for US IPs). Klines include quote volume, trade count and
# taker-buy volume, which give us order-flow features for free.

# %%
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com",
         "https://api2.binance.com", "https://api3.binance.com", "https://api4.binance.com"]
EXCLUDE_BASES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "AEUR", "EURI", "USDE", "USD1",
                 "XUSD", "BFUSD", "USDS", "PYUSD", "RLUSD", "UST", "WBTC", "WBETH", "BETH", "PAXG", "XAUT", "CBBTC"}
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
         "trades", "tb_base", "tb_quote", "ignore"]
KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "tb_quote"]


def pick_host():
    for h in HOSTS:
        try:
            if requests.get(h + "/api/v3/ping", timeout=10).status_code == 200:
                return h
        except requests.RequestException:
            pass
    raise RuntimeError("No Binance market-data host reachable from this runtime (geo-block / network). "
                       "Try a different Colab runtime/region, or edit HOSTS / swap in another exchange's klines.")


def _get(host, path, params, retries=6):
    for i in range(retries):
        try:
            r = requests.get(host + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                time.sleep(float(r.headers.get("Retry-After", 2 ** i)))
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(1 + i)
    raise RuntimeError(f"request failed: {path} {params}")


def get_candidates(host):
    t = pd.DataFrame(_get(host, "/api/v3/ticker/24hr", {}))
    t = t[t["symbol"].str.endswith("USDT")].copy()
    t["base"] = t["symbol"].str[:-4]
    t = t[~t["base"].isin(EXCLUDE_BASES)]
    t["qv"] = t["quoteVolume"].astype(float)
    return t.sort_values("qv", ascending=False)["symbol"].head(N_CANDIDATES).tolist()


def fetch_klines(host, symbol, start_ms):
    out, cur = [], start_ms
    while True:
        rows = _get(host, "/api/v3/klines", dict(symbol=symbol, interval=INTERVAL, startTime=cur, limit=1000))
        if not rows:
            break
        out.extend(rows)
        cur = rows[-1][0] + 1
        if len(rows) < 1000:
            break
    df = pd.DataFrame(out, columns=KCOLS).drop_duplicates("open_time")
    df = df[df["close_time"] < int(time.time() * 1000)]          # drop the still-open last bar
    for c in KEEP:
        df[c] = df[c].astype(float)
    df.index = pd.DatetimeIndex(pd.to_datetime(df["open_time"].values, unit="ms", utc=True))
    return df[KEEP]


def download_universe(force=False):
    cache = f"{CACHE_DIR}/raw_{INTERVAL}_{START_DATE}_{N_CANDIDATES}.pkl"
    if os.path.exists(cache) and not force:
        print("loading cache", cache)
        return pd.read_pickle(cache)
    host = pick_host()
    print("using host:", host)
    symbols = get_candidates(host)
    start_ms = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)

    def work(sym):
        try:
            return sym, fetch_klines(host, sym, start_ms)
        except Exception as e:
            return sym, e

    raw = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for sym, res in tqdm(ex.map(work, symbols), total=len(symbols), desc="klines"):
            if isinstance(res, Exception):
                print("skip", sym, res)
            else:
                raw[sym] = res
    pd.to_pickle(raw, cache)
    return raw

# %%
raw = download_universe()

# Remove coins with too little history or pegged / non-volatile behaviour (stablecoin-like)
raw = {s: d for s, d in raw.items()
       if len(d) >= MIN_BARS and np.log(d["close"]).diff().std() > 0.003}
print(f"{len(raw)} coins kept after history/volatility filters")
assert len(raw) >= 50, f"Only {len(raw)} usable coins; raise N_CANDIDATES or lower MIN_BARS / move START_DATE later"

# %% [markdown]
# ## 2. Panel construction
# Everything is stored as *wide* frames (time x coin) on a complete 4H grid. Missing bars stay NaN
# (no forward-filling), so any feature/target touching a gap is dropped rather than invented.

# %%
FIELDS = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
          "qvol": "quote_volume", "trades": "trades", "tbq": "tb_quote"}


def make_panel(raw):
    start = min(d.index.min() for d in raw.values())
    end = max(d.index.max() for d in raw.values())
    idx = pd.date_range(start, end, freq="4h", tz="UTC")
    syms = sorted(raw)
    P = {k: pd.DataFrame({s: raw[s][col] for s in syms}).reindex(idx)[syms] for k, col in FIELDS.items()}
    return P


P = make_panel(raw)
idx = P["close"].index
print(f"panel: {P['close'].shape[1]} coins x {len(idx)} bars | {idx[0]} -> {idx[-1]}")
_nn = P["close"].notna()
print(f"data coverage inside each coin's listing window: {_nn.sum().sum() / _nn.cummax().sum().sum():.4f}")

# %% [markdown]
# ## 3. Feature engineering
# * Every feature uses only data up to and including bar *t* (rolling windows look backwards only).
# * Per-coin features are computed on all coins, masked to the **point-in-time universe**, then converted to
#   **cross-sectional percentile ranks in [-0.5, 0.5] at each bar**. This makes features stationary, outlier-proof,
#   and directly aligned with a rank-based objective. (Subtracting the market from a feature is a no-op after ranking.)
# * Target: forward log return from `t+EXEC_DELAY` to `t+EXEC_DELAY+HORIZON`, converted to cross-sectional
#   percentile rank in [-0.5, 0.5]. With a squared-error loss this directly optimises Pearson correlation with
#   the realised *rank*, i.e. the RankIC.
# * Optional time-only "market state" features let trees condition on regime; they cannot change the ordering by
#   themselves, only through interactions. They are the easiest way to overfit to history, so they are few and flagged.

# %%
RET_LB = [1, 2, 3, 6, 12, 24, 42, 90, 180]


def compute_raw_features(P):
    C, Hi, Lo = P["close"], P["high"], P["low"]
    V, QV, NT, TBQ = P["volume"], P["qvol"], P["trades"], P["tbq"]
    logC = np.log(C)
    r1 = logC.diff()
    F = {}

    def rolling(x, w, minfrac=0.8):
        return x.rolling(w, min_periods=int(np.ceil(w * minfrac)))

    # --- momentum / reversal
    for k in RET_LB:
        F[f"ret_{k}"] = logC.diff(k)
    F["mom_42_ex6"] = F["ret_42"] - F["ret_6"]          # medium-term momentum skipping the latest reversal window
    F["mom_180_ex42"] = F["ret_180"] - F["ret_42"]

    # --- volatility & return distribution
    for w in (6, 18, 42, 180):
        F[f"rv_{w}"] = rolling(r1, w).std()
    F["pk_42"] = np.sqrt(rolling(np.log(Hi / Lo) ** 2 / (4 * np.log(2)), 42).mean())   # Parkinson range vol
    F["vr_6_42"] = F["rv_6"] / F["rv_42"]
    F["vr_42_180"] = F["rv_42"] / F["rv_180"]
    F["skew_42"] = rolling(r1, 42).skew()
    F["kurt_42"] = rolling(r1, 42).kurt()
    F["max_ret_42"] = rolling(r1, 42).max()             # "lottery" effect
    F["min_ret_42"] = rolling(r1, 42).min()
    F["downvol_42"] = np.sqrt(rolling(r1.clip(upper=0) ** 2, 42).mean())

    # --- liquidity / participation
    dv = {w: rolling(QV, w).mean() for w in (6, 42, 180)}
    F["dv_chg_6_42"] = dv[6] / dv[42]
    F["dv_chg_42_180"] = dv[42] / dv[180]
    F["size_dv_180"] = dv[180]
    F["amihud_42"] = rolling(r1.abs() / (QV + 1.0), 42).mean()
    F["ntrades_chg"] = rolling(NT, 6).mean() / rolling(NT, 42).mean()
    avg_sz = lambda w: rolling(QV, w).sum() / rolling(NT, w).sum()
    F["trade_size_chg"] = avg_sz(6) / avg_sz(42)

    # --- order flow (taker buy vs sell) and VWAP deviation
    net = 2 * TBQ - QV
    for w in (3, 12, 42):
        F[f"imb_{w}"] = net.rolling(w, min_periods=w).sum() / QV.rolling(w, min_periods=w).sum()
    for w in (18, 90):
        vwap = rolling(QV, w).sum() / rolling(V, w).sum()
        F[f"vwap_dev_{w}"] = C / vwap - 1

    # --- price location
    for w in (42, 180):
        F[f"ma_dist_{w}"] = C / rolling(C, w).mean() - 1
        lo, hi = rolling(Lo, w).min(), rolling(Hi, w).max()
        F[f"donch_{w}"] = (C - lo) / (hi - lo)

    # --- market-relative: rolling beta, idiosyncratic vol, residual momentum
    mkt = r1.mean(axis=1)
    W, mp = 180, 150
    Er = r1.rolling(W, min_periods=mp).mean()
    Em = mkt.rolling(W, min_periods=mp).mean()
    Erm = r1.mul(mkt, axis=0).rolling(W, min_periods=mp).mean()
    Vm = mkt.rolling(W, min_periods=mp).var(ddof=0)
    Vr = r1.rolling(W, min_periods=mp).var(ddof=0)
    beta = (Erm - Er.mul(Em, axis=0)).div(Vm, axis=0)
    F["beta_180"] = beta
    F["idio_vol_180"] = np.sqrt((Vr - beta.pow(2).mul(Vm, axis=0)).clip(lower=0))
    resid = r1 - beta.shift(1).mul(mkt, axis=0)          # beta known one bar earlier
    F["resid_mom_6"] = resid.rolling(6, min_periods=6).sum()
    F["resid_mom_42"] = rolling(resid, 42).sum()

    # --- time-only market-state features (same value for every coin at a given bar)
    disp = r1.std(axis=1)
    M = {
        "mkt_ret_42": rolling(mkt, 42).sum(),
        "mkt_vol_ratio": rolling(mkt, 18).std() / rolling(mkt, 180).std(),
        "disp_ratio": rolling(disp, 18).mean() / rolling(disp, 180).mean(),
        "breadth_42": (F["ret_42"] > 0).astype(float).where(F["ret_42"].notna()).mean(axis=1),
    }

    # --- point-in-time universe: trailing liquidity rank, known at t
    adv = rolling(QV, 180, 0.67).mean()
    in_univ = ((adv.rank(axis=1, ascending=False) <= UNIVERSE_TOP_K) & (adv >= MIN_ADV_USDT)
               & C.notna() & (QV > 0))
    for k in F:
        F[k] = F[k].replace([np.inf, -np.inf], np.nan)
    return F, M, in_univ


def cs_rank(F, in_univ):
    """Cross-sectional percentile rank in [-0.5, 0.5] among universe members at each bar."""
    return {k: v.where(in_univ).rank(axis=1, pct=True) - 0.5 for k, v in F.items()}

# %% [markdown]
# ### 3b. Look-ahead unit test
# Recompute all features on data **truncated** at bar T and compare against the same bars computed from the full
# history. If any feature at bar <= T changes, it was using future data.

# %%
def lookahead_test(P, n_coins=20, cut_frac=0.7):
    cols = list(P["close"].columns[:n_coins])
    Ps = {k: v[cols] for k, v in P.items()}
    cut = int(len(idx) * cut_frac)
    Pt = {k: v.iloc[: cut + 1] for k, v in Ps.items()}
    Ff, Mf, Uf = compute_raw_features(Ps)
    Ft, Mt, Ut = compute_raw_features(Pt)
    bad = []
    for k in Ff:
        a, b = Ff[k].iloc[: cut + 1].values, Ft[k].values
        if not np.allclose(a, b, rtol=1e-3, atol=1e-7, equal_nan=True):
            bad.append(k)
    for k in Mf:
        a, b = Mf[k].iloc[: cut + 1].values, Mt[k].values
        if not np.allclose(a, b, rtol=1e-3, atol=1e-7, equal_nan=True):
            bad.append(k)
    if not (Uf.iloc[: cut + 1].values == Ut.values).all():
        bad.append("universe_mask")
    assert not bad, f"LOOK-AHEAD LEAK in: {bad}"
    print(f"look-ahead test passed: {len(Ff) + len(Mf)} features + universe mask unchanged by truncation at bar {cut}")


lookahead_test(P)

# %% [markdown]
# ### 3c. Assemble the modelling table (one row per coin x bar)

# %%
F, M, in_univ = compute_raw_features(P)
R = cs_rank(F, in_univ)
COIN_FEATS = list(R)
MKT_FEATS = list(M) if USE_MARKET_STATE else []
FEATS = COIN_FEATS + MKT_FEATS

logC = np.log(P["close"])
fwd = logC.shift(-(EXEC_DELAY + HORIZON)) - logC.shift(-EXEC_DELAY)

vmask = (in_univ & fwd.notna()).values.copy()   # copy: .values can be read-only under pandas copy-on-write
for v in R.values():
    vmask &= v.notna().values
if USE_MARKET_STATE:
    vmask &= pd.concat(M, axis=1).notna().all(axis=1).values[:, None]

ti, ci = np.nonzero(vmask)
data = {k: R[k].values[ti, ci].astype(np.float32) for k in COIN_FEATS}
for k in MKT_FEATS:
    data[k] = M[k].values[ti].astype(np.float32)
df = pd.DataFrame(data)
df.insert(0, "tidx", ti.astype(np.int32))
df.insert(1, "sym", pd.Categorical(np.array(P["close"].columns)[ci]))
df["fwd"] = fwd.values[ti, ci].astype(np.float32)

n_per_t = df.groupby("tidx")["fwd"].transform("size")
df = df[n_per_t >= MIN_COINS_PER_TS].reset_index(drop=True)
df["y"] = (df.groupby("tidx")["fwd"].rank(pct=True) - 0.5).astype(np.float32)

# placebo target: y shuffled across coins *within* each bar (destroys all signal, keeps the structure)
_perm = np.lexsort((rng_global.random(len(df)), df["tidx"].values))
df["y_shuf"] = df["y"].values[_perm]
df["ts"] = idx[df["tidx"].values]

print(f"rows: {len(df):,} | distinct coins used: {df['sym'].nunique()} | bars: {df['tidx'].nunique():,} | features: {len(FEATS)}")
print(f"coins per bar: mean {df.groupby('tidx').size().mean():.1f}, min {df.groupby('tidx').size().min()}, max {df.groupby('tidx').size().max()}")
print(f"{df['ts'].min()} -> {df['ts'].max()}")
assert df["sym"].nunique() >= 50

# %% [markdown]
# ## 4. Evaluation utilities
# `FastIC` computes the per-bar Spearman correlation for *all* bars at once (vectorised), which is what makes
# hundreds of CV evaluations cheap. `newey_west_t` corrects the t-stat of the mean IC for autocorrelation caused by
# overlapping forward-return windows.

# %%
class FastIC:
    """Per-bar Spearman(pred, realised-return-rank). Rows must be sorted by tidx."""

    def __init__(self, tidx, y):
        self.t = np.asarray(tidx)
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.t)) + 1]
        self.counts = np.diff(np.r_[self.starts, len(self.t)])
        self.times = self.t[self.starts]
        y = np.asarray(y, dtype=np.float64)
        self.yc = y - np.repeat(np.add.reduceat(y, self.starts) / self.counts, self.counts)
        self.yss = np.add.reduceat(self.yc ** 2, self.starts)

    def __call__(self, pred):
        pr = pd.Series(np.asarray(pred, dtype=np.float64)).groupby(self.t).rank().values
        pc = pr - np.repeat(np.add.reduceat(pr, self.starts) / self.counts, self.counts)
        num = np.add.reduceat(pc * self.yc, self.starts)
        den = np.sqrt(np.add.reduceat(pc ** 2, self.starts) * self.yss)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def newey_west_t(x, lags):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n, e = len(x), x - x.mean()
    var = e @ e / n
    for l in range(1, lags + 1):
        var += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(var / n)


def ic_summary(ic):
    ic = np.asarray(ic, float)
    ic = ic[~np.isnan(ic)]
    return dict(mean_IC=ic.mean(), std_IC=ic.std(ddof=1), ICIR=ic.mean() / ic.std(ddof=1),
                hit_rate=(ic > 0).mean(), NW_t=newey_west_t(ic, NW_LAGS), n_bars=len(ic))


def quantile_returns(d, pred, n=5):
    """Mean forward return (bps, in excess of the cross-sectional mean) by predicted quintile."""
    t = d["tidx"].values
    pr = pd.Series(np.asarray(pred, float)).groupby(t).rank(pct=True).values
    q = np.minimum(np.ceil(pr * n).astype(int) - 1, n - 1)
    r = d["fwd"].values.astype(float)
    r_ex = r - pd.Series(r).groupby(t).transform("mean").values
    tab = pd.DataFrame({"t": t, "q": q, "r": r_ex}).groupby(["t", "q"])["r"].mean().unstack()
    res = tab.mean() * 1e4
    res.index = [f"Q{int(i) + 1}" for i in res.index]
    return res


def long_short_backtest(d, pred, q=QUANTILE, cost_bps=COST_BPS):
    """Dollar-neutral top-q / bottom-q equal-weight portfolio, entered every bar with 1/HORIZON of capital
    (i.e. HORIZON staggered sub-books, each rebalanced every HORIZON bars). Costs charged on traded notional."""
    d = d[["tidx", "sym", "fwd"]].copy()
    d["sym"] = d["sym"].astype(str)
    d["p"] = np.asarray(pred, float)
    d["pr"] = d.groupby("tidx")["p"].rank(pct=True)
    d["L"] = d["pr"] > 1 - q
    d["S"] = d["pr"] <= q
    nl = d.groupby("tidx")["L"].transform("sum").clip(lower=1)
    ns = d.groupby("tidx")["S"].transform("sum").clip(lower=1)
    d["w"] = np.where(d["L"], 1.0 / nl, 0.0) - np.where(d["S"], 1.0 / ns, 0.0)
    gross = (d["w"] * d["fwd"]).groupby(d["tidx"]).sum()
    Wt = d.pivot(index="tidx", columns="sym", values="w").fillna(0.0)
    Wt = Wt.reindex(range(int(Wt.index.min()), int(Wt.index.max()) + 1)).fillna(0.0)
    turnover = (Wt - Wt.shift(HORIZON).fillna(0.0)).abs().sum(axis=1)
    bt = pd.DataFrame({"gross": gross.reindex(Wt.index), "turnover": turnover})
    bt["cost"] = bt["turnover"] * cost_bps / 1e4
    bt["net"] = bt["gross"] - bt["cost"]
    return bt.dropna()


def bt_metrics(bt):
    ppy = BARS_PER_DAY * 365 / HORIZON
    out = {}
    for col in ("gross", "net"):
        sh = [bt[col].iloc[o::HORIZON].mean() / bt[col].iloc[o::HORIZON].std() * np.sqrt(ppy) for o in range(HORIZON)]
        eq = (bt[col] / HORIZON).cumsum()
        years = (bt.index[-1] - bt.index[0] + 1) / (BARS_PER_DAY * 365)
        out[col] = dict(sharpe=np.mean(sh), ann_return=eq.iloc[-1] / years, max_drawdown=(eq - eq.cummax()).min())
    out["avg_turnover_per_rebalance"] = bt["turnover"].mean()
    return out

# %% [markdown]
# ## 5. Splits: untouched test set + purged walk-forward CV on the development period
# ```
# time ->  |---------------- development ----------------|--embargo--|------ test (final) ------|
#          |train-----|emb|val1|        expanding folds, each purged by EMBARGO bars
# ```
# Training rows end `EMBARGO` bars before each validation/test block so overlapping forward-return windows can't leak.

# %%
ut = np.unique(df["tidx"].values)
test_start = int(ut[int(len(ut) * (1 - TEST_FRAC))])
dev_end = test_start - EMBARGO
dev = df[df["tidx"] < dev_end].reset_index(drop=True)
test_i0 = int(np.searchsorted(df["tidx"].values, test_start))
test = df.iloc[test_i0:].reset_index(drop=True)
print(f"dev : {idx[dev['tidx'].min()]} -> {idx[dev['tidx'].max()]}  ({len(dev):,} rows)")
print(f"test: {idx[test['tidx'].min()]} -> {idx[test['tidx'].max()]}  ({len(test):,} rows)")

dev_t = np.unique(dev["tidx"].values)
edges = np.linspace(CV_START_FRAC, 1.0, N_FOLDS + 1)
bounds = [int(dev_t[min(int(e * len(dev_t)), len(dev_t) - 1)]) for e in edges[:-1]] + [int(dev_t[-1]) + 1]

BASE = dict(objective="reg:squarederror", tree_method="hist", max_bin=128, learning_rate=LEARNING_RATE,
            base_score=0.0, verbosity=0)
if USE_GPU:
    BASE["device"] = "cuda"

folds = []
for k in range(N_FOLDS):
    v0, v1 = bounds[k], bounds[k + 1]
    tr = dev[(dev["tidx"] < v0 - EMBARGO) & (dev["tidx"] % STRIDE == 0)]
    va = dev[(dev["tidx"] >= v0) & (dev["tidx"] < v1)]
    folds.append(dict(
        dtrain=xgb.DMatrix(tr[FEATS].values, label=tr["y"].values, feature_names=FEATS),
        dval=xgb.DMatrix(va[FEATS].values, feature_names=FEATS),
        fic=FastIC(va["tidx"].values, va["y"].values),
        span=f"{idx[v0].date()} -> {idx[min(v1, len(idx)) - 1].date()}", n_train=len(tr), n_val=len(va)))
    print(f"fold {k + 1}: train {len(tr):,} rows (strided) | val {len(va):,} rows | {folds[-1]['span']}")

# %% [markdown]
# ## 6. Hyper-parameter search (random search, constrained to *strongly regularised* shallow trees)
# * Signal-to-noise in returns is tiny, so the search space only contains shallow trees, large `min_child_weight`
#   (= minimum rows per leaf, since hessian = 1 for squared error), row/column subsampling, L1/L2 and `gamma`.
# * The number of boosting rounds is treated as a hyper-parameter chosen from checkpoints
#   (no per-fold early stopping on the validation fold, which would leak it into model selection).
# * Selection score = mean fold RankIC - `STD_PENALTY` x std across folds, so unstable configs lose.

# %%
CKPTS = list(range(CKPT_EVERY, MAX_ROUNDS + 1, CKPT_EVERY))


def sample_params(rng):
    return dict(
        max_depth=int(rng.choice([2, 3, 4, 5, 6])),
        min_child_weight=float(rng.choice([100, 300, 1000, 3000])),
        subsample=float(rng.choice([0.5, 0.7, 0.9])),
        colsample_bytree=float(rng.choice([0.3, 0.5, 0.7, 0.9])),
        reg_lambda=float(np.exp(rng.uniform(np.log(1), np.log(500)))),
        reg_alpha=float(rng.choice([0.0, 0.5, 2.0, 8.0])),
        gamma=float(rng.choice([0.0, 0.05, 0.2, 1.0])),
    )


def run_cv(params):
    curve = np.full((len(folds), len(CKPTS)), np.nan)
    for i, f in enumerate(folds):
        bst = xgb.train({**BASE, **params, "seed": SEED}, f["dtrain"], num_boost_round=MAX_ROUNDS)
        for j, k in enumerate(CKPTS):
            curve[i, j] = np.nanmean(f["fic"](bst.predict(f["dval"], iteration_range=(0, k))))
    return curve


BASELINE_PARAMS = dict(max_depth=3, min_child_weight=1000.0, subsample=0.7, colsample_bytree=0.5,
                       reg_lambda=50.0, reg_alpha=0.0, gamma=0.0)
rng = np.random.default_rng(SEED)
candidates = [BASELINE_PARAMS] + [sample_params(rng) for _ in range(N_TRIALS - 1)]

trial_log = []
t0 = time.time()
for n, p in enumerate(candidates):
    curve = run_cv(p)
    score = curve.mean(0) - STD_PENALTY * curve.std(0)
    j = int(np.nanargmax(score))
    trial_log.append(dict(params=p, curve=curve, j=j, rounds=CKPTS[j], score=score[j],
                          mean_ic=curve[:, j].mean(), std_ic=curve[:, j].std()))
    print(f"trial {n + 1:2d}/{len(candidates)} | depth {p['max_depth']} mcw {p['min_child_weight']:>6.0f} "
          f"sub {p['subsample']:.1f} col {p['colsample_bytree']:.1f} lam {p['reg_lambda']:6.1f} "
          f"| rounds {CKPTS[j]:3d} | CV IC {trial_log[-1]['mean_ic']:+.4f} +/- {trial_log[-1]['std_ic']:.4f} "
          f"| score {score[j]:+.4f} | {time.time() - t0:.0f}s")

best = max(trial_log, key=lambda r: r["score"])
BEST_PARAMS, BEST_ROUNDS = best["params"], best["rounds"]
print("\nBEST:", {k: round(v, 3) if isinstance(v, float) else v for k, v in BEST_PARAMS.items()}, "| rounds", BEST_ROUNDS)
print("fold ICs at chosen config:", np.round(best["curve"][:, best["j"]], 4))
if BEST_ROUNDS == CKPTS[-1]:
    print("NOTE: best rounds is the maximum checkpoint - the model may be under-trained; consider raising MAX_ROUNDS.")
if best["mean_ic"] <= 0:
    print("WARNING: CV RankIC <= 0. There is no exploitable signal in this setup; don't expect the test to differ.")

# %% [markdown]
# ## 7. Final evaluation on the untouched test period
# The test set is used **once**, with the configuration frozen above. Models are refit every `REFIT_EVERY_DAYS`
# on an expanding window (purged by `EMBARGO`), mimicking live use. Each model is a bag of `N_BAGS_FINAL`
# XGBoosts trained on different bar-offsets (`tidx % STRIDE`), so all data is used while hyper-parameters stay
# consistent with what was validated.

# %%
def xgb_fit(train, ycol="y", params=None, rounds=None, n_bags=N_BAGS_FINAL):
    params = params or BEST_PARAMS
    rounds = rounds or BEST_ROUNDS
    models = []
    for b in range(n_bags):
        sub = train[train["tidx"] % STRIDE == b]
        d = xgb.DMatrix(sub[FEATS].values, label=sub[ycol].values, feature_names=FEATS)
        models.append(xgb.train({**BASE, **params, "seed": SEED + b}, d, num_boost_round=rounds))
    return models


def xgb_predict(models, X):
    X = X[FEATS].values if isinstance(X, pd.DataFrame) else X
    d = xgb.DMatrix(X, feature_names=FEATS)
    return np.mean([m.predict(d) for m in models], axis=0)


def walk_forward(fit_fn, pred_fn, refit_bars):
    t = df["tidx"].values
    out = np.full(len(df), np.nan)
    for s in range(test_start, int(t.max()) + 1, refit_bars):
        lo, hi = np.searchsorted(t, s), np.searchsorted(t, s + refit_bars)
        if hi == lo:
            continue
        tr = df.iloc[: np.searchsorted(t, s - EMBARGO)]
        out[lo:hi] = pred_fn(fit_fn(tr), df.iloc[lo:hi])
    return out[test_i0:]


refit_bars = REFIT_EVERY_DAYS * BARS_PER_DAY
test_ic = FastIC(test["tidx"].values, test["y"].values)
dev_ic = FastIC(dev["tidx"].values, dev["y"].values)

print("walk-forward XGBoost ...")
pred_xgb_wf = walk_forward(xgb_fit, xgb_predict, refit_bars)

print("frozen XGBoost (trained once on dev) ...")
models_frozen = xgb_fit(dev)
pred_xgb_frozen = xgb_predict(models_frozen, test)
pred_xgb_insample = xgb_predict(models_frozen, dev)

print("Ridge baseline (walk-forward, per-coin features only) ...")
pred_ridge = walk_forward(
    lambda tr: Ridge(alpha=1000.0).fit(tr[COIN_FEATS].values, tr["y"].values),
    lambda m, d: m.predict(d[COIN_FEATS].values), refit_bars)

print("single-factor baseline (feature + sign chosen on dev only) ...")
dev_factor_ic = pd.Series({f: np.nanmean(dev_ic(dev[f].values)) for f in COIN_FEATS})
best_factor = dev_factor_ic.abs().idxmax()
sign = np.sign(dev_factor_ic[best_factor])
pred_factor = sign * test[best_factor].values

print("placebo: train on within-bar shuffled targets ...")
models_placebo = xgb_fit(dev, ycol="y_shuf")
pred_placebo = xgb_predict(models_placebo, test)

# %% [markdown]
# ## 8. Results

# %%
ics = {
    "XGB walk-forward (TEST)":        test_ic(pred_xgb_wf),
    "XGB frozen (TEST)":              test_ic(pred_xgb_frozen),
    "Ridge walk-forward (TEST)":      test_ic(pred_ridge),
    f"best dev factor: {'-' if sign < 0 else '+'}{best_factor} (TEST)": test_ic(pred_factor),
    "PLACEBO shuffled labels (TEST)": test_ic(pred_placebo),
    "XGB in-sample (DEV)":            dev_ic(pred_xgb_insample),
}
summ = pd.DataFrame({k: ic_summary(v) for k, v in ics.items()}).T
print("RankIC summary (per 4H bar; forward window = %d bars, NW lags = %d)" % (HORIZON, NW_LAGS))
print(summ.to_string())

print("\nCross-validation RankIC per fold (chosen config):")
for k, f in enumerate(folds):
    print(f"  fold {k + 1} [{f['span']}]: {best['curve'][k, best['j']]:+.4f}")
print(f"  CV mean {best['mean_ic']:+.4f} | TEST (walk-forward) {summ.iloc[0]['mean_IC']:+.4f}")

fac = pd.DataFrame({"dev_IC": dev_factor_ic,
                    "test_IC": pd.Series({f: np.nanmean(test_ic(test[f].values)) for f in COIN_FEATS})})
fac = fac.reindex(dev_factor_ic.abs().sort_values(ascending=False).index)
print("\nSingle-factor RankIC, top 10 on dev (watch dev -> test decay):")
print(fac.head(10).to_string())

# ----- overfitting diagnostics
wf, ins, plc, rdg = (summ.iloc[0], summ.iloc[5], summ.iloc[4], summ.iloc[2])
print("\n=== overfitting / leakage checks ===")
print(f"in-sample IC {ins['mean_IC']:+.4f}  vs  CV IC {best['mean_ic']:+.4f}  vs  test IC {wf['mean_IC']:+.4f}")
if ins["mean_IC"] > 4 * max(wf["mean_IC"], 1e-4):
    print(" - large in-sample/out-of-sample gap -> expected for flexible models; trust only the TEST number")
if abs(plc["NW_t"]) > 2.5:
    print(" - !! placebo model shows 'significant' IC: investigate leakage (should be ~0)")
else:
    print(f" - placebo IC {plc['mean_IC']:+.4f} (NW t {plc['NW_t']:+.1f}): consistent with zero, pipeline looks leak-free")
if wf["mean_IC"] <= rdg["mean_IC"]:
    print(" - XGBoost does not beat Ridge on the test: the non-linear model isn't earning its complexity")
if wf["NW_t"] < 2:
    print(f" - test RankIC NW t-stat {wf['NW_t']:.1f} < 2: not statistically distinguishable from zero")
if best["curve"][:, best["j"]].min() < 0:
    print(" - at least one CV fold has negative IC: signal is regime-dependent")

# %%
# ----- permutation importance on the test set (global shuffle of one column at a time, frozen model, bag 0)
sub = test[test["tidx"] % 3 == 0].reset_index(drop=True)
sub_ic = FastIC(sub["tidx"].values, sub["y"].values)
Xs = sub[FEATS].values.copy()
base_ic = np.nanmean(sub_ic(xgb_predict(models_frozen[:1], Xs)))
rng_p = np.random.default_rng(SEED)
perm_imp = {}
for j, f in enumerate(FEATS):
    Xp = Xs.copy()
    Xp[:, j] = rng_p.permutation(Xp[:, j])
    perm_imp[f] = base_ic - np.nanmean(sub_ic(xgb_predict(models_frozen[:1], Xp)))
gain = pd.concat([pd.Series(m.get_score(importance_type="gain")) for m in models_frozen], axis=1).mean(axis=1)
imp = pd.DataFrame({"perm_IC_drop": pd.Series(perm_imp), "gain": gain.reindex(FEATS).fillna(0.0)})
imp = imp.sort_values("perm_IC_drop", ascending=False)
print("Top features (permutation = drop in test RankIC when the feature is shuffled):")
print(imp.head(15).to_string())

# ----- portfolio view (walk-forward predictions)
qr = quantile_returns(test, pred_xgb_wf)
bt = long_short_backtest(test, pred_xgb_wf)
btm = bt_metrics(bt)
print(f"\nQuintile mean forward return, bps per {HORIZON * 4}h, excess over cross-sectional mean (Q1 = lowest score):")
print(qr.to_string(float_format="{:.1f}".format))
print(f"\nLong/short top/bottom {int(QUANTILE * 100)}% | cost {COST_BPS} bps per unit traded notional")
print(pd.DataFrame({k: btm[k] for k in ("gross", "net")}).to_string())
print(f"avg turnover per rebalance (sum |dw|, 4.0 = full rotation): {btm['avg_turnover_per_rebalance']:.2f}")

# %%
fig, ax = plt.subplots(3, 2, figsize=(16, 14))
tt = idx[test_ic.times]

for name, col in [("XGB walk-forward", ics["XGB walk-forward (TEST)"]), ("XGB frozen", ics["XGB frozen (TEST)"]),
                  ("Ridge", ics["Ridge walk-forward (TEST)"]), ("Placebo", ics["PLACEBO shuffled labels (TEST)"])]:
    ax[0, 0].plot(tt, np.nancumsum(col), label=name, lw=2 if "walk" in name else 1)
ax[0, 0].axhline(0, c="k", lw=.5)
ax[0, 0].set_title("Cumulative RankIC on the untouched test period"); ax[0, 0].legend()

fold_ic = best["curve"][:, best["j"]]
labels = [f"CV{k + 1}" for k in range(len(fold_ic))] + ["TEST wf", "TEST frozen"]
vals = list(fold_ic) + [summ.iloc[0]["mean_IC"], summ.iloc[1]["mean_IC"]]
ax[0, 1].bar(labels, vals, color=["tab:blue"] * len(fold_ic) + ["tab:green", "tab:olive"])
ax[0, 1].axhline(0, c="k", lw=.5); ax[0, 1].set_title("Mean RankIC by CV fold and test (stability check)")

c = best["curve"]
ax[1, 0].plot(CKPTS, c.mean(0), label="mean over folds", lw=2)
ax[1, 0].fill_between(CKPTS, c.mean(0) - c.std(0), c.mean(0) + c.std(0), alpha=.25)
for k in range(c.shape[0]):
    ax[1, 0].plot(CKPTS, c[k], lw=.7, alpha=.6)
ax[1, 0].axvline(BEST_ROUNDS, c="r", ls="--", label=f"chosen rounds = {BEST_ROUNDS}")
ax[1, 0].set_title("CV RankIC vs boosting rounds (best config)"); ax[1, 0].legend(); ax[1, 0].set_xlabel("rounds")

ax[1, 1].bar([f"Q{i + 1}" for i in range(len(qr))], qr.values, color="tab:purple")
ax[1, 1].axhline(0, c="k", lw=.5)
ax[1, 1].set_title(f"Test: forward return by predicted quintile (bps / {HORIZON * 4}h, excess)")

eq_g, eq_n = (bt["gross"] / HORIZON).cumsum(), (bt["net"] / HORIZON).cumsum()
bt_t = idx[bt.index.values]
ax[2, 0].plot(bt_t, eq_g.values, label="gross"); ax[2, 0].plot(bt_t, eq_n.values, label=f"net ({COST_BPS} bps)")
ax[2, 0].axhline(0, c="k", lw=.5); ax[2, 0].legend()
ax[2, 0].set_title("Test: L/S quintile portfolio, cumulative return (arithmetic, 1/H capital per entry bar)")

top = imp.head(15)[::-1]
ax[2, 1].barh(top.index, top["perm_IC_drop"], color="tab:orange")
ax[2, 1].set_title("Permutation importance (drop in test RankIC)")
plt.tight_layout()
plt.savefig(f"{CACHE_DIR}/results.png", dpi=120)
plt.show()

out = test[["ts", "sym", "fwd", "y"]].copy()
out["pred_xgb_wf"] = pred_xgb_wf
out.to_csv(f"{CACHE_DIR}/test_predictions.csv", index=False)
print("saved:", f"{CACHE_DIR}/results.png", f"{CACHE_DIR}/test_predictions.csv")

# %% [markdown]
# ## Reading the results honestly
# * **Typical magnitudes:** liquid-crypto cross-sectional RankIC at 4H-24h horizons is small (roughly 0.01-0.04).
#   An IC of 0.02 with ICIR ~0.1 per bar can still be a decent signal, but it is only trustworthy if the
#   Newey-West t-stat on the *test* period is comfortably above 2 and the CV folds all agree in sign.
# * **Costs dominate:** check the `net` vs `gross` rows. If turnover is high the edge can vanish; try a longer
#   `HORIZON`, a lower rebalance frequency, or a turnover penalty before believing any Sharpe.
# * **Known limitations:** (1) the coin list is today's top-volume coins, so delisted/dead coins are missing
#   (survivorship bias flatters long-side results; the point-in-time liquidity filter only partly mitigates it);
#   (2) shorting/funding costs and market impact are not modelled; (3) you tried `N_TRIALS` configs on the CV data,
#   so the CV score is optimistic - only the test block is clean, and once you look at it and iterate, it is no
#   longer clean. For further work, use a fresh later period or a second, differently-seeded holdout.
