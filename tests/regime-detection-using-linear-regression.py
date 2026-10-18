# # BTC 4H Regime Detector: TREND vs SIDEWAYS/CHOPPY (Linear Regression)
#
# **What the model does:** at the close of each 4H bar it predicts how *directional* the next `H` bars will be.
#
# **Target (continuous):** forward Efficiency Ratio = |net move| / total path travelled over the next `H` bars.
# 1.0 = straight line (clean trend, up or down), near 0 = lots of movement but going nowhere (chop).
#
# **Label (binary):** TREND if forward ER is in the top (1-`TREND_Q`) of the *training* distribution, else RANGE.
# Bullish and bearish trends are both "TREND". Direction is NOT scored.
#
# **Evaluation:** purged walk-forward (expanding window, never trains on data whose label window overlaps the test period).
# We report accuracy overall, **precision when it says TREND**, **precision when it says RANGE**, AUC, and a naive baseline.

# %%
import subprocess, sys, warnings
for pkg in ["ccxt"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", pkg],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             precision_recall_fscore_support, matthews_corrcoef,
                             ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

# ----------------------------- CONFIG ---------------------------------------
SYMBOL      = "BTC/USDT"
TIMEFRAME   = "4h"
START_DATE  = "2019-01-01"
CSV_PATH    = None            # optional: your own 4H CSV (timestamp,open,high,low,close,volume)

H           = 12              # look-ahead window in bars (12 x 4h = 2 days)
TREND_Q     = 0.60            # top 40% of forward ER = "clear trend"
INIT_TRAIN_BARS = 6 * 365 * 3 # first training window: ~3 years of 4H bars
STEP_BARS       = 6 * 365 // 2 # re-train every ~6 months
RUN_SENSITIVITY = True        # grid over H and TREND_Q (takes a minute)

# %%
# ----------------------------- DATA -----------------------------------------
def _clean(df):
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)

def fetch_ccxt(exchange_id, symbol, timeframe, start_date):
    import ccxt
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.parse8601(f"{start_date}T00:00:00Z")
    now_ms = ex.milliseconds()
    rows = []
    while since < now_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        nxt = batch[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
    if not rows:
        raise ValueError("no data returned")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    # drop the still-forming candle so we never use unfinished data
    df = df[df.index + pd.Timedelta(milliseconds=tf_ms) <= pd.Timestamp.now(tz="UTC")]
    return _clean(df)

def fetch_yfinance():
    import yfinance as yf
    raw = yf.download("BTC-USD", interval="1h", period="730d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index, utc=True)
    out = raw.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()
    return _clean(out)

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = "timestamp" if "timestamp" in df.columns else df.columns[0]
    ts = df[tcol]
    if pd.api.types.is_numeric_dtype(ts):
        df[tcol] = pd.to_datetime(ts, unit="ms" if ts.iloc[0] > 1e11 else "s", utc=True)
    else:
        df[tcol] = pd.to_datetime(ts, utc=True)
    return _clean(df.set_index(tcol))

def load_data():
    if CSV_PATH:
        return load_csv(CSV_PATH)
    for ex_id in ["binance", "binanceus", "kucoin", "okx", "bybit"]:
        try:
            df = fetch_ccxt(ex_id, SYMBOL, TIMEFRAME, START_DATE)
            if len(df) > 3000:
                print(f"Loaded {len(df):,} bars from {ex_id}")
                return df
        except Exception as e:
            print(f"  {ex_id} failed: {type(e).__name__}")
    print("All exchanges failed -> falling back to yfinance (only ~2 years of 4H bars)")
    return fetch_yfinance()

df = load_data()
print(df.index[0], "->", df.index[-1], f"| {len(df):,} bars")
df.tail(3)

# %%
# ----------------------------- FEATURES + TARGET ----------------------------
def er_back(logc, n):
    """Trailing Efficiency Ratio over n bars (uses only past data)."""
    net = (logc - logc.shift(n)).abs()
    path = logc.diff().abs().rolling(n).sum()
    return net / path.replace(0, np.nan)

def build_features(df):
    """Every feature uses data up to and including the CURRENT bar's close only."""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    logc = np.log(c)
    r = logc.diff()
    X = pd.DataFrame(index=df.index)

    # trend efficiency / linearity of the recent path
    for n in (10, 20, 50):
        X[f"er_{n}"] = er_back(logc, n)
    t = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    for n in (20, 50):
        X[f"r2_{n}"] = logc.rolling(n).corr(t) ** 2

    # ADX (Wilder)
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    kw = dict(alpha=1 / 14, adjust=False, min_periods=14)
    atr = tr.ewm(**kw).mean()
    pdi = 100 * plus_dm.ewm(**kw).mean() / atr
    mdi = 100 * minus_dm.ewm(**kw).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    X["adx"] = dx.ewm(**kw).mean()

    # choppiness index
    for n in (14, 28):
        rng = h.rolling(n).max() - l.rolling(n).min()
        X[f"chop_{n}"] = 100 * np.log10(tr.rolling(n).sum() / rng) / np.log10(n)

    # volatility level / expansion
    X["atr_pct"] = atr / c
    X["vol_ratio"] = r.rolling(10).std() / r.rolling(50).std()
    sma, sd = c.rolling(20).mean(), c.rolling(20).std()
    bbw = 4 * sd / sma
    X["bbw"] = bbw
    X["bbw_rel"] = bbw / bbw.rolling(120).mean()

    # EMA structure (in ATR units)
    ema20, ema50 = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    X["ema_spread"] = (ema20 - ema50).abs() / atr
    X["ema50_slope"] = (ema50 - ema50.shift(10)).abs() / atr
    X["dist_ema50"] = (c - ema50).abs() / atr

    # size of recent moves relative to normal volatility
    sig = r.rolling(50).std()
    for n in (6, 12, 24):
        X[f"zret_{n}"] = (logc - logc.shift(n)).abs() / (sig * np.sqrt(n))

    # whipsaw rate + volume
    X["flip_20"] = (np.sign(r) != np.sign(r.shift())).astype(float).rolling(20).mean()
    X["vol_rel"] = np.log(v / v.rolling(20).mean())

    dir_hint = np.sign(ema20 - ema50)      # display only, NOT a model input, NOT scored
    return X.replace([np.inf, -np.inf], np.nan), logc, dir_hint

EXTRA_COLS = ["er_fwd", "er_base", "dir_hint", "close"]

def make_dataset(df, H):
    X, logc, dir_hint = build_features(df)
    d = X.copy()
    # TARGET: forward Efficiency Ratio over bars t+1 ... t+H
    path_fwd = logc.diff().abs().rolling(H).sum().shift(-H)
    net_fwd = (logc.shift(-H) - logc).abs()
    d["er_fwd"] = net_fwd / path_fwd.replace(0, np.nan)
    d["er_base"] = er_back(logc, H)        # naive baseline: "regime persists"
    d["dir_hint"] = dir_hint
    d["close"] = df["close"]
    return d

data = make_dataset(df, H)
FEATURES = [c for c in data.columns if c not in EXTRA_COLS]
print(f"{len(FEATURES)} features:", FEATURES)

# %%
# ----------------------------- WALK-FORWARD ENGINE --------------------------
def walk_forward(d, H, q, init_train, step):
    feats = [c for c in d.columns if c not in EXTRA_COLS]
    lab = d.dropna(subset=feats + ["er_fwd", "er_base"]).copy()
    n = len(lab)
    init_train = min(init_train, int(n * 0.5))
    step = min(step, max(int(n * 0.1), 50))

    out, fold = [], 0
    for s in range(init_train, n, step):
        e = min(s + step, n)
        te = lab.iloc[s:e]
        if len(te) < 20:
            continue
        tr = lab.iloc[: s - H]          # PURGE: drop the last H train rows (their labels peek into the test period)

        thr = tr["er_fwd"].quantile(q)  # "clear trend" cut-off, learned from TRAIN only
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(tr[feats], tr["er_fwd"])
        s_tr, s_te = model.predict(tr[feats]), model.predict(te[feats])
        cut = np.quantile(s_tr, q)      # same base rate as the labels, no tuning on test
        base_cut = tr["er_base"].quantile(q)

        out.append(pd.DataFrame({
            "fold": fold,
            "er_fwd": te["er_fwd"].values,
            "er_thr": thr,
            "y_true": (te["er_fwd"].values >= thr).astype(int),
            "score": s_te,
            "y_pred": (s_te >= cut).astype(int),
            "base_score": te["er_base"].values,
            "base_pred": (te["er_base"].values >= base_cut).astype(int),
            "dir_hint": te["dir_hint"].values,
        }, index=te.index))
        fold += 1
    return pd.concat(out)

def evaluate(y_true, y_pred, score=None):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1, 0], zero_division=0)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "precision_TREND": p[0],   # when it says TREND, how often is it right?
        "precision_RANGE": p[1],   # when it says RANGE, how often is it right?
        "recall_TREND": r[0],
        "recall_RANGE": r[1],
        "F1_TREND": f[0],
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        m["AUC"] = roc_auc_score(y_true, score)
    return m

def block_bootstrap_ci(y_true, y_pred, stat, block=24, n_boot=1000, seed=42):
    """Labels overlap in time, so resample in blocks (not single bars) to get honest CIs."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    nb = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        st = rng.integers(0, n - block + 1, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        vals.append(stat(y_true[idx], y_pred[idx]))
    return np.percentile(vals, [2.5, 97.5])

# %%
# ----------------------------- RUN + REPORT ---------------------------------
oos = walk_forward(data, H, TREND_Q, INIT_TRAIN_BARS, STEP_BARS)

m_lr = evaluate(oos.y_true, oos.y_pred, oos.score)
m_bs = evaluate(oos.y_true, oos.base_pred, oos.base_score)
base_rate = oos.y_true.mean()

print(f"OOS period : {oos.index[0]} -> {oos.index[-1]}  ({len(oos):,} bars, {oos.fold.nunique()} folds)")
print(f"Setup      : look-ahead H={H} bars ({H*4}h), TREND = top {(1-TREND_Q):.0%} of forward ER")
print(f"Base rate  : {base_rate:.1%} of OOS bars were actually TREND")
print(f"Majority-class accuracy (dumb baseline): {max(base_rate, 1-base_rate):.1%}\n")

print(pd.DataFrame({"Linear Regression": m_lr, "Baseline (trailing ER persists)": m_bs}).round(3))

print("\n95% block-bootstrap confidence intervals (Linear Regression):")
for name, fn in [("accuracy", accuracy_score),
                 ("balanced_acc", balanced_accuracy_score),
                 ("precision_TREND", lambda a, b: precision_recall_fscore_support(a, b, labels=[1], zero_division=0)[0][0]),
                 ("precision_RANGE", lambda a, b: precision_recall_fscore_support(a, b, labels=[0], zero_division=0)[0][0])]:
    lo, hi = block_bootstrap_ci(oos.y_true, oos.y_pred, fn, block=max(2 * H, 24))
    print(f"  {name:16s} [{lo:.3f}, {hi:.3f}]")

# Does a higher score really mean a more trending future?
oos["bucket"] = pd.qcut(oos["score"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
lift = oos.groupby("bucket", observed=True).agg(bars=("y_true", "size"),
                                                actual_TREND_rate=("y_true", "mean"),
                                                mean_forward_ER=("er_fwd", "mean")).round(3)
print("\nPredicted-score quintiles vs what actually happened (should rise monotonically):")
print(lift)

per_fold = oos.groupby("fold").apply(lambda g: pd.Series({
    "start": g.index[0].date(), "bars": len(g),
    "base_rate": g.y_true.mean(),
    "acc_LR": accuracy_score(g.y_true, g.y_pred),
    "acc_baseline": accuracy_score(g.y_true, g.base_pred)})).round(3)
print("\nPer-fold (is the edge stable through time?):")
print(per_fold)

# %%
# ----------------------------- PLOTS ----------------------------------------
def plot_results(oos, close, n_last=1200):
    seg = oos.iloc[-n_last:]
    px = close.reindex(seg.index)
    fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True,
                           gridspec_kw={"height_ratios": [4, 0.6, 0.6]})
    ax[0].plot(px.index, px.values, lw=1, color="black")
    ax[0].set_yscale("log")
    ax[0].set_title(f"BTC 4H, last {len(seg)} out-of-sample bars: actual vs predicted regime")
    for a, col, name in ((ax[1], "y_true", "Actual"), (ax[2], "y_pred", "Predicted")):
        m = seg[col].values == 1
        a.fill_between(seg.index, 0, 1, where=m, step="mid", color="tab:green", alpha=0.65)
        a.fill_between(seg.index, 0, 1, where=~m, step="mid", color="tab:gray", alpha=0.35)
        a.set_yticks([])
        a.set_ylabel(name, rotation=0, ha="right", va="center")
    ax[0].text(0.01, 0.95, "green = TREND   grey = SIDEWAYS/CHOP", transform=ax[0].transAxes, va="top")
    plt.tight_layout(); plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        oos.y_true, oos.y_pred, display_labels=["RANGE", "TREND"], normalize="pred",
        values_format=".2f", cmap="Blues", colorbar=False, ax=ax[0])
    ax[0].set_title("Each column = what the model said;\nvalue = share of those calls that were right")
    pf = oos.groupby("fold").apply(lambda g: pd.Series({
        "Linear Regression": accuracy_score(g.y_true, g.y_pred),
        "Baseline": accuracy_score(g.y_true, g.base_pred),
        "Majority class": max(g.y_true.mean(), 1 - g.y_true.mean())}))
    pf.plot.bar(ax=ax[1]); ax[1].set_ylim(0.4, 1); ax[1].set_title("Accuracy by walk-forward fold")
    ax[1].set_xlabel("fold"); ax[1].legend(loc="lower right")
    plt.tight_layout(); plt.show()

plot_results(oos, data["close"])

# %%
# ----------------------------- SENSITIVITY (optional) -----------------------
# NOTE: if you pick the best cell of this grid, you're data-mining. Use it to check
# the result is STABLE across settings, not to cherry-pick one.
if RUN_SENSITIVITY:
    rows = []
    for H_ in (6, 12, 18, 24):
        d_ = make_dataset(df, H_)
        for q_ in (0.5, 0.6, 0.7):
            o_ = walk_forward(d_, H_, q_, INIT_TRAIN_BARS, STEP_BARS)
            a = evaluate(o_.y_true, o_.y_pred, o_.score)
            b = evaluate(o_.y_true, o_.base_pred, o_.base_score)
            rows.append({"H": H_, "TREND_Q": q_, "base_rate": o_.y_true.mean(),
                         "acc": a["accuracy"], "bal_acc": a["balanced_acc"],
                         "prec_TREND": a["precision_TREND"], "prec_RANGE": a["precision_RANGE"],
                         "AUC": a["AUC"], "AUC_baseline": b["AUC"]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

# %%
# ----------------------------- CURRENT REGIME -------------------------------
def current_regime(d, H, q):
    feats = [c for c in d.columns if c not in EXTRA_COLS]
    lab = d.dropna(subset=feats + ["er_fwd"])
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(lab[feats], lab["er_fwd"])
    s_tr = model.predict(lab[feats])
    cut = np.quantile(s_tr, q)
    last = d[feats].dropna().iloc[[-1]]
    score = model.predict(last)[0]
    is_trend = score >= cut
    direction = {1: "bullish bias", -1: "bearish bias"}.get(int(d.loc[last.index[0], "dir_hint"]), "no bias")
    print(f"Latest closed bar : {last.index[0]}")
    print(f"Predicted regime  : {'TREND' if is_trend else 'SIDEWAYS / CHOP'} over the next {H} bars")
    print(f"Score percentile  : {(s_tr < score).mean():.0%} of training bars scored lower")
    if is_trend:
        print(f"EMA20/50 hint     : {direction} (display only, not scored)")

current_regime(data, H, TREND_Q)
