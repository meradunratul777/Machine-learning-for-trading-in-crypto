
# # BTC 4H Volatility-Regime Filter (Linear Regression, HAR-style)
#
# **Question answered at each 4H close:** will the next `H` bars be HIGH-volatility or LOW-volatility?
#
# **Target (continuous):** log of forward realized volatility, sqrt(mean(r^2)) over bars t+1..t+H.
# **Label (binary):** HIGH if forward vol is above the training median (`VOL_Q`), else LOW.
#   - `LABEL_MODE = "relative"`: forward vol compared with the trailing 60-day norm (answers "is vol about to expand or contract?"; immune to BTC's long-run vol decline).
#   - `LABEL_MODE = "absolute"`: forward vol compared with the whole training history (easier; a lot of the accuracy is just persistence/level).
#
# **Fair test:** purged walk-forward, and the model must beat three naive volatility forecasts
# (persistence, RiskMetrics-style EWMA, 1-week realized vol) with paired block-bootstrap confidence intervals.

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
                             precision_recall_fscore_support, matthews_corrcoef)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", None)

# ----------------------------- CONFIG ---------------------------------------
SYMBOL      = "BTC/USDT"
TIMEFRAME   = "4h"
START_DATE  = "2019-01-01"
CSV_PATH    = None             # optional: your own 4H CSV (timestamp,open,high,low,close,volume)

H           = 12               # forecast window in bars (12 x 4h = 2 days)
VOL_Q       = 0.50             # HIGH = forward vol above the training median
LABEL_MODE  = "relative"       # "relative" (vs trailing 60-day vol) or "absolute"
INIT_TRAIN_BARS = 6 * 365 * 3
STEP_BARS       = 6 * 365 // 2
RUN_SENSITIVITY = True

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
EPS = 1e-14
REF_COL = "lrv_360"            # trailing 60-day realized vol = "normal" level for LABEL_MODE="relative"

def build_features(df):
    """Every feature uses data up to and including the CURRENT bar's close only."""
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
    logc = np.log(c)
    r = logc.diff()
    r2 = r ** 2
    X = pd.DataFrame(index=df.index)

    # HAR-style realized-vol ladder: 12h ... 60 days (log scale)
    for n in (3, 6, 12, 24, 42, 90, 180, 360):
        X[f"lrv_{n}"] = 0.5 * np.log(r2.rolling(n).mean() + EPS)
    X["lrv_ewma"] = 0.5 * np.log(r2.ewm(alpha=0.06, adjust=False, min_periods=30).mean() + EPS)  # lambda = 0.94

    # range-based vol (uses high/low, more efficient than close-to-close)
    pk = np.log(h / l) ** 2 / (4 * np.log(2))
    for n in (6, 24, 90):
        X[f"lpk_{n}"] = 0.5 * np.log(pk.rolling(n).mean() + EPS)

    X["lmax_12"] = np.log(r.abs().rolling(12).max() + 1e-7)          # biggest recent shock
    X["ret_6"], X["ret_42"] = logc - logc.shift(6), logc - logc.shift(42)   # leverage effect (signed returns)
    X["down_share_42"] = (r.clip(upper=0) ** 2).rolling(42).sum() / (r2.rolling(42).sum() + EPS)
    X["vol_rel"] = np.log(v / v.rolling(42).mean())

    for k in range(6):                                                # weekend/weekday effect (Sunday = baseline)
        X[f"dow_{k}"] = (df.index.dayofweek == k).astype(float)
    return X.replace([np.inf, -np.inf], np.nan), r2

EXTRA_COLS = ["y_abs", "ref", "b_persist", "b_ewma", "b_week", "close"]

def make_dataset(df, H):
    X, r2 = build_features(df)
    d = X.copy()
    d["y_abs"] = 0.5 * np.log(r2.rolling(H).mean().shift(-H) + EPS)   # TARGET: forward realized vol over t+1..t+H
    d["ref"] = X[REF_COL]
    d["b_persist"] = 0.5 * np.log(r2.rolling(H).mean() + EPS)          # baseline 1: "last H bars' vol continues"
    d["b_ewma"] = X["lrv_ewma"]                                        # baseline 2: RiskMetrics EWMA
    d["b_week"] = X["lrv_42"]                                          # baseline 3: last week's realized vol
    d["close"] = df["close"]
    return d

data = make_dataset(df, H)
FEATURES = [c for c in data.columns if c not in EXTRA_COLS]
print(f"{len(FEATURES)} features:", FEATURES)

# %%
# ----------------------------- WALK-FORWARD ENGINE --------------------------
SCORERS = ["model", "persist", "ewma", "week"]
BASE_COL = {"persist": "b_persist", "ewma": "b_ewma", "week": "b_week"}

def walk_forward(d, feats, H, q, init_train, step, relative):
    need = feats + ["y_abs", "ref", "b_persist", "b_ewma", "b_week"]
    lab = d.dropna(subset=need).copy()
    n = len(lab)
    init_train = min(init_train, int(n * 0.5))
    step = min(step, max(int(n * 0.1), 50))

    out, fold = [], 0
    for s in range(init_train, n, step):
        e = min(s + step, n)
        te = lab.iloc[s:e]
        if len(te) < 20:
            continue
        tr = lab.iloc[: s - H]                                   # PURGE: labels of the last H train rows overlap the test period
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(tr[feats], tr["y_abs"])

        off_tr = tr["ref"].values if relative else 0.0
        off_te = te["ref"].values if relative else 0.0
        y_tr, y_te = tr["y_abs"].values - off_tr, te["y_abs"].values - off_te
        thr = np.quantile(y_tr, q)
        t1, t2 = np.quantile(y_tr, [1 / 3, 2 / 3])

        sc_tr = {"model": model.predict(tr[feats]) - off_tr}
        sc_te = {"model": model.predict(te[feats]) - off_te}
        for k, col in BASE_COL.items():
            sc_tr[k], sc_te[k] = tr[col].values - off_tr, te[col].values - off_te

        row = {"fold": fold, "y": y_te, "y_abs": te["y_abs"].values, "off": np.broadcast_to(off_te, len(te)),
               "y_true": (y_te >= thr).astype(int), "y3": (y_te >= t1).astype(int) + (y_te >= t2).astype(int)}
        for k in SCORERS:                                        # every cut-off comes from TRAIN scores only
            cut = np.quantile(sc_tr[k], q)
            c1, c2 = np.quantile(sc_tr[k], [1 / 3, 2 / 3])
            row[f"s_{k}"] = sc_te[k]
            row[f"p_{k}"] = (sc_te[k] >= cut).astype(int)
            row[f"p3_{k}"] = (sc_te[k] >= c1).astype(int) + (sc_te[k] >= c2).astype(int)
        out.append(pd.DataFrame(row, index=te.index))
        fold += 1
    return pd.concat(out)

def evaluate(y_true, y_pred, score=None):
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1, 0], zero_division=0)
    m = {"accuracy": accuracy_score(y_true, y_pred), "balanced_acc": balanced_accuracy_score(y_true, y_pred),
         "precision_HIGH": p[0], "precision_LOW": p[1], "recall_HIGH": r[0], "recall_LOW": r[1],
         "MCC": matthews_corrcoef(y_true, y_pred)}
    if score is not None and len(np.unique(y_true)) == 2:
        m["AUC"] = roc_auc_score(y_true, score)
    return m

def _blocks(n, block, rng):
    nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block + 1, nb)
    return (st[:, None] + np.arange(block)).ravel()[:n]

def block_ci(fn, *arrays, block=24, n_boot=500, seed=0):
    """95% CI of fn(*arrays) with a moving-block bootstrap (forecast windows overlap, so single-bar resampling would be too optimistic)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    vals = []
    for _ in range(n_boot):
        i = _blocks(len(arrays[0]), block, rng)
        try:
            v = fn(*[a[i] for a in arrays])
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5])

# %%
# ----------------------------- RUN + REPORT ---------------------------------
RELATIVE = (LABEL_MODE == "relative")
oos = walk_forward(data, FEATURES, H, VOL_Q, INIT_TRAIN_BARS, STEP_BARS, RELATIVE)
blk = max(2 * H, 24)

print(f"OOS: {oos.index[0]} -> {oos.index[-1]} | {len(oos):,} bars | {oos.fold.nunique()} folds")
print(f"Task: will the next {H} bars ({H*4}h) be HIGH vol (top {(1-VOL_Q):.0%}) or LOW vol? | label mode = {LABEL_MODE}")
print(f"HIGH base rate = {oos.y_true.mean():.1%}  -> chance accuracy about 50% when classes are balanced\n")

names = {"model": "Linear Regression (HAR+)", "persist": "Baseline: last-H-bars vol persists",
         "ewma": "Baseline: EWMA (lambda 0.94)", "week": "Baseline: 1-week realized vol"}

# --- classification quality
cls = pd.DataFrame({names[k]: evaluate(oos.y_true, oos[f"p_{k}"], oos[f"s_{k}"]) for k in SCORERS}).round(3)
print("=== HIGH vs LOW classification ===")
print(cls)

print("\n95% block-bootstrap CI (model):")
for nm, fn in [("accuracy", accuracy_score), ("balanced_acc", balanced_accuracy_score)]:
    lo, hi = block_ci(fn, oos.y_true, oos.p_model, block=blk)
    print(f"  {nm:14s} [{lo:.3f}, {hi:.3f}]")

# --- forecasting quality (log-vol scale)
reg = {}
for k in SCORERS:
    err = oos["y"] - oos[f"s_{k}"]
    reg[names[k]] = {"corr": np.corrcoef(oos["y"], oos[f"s_{k}"])[0, 1],
                     "spearman": oos["y"].corr(oos[f"s_{k}"], method="spearman"),
                     "RMSE_log": np.sqrt((err ** 2).mean())}
print("\n=== Forecast quality (log forward vol) ===")
print(pd.DataFrame(reg).T.round(3))

# --- does the model beat the naive forecasts? (paired, same bars)
print("\n=== Model vs each baseline (positive = model better; CI above 0 = real improvement) ===")
rows = []
for k in ("persist", "ewma", "week"):
    auc_d = roc_auc_score(oos.y_true, oos.s_model) - roc_auc_score(oos.y_true, oos[f"s_{k}"])
    a_lo, a_hi = block_ci(lambda y, a, b: roc_auc_score(y, a) - roc_auc_score(y, b),
                          oos.y_true, oos.s_model, oos[f"s_{k}"], block=blk)
    mse_d = ((oos.y - oos[f"s_{k}"]) ** 2).mean() - ((oos.y - oos.s_model) ** 2).mean()
    m_lo, m_hi = block_ci(lambda yy, a, b: ((yy - b) ** 2).mean() - ((yy - a) ** 2).mean(),
                          oos.y, oos.s_model, oos[f"s_{k}"], block=blk)
    rows.append({"vs": names[k], "AUC gain": f"{auc_d:+.3f} [{a_lo:+.3f}, {a_hi:+.3f}]",
                 "MSE gain (log vol)": f"{mse_d:+.3f} [{m_lo:+.3f}, {m_hi:+.3f}]"})
print(pd.DataFrame(rows).to_string(index=False))

# --- 3-state: LOW / NORMAL / HIGH (terciles)
print("\n=== 3-state (LOW / NORMAL / HIGH), chance = 33% ===")
tri = {}
for k in SCORERS:
    acc3 = (oos[f"p3_{k}"] == oos.y3).mean()
    severe = (((oos[f"p3_{k}"] == 0) & (oos.y3 == 2)) | ((oos[f"p3_{k}"] == 2) & (oos.y3 == 0))).mean()
    tri[names[k]] = {"accuracy_3state": acc3, "severe_miss_rate": severe}
print(pd.DataFrame(tri).T.round(3))
print("(severe miss = said LOW when it became HIGH, or the reverse; random calls would give ~22%)")

# --- does a higher predicted score really mean higher realized vol?
oos["bucket"] = pd.qcut(oos["s_model"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
lift = oos.groupby("bucket", observed=True).agg(
    bars=("y_true", "size"), actual_HIGH_rate=("y_true", "mean"),
    mean_fwd_vol_pct=("y_abs", lambda s: np.exp(s).mean() * 100)).round(3)
print("\n=== Predicted-score quintiles vs what happened (should rise monotonically) ===")
print(lift)
fv = np.exp(oos.y_abs)
print(f"\nAs a filter: mean forward vol when model says HIGH = {fv[oos.p_model==1].mean()*100:.3f}% per bar, "
      f"LOW = {fv[oos.p_model==0].mean()*100:.3f}% (ratio {fv[oos.p_model==1].mean()/fv[oos.p_model==0].mean():.2f}x; "
      f"a perfect oracle would give {fv[oos.y_true==1].mean()/fv[oos.y_true==0].mean():.2f}x)")

# --- stability through time
pf = oos.groupby("fold").apply(lambda g: pd.Series({
    "start": g.index[0].date(), "HIGH_rate": g.y_true.mean(),
    "acc_model": accuracy_score(g.y_true, g.p_model), "acc_persist": accuracy_score(g.y_true, g.p_persist),
    "AUC_model": roc_auc_score(g.y_true, g.s_model), "AUC_persist": roc_auc_score(g.y_true, g.s_persist)})).round(3)
print("\n=== Per fold ===")
print(pf)

# %%
# ----------------------------- PLOTS ----------------------------------------
seg = oos.iloc[-1500:]
fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax[0].plot(seg.index, np.exp(seg.y_abs) * 100, lw=0.8, label="actual forward vol (next %d bars)" % H, color="tab:gray")
ax[0].plot(seg.index, np.exp(seg.s_model + seg.off) * 100, lw=1.2, label="model forecast", color="tab:blue")
ax[0].set_yscale("log"); ax[0].set_ylabel("vol, % per 4H bar"); ax[0].legend(); ax[0].set_title("Forecast vs realized volatility (last 1500 OOS bars)")
m = seg.y_true.values == 1; p = seg.p_model.values == 1
ax[1].fill_between(seg.index, 0.5, 1, where=m, step="mid", color="tab:red", alpha=0.5, label="actual HIGH")
ax[1].fill_between(seg.index, 0, 0.5, where=p, step="mid", color="tab:blue", alpha=0.5, label="predicted HIGH")
ax[1].set_yticks([]); ax[1].legend(loc="upper right", ncol=2)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].hexbin(oos.s_model, oos.y, gridsize=45, bins="log", cmap="Blues")
lim = [min(oos.s_model.min(), oos.y.min()), max(oos.s_model.max(), oos.y.max())]
ax[0].plot(lim, lim, "r--", lw=1); ax[0].set_xlabel("predicted (log vol, label scale)"); ax[0].set_ylabel("actual")
ax[0].set_title("Predicted vs actual")
pf[["acc_model", "acc_persist"]].plot.bar(ax=ax[1]); ax[1].axhline(0.5, color="k", ls=":")
ax[1].set_ylim(0.4, 1); ax[1].set_title("Accuracy by fold (dotted = chance)")
plt.tight_layout(); plt.show()

# %%
# ----------------------------- SENSITIVITY (optional) -----------------------
# Look for STABILITY across settings. Don't cherry-pick the best cell.
if RUN_SENSITIVITY:
    rows = []
    for H_ in (6, 12, 24, 42):
        d_ = make_dataset(df, H_)
        for q_ in (0.5, 0.67):
            o_ = walk_forward(d_, FEATURES, H_, q_, INIT_TRAIN_BARS, STEP_BARS, RELATIVE)
            best_base = max(roc_auc_score(o_.y_true, o_[f"s_{k}"]) for k in ("persist", "ewma", "week"))
            rows.append({"H": H_, "VOL_Q": q_, "HIGH_rate": o_.y_true.mean(),
                         "acc": accuracy_score(o_.y_true, o_.p_model),
                         "bal_acc": balanced_accuracy_score(o_.y_true, o_.p_model),
                         "AUC_model": roc_auc_score(o_.y_true, o_.s_model),
                         "AUC_best_baseline": best_base,
                         "corr": np.corrcoef(o_.y, o_.s_model)[0, 1]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

# %%
# ----------------------------- CURRENT REGIME -------------------------------
def current_vol_regime(d, feats, H, q, relative):
    lab = d.dropna(subset=feats + ["y_abs", "ref"])
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(lab[feats], lab["y_abs"])
    off_tr = lab["ref"].values if relative else 0.0
    s_tr = model.predict(lab[feats]) - off_tr
    cut = np.quantile(s_tr, q)
    c1, c2 = np.quantile(s_tr, [1 / 3, 2 / 3])
    last = d[feats].dropna().iloc[[-1]]
    pred_abs = model.predict(last)[0]
    score = pred_abs - (d.loc[last.index[0], "ref"] if relative else 0.0)
    state = "HIGH" if score >= cut else "LOW"
    tier = ["LOW", "NORMAL", "HIGH"][int(score >= c1) + int(score >= c2)]
    vol = np.exp(pred_abs)
    print(f"Latest closed bar     : {last.index[0]}")
    print(f"Forecast vol (next {H}): {vol*100:.2f}% per 4H bar | {vol*np.sqrt(6)*100:.2f}% daily | {vol*np.sqrt(6*365)*100:.0f}% annualized")
    print(f"Regime                : {state} ({LABEL_MODE}), 3-state = {tier}, score percentile {(s_tr < score).mean():.0%}")

current_vol_regime(data, FEATURES, H, VOL_Q, RELATIVE)
