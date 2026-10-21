# %% [markdown]
# # Toxic order-flow classifier (1-minute, crypto) – Logistic Regression
#
# **Goal:** for a market maker, estimate `P(toxic)` = probability that the aggressive flow we just
# absorbed in the last 1-minute bar was *informed*, i.e. price keeps moving against us afterwards
# (adverse selection). The quoting engine can use it to widen spreads / skew / pull quotes.
#
# **Label (markout based, no look-ahead in features):**
# * `side_t = sign(taker_buy - taker_sell)` in bar *t*  (+1: takers bought from us, -1: takers sold to us)
# * `markout_t = side_t * log(close_{t+H} / close_t)` in bps  (> 0 means price moved against the MM)
# * `toxic_t = 1  if  markout_t > max(min_edge_bps, k_sigma * sigma_t * sqrt(H))`  (sigma_t = trailing 60m vol)
#
# **Key design choices**
# * All *directional* features are multiplied by `side_t` ("side-aligned"), otherwise a linear model
#   cannot use them (buy-side and sell-side flow would need opposite coefficients).
# * Rolling features never see the future; a causality unit-test proves it.
# * Overlapping H-minute labels -> purged chronological split + purged walk-forward CV (gap = H).
# * Winsorise + standardise inside the sklearn Pipeline (fit on train only).
# * No class re-weighting, so probabilities stay calibrated and usable as a spread-widening input.
# * Metrics: ROC-AUC (block-bootstrap CI), PR-AUC, log-loss/Brier skill vs. prior, calibration (ECE),
#   decile lift **in bps of adverse markout**, and weekly stability.
#
# Data: Binance public data (data.binance.vision) 1m klines – includes taker-buy volume, which gives
# us signed order flow without needing trade-level data.

# %% Imports
import os, sys, json, time, zipfile, warnings
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                             roc_curve, precision_recall_curve, confusion_matrix)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
EPS = 1e-12


# %% Configuration
@dataclass
class Config:
    # ---- data ----
    symbol: str = "BTCUSDT"
    market: str = "spot"                 # "spot" or "um" (USD-M futures)
    n_months: int = 12                   # number of monthly files to load
    end_month: str = None                # "YYYY-MM"; None = last fully completed month
    data_dir: str = "./data"
    out_dir: str = "./outputs"
    force_synthetic: bool = False        # True = skip download, use simulated data (smoke test only)
    allow_synthetic_fallback: bool = True  # if download fails -> synthetic data (loudly flagged)
    # ---- label ----
    horizon: int = 3                     # H minutes for the markout
    k_sigma: float = 0.5                 # toxic if markout > k_sigma * sigma * sqrt(H) ...
    min_edge_bps: float = 1.0            # ... and at least this many bps (fee/half-spread proxy)
    # ---- split / CV ----
    train_frac: float = 0.60
    val_frac: float = 0.15               # test = remainder (0.25)
    cv_splits: int = 5
    c_grid: tuple = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    winsor_q: float = 0.005
    seed: int = 42


cfg = Config()            # <- edit here, e.g. Config(symbol="ETHUSDT", market="um", n_months=18)
os.makedirs(cfg.data_dir, exist_ok=True)
os.makedirs(cfg.out_dir, exist_ok=True)
np.random.seed(cfg.seed)


# %% Data engineering: download / parse / clean Binance 1m klines
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]
BASE_URL = "https://data.binance.vision/data"


def month_list(cfg: Config):
    if cfg.end_month:
        end = pd.Period(cfg.end_month, "M")
    else:
        end = pd.Timestamp.now("UTC").tz_localize(None).to_period("M") - 1
    return [str(end - i) for i in range(cfg.n_months)][::-1]


def _download_month(cfg: Config, ym: str, retries: int = 3):
    kind = "spot" if cfg.market == "spot" else "futures/um"
    url = f"{BASE_URL}/{kind}/monthly/klines/{cfg.symbol}/1m/{cfg.symbol}-1m-{ym}.zip"
    path = os.path.join(cfg.data_dir, f"{cfg.symbol}_{cfg.market}_1m_{ym}.zip")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                print(f"  [skip] {ym}: not available (404)")
                return None
            r.raise_for_status()
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, path)
            return path
        except Exception as e:                                   # noqa: BLE001
            if attempt == retries - 1:
                print(f"  [fail] {ym}: {type(e).__name__}: {e}")
                return None
            time.sleep(1.5 * (attempt + 1))


def _read_kline_zip(path: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            raw = pd.read_csv(f, header=None, names=KLINE_COLS, usecols=range(12), dtype=str)
    t = pd.to_numeric(raw["open_time"], errors="coerce")
    raw = raw.loc[t.notna()].copy()                              # drops header row if the file has one
    t = t.loc[raw.index].astype("int64")
    unit = "us" if t.median() > 1e14 else "ms"                   # Binance spot switched to µs in 2025
    raw.index = pd.to_datetime(t, unit=unit)
    raw.index.name = "ts"
    num = ["open", "high", "low", "close", "volume", "quote_volume", "n_trades",
           "taker_buy_base", "taker_buy_quote"]
    return raw[num].astype("float64")


def clean_klines(df: pd.DataFrame):
    """Dedupe, sort, sanity-fix, and reindex to a complete 1-minute grid (gaps stay NaN – never forward-filled)."""
    rep = {}
    rep["dupes"] = int(df.index.duplicated().sum())
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["volume"] < 0)
    rep["bad_rows"] = int(bad.sum())
    df = df.mask(bad)
    over = df["taker_buy_base"] > df["volume"]
    rep["taker_gt_volume_clipped"] = int(over.sum())
    df["taker_buy_base"] = df["taker_buy_base"].clip(upper=df["volume"])
    full = pd.date_range(df.index[0], df.index[-1], freq="1min")
    rep["missing_minutes"] = int(len(full) - len(df))
    df = df.reindex(full)
    df.index.name = "ts"
    isna = df["close"].isna()
    runs = isna.groupby((isna != isna.shift()).cumsum()).sum()
    rep["longest_gap_min"] = int(runs.max()) if len(runs) else 0
    rep["zero_volume_bars"] = int((df["volume"] == 0).sum())
    return df, rep


def make_synthetic_klines(n_days: int = 365, seed: int = 7) -> pd.DataFrame:
    """Simulated 1m klines with a latent persistent 'informed flow' factor. SMOKE TEST ONLY."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    mod = (idx.hour * 60 + idx.minute).to_numpy()
    season = 1 + 0.5 * np.sin(2 * np.pi * (mod - 780) / 1440)
    lv = lfilter([1], [1, -0.995], rng.normal(0, 0.04, n))
    sigma = 3.0 * np.exp(lv) * season                            # bps / minute
    phi = 0.92
    z = lfilter([1], [1, -phi], rng.normal(0, 1, n)) * np.sqrt(1 - phi ** 2)   # informed-flow state
    ret = sigma * (0.12 * np.r_[0, z[:-1]] + rng.normal(0, 1, n))              # bps
    close = 60000 * np.exp(np.cumsum(ret) / 1e4)
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 0.6, (2, n))) * sigma / 1e4
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    volume = 25 * season * np.exp(1.2 * lv + 0.35 * rng.normal(size=n)) * (1 + 0.3 * np.abs(z))
    p_buy = 1 / (1 + np.exp(-(0.55 * z + 0.45 * rng.normal(size=n))))
    taker_buy = volume * p_buy
    avg_size = 0.02 * np.exp(0.3 * rng.normal(size=n) + 0.25 * np.abs(z))
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume,
                       "quote_volume": volume * close, "n_trades": np.maximum(1, volume / avg_size).round(),
                       "taker_buy_base": taker_buy, "taker_buy_quote": taker_buy * close}, index=idx)
    df.index.name = "ts"
    for start, length in [(30_000, 5), (150_000, 40), (320_000, 300)]:      # exchange-outage style gaps
        df = df.drop(df.index[start:start + length])
    return df


def load_data(cfg: Config):
    if not cfg.force_synthetic:
        months = month_list(cfg)
        print(f"Downloading {cfg.symbol} ({cfg.market}) 1m klines: {months[0]} -> {months[-1]}")
        with ThreadPoolExecutor(max_workers=4) as ex:
            paths = list(ex.map(lambda m: _download_month(cfg, m), months))
        frames = [_read_kline_zip(p) for p in paths if p]
        if frames:
            return pd.concat(frames), "binance"
        if not cfg.allow_synthetic_fallback:
            raise RuntimeError("No data downloaded. Check internet access / symbol / market.")
    print("\n" + "!" * 78 + "\n!!  USING SYNTHETIC DATA – results are a pipeline smoke test, NOT real market evidence  !!\n"
          + "!" * 78 + "\n")
    return make_synthetic_klines(seed=cfg.seed), "SYNTHETIC"


raw, DATA_SOURCE = load_data(cfg)
df, rep = clean_klines(raw)
print(f"\nData source : {DATA_SOURCE}")
print(f"Bars        : {len(df):,}   ({df.index[0]} -> {df.index[-1]} UTC)")
print("Data quality:", rep)


# %% Feature engineering (strictly causal: every feature at row t uses data up to and including bar t)
def make_features(df: pd.DataFrame, cfg: Config):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    qv, nt, buy = df["quote_volume"], df["n_trades"], df["taker_buy_base"]

    sv = 2.0 * buy - v                                  # signed volume = taker buy - taker sell
    ofi = sv / v.where(v > 0)                           # order-flow imbalance in [-1, 1]
    side = np.sign(ofi)                                 # +1 takers bought (MM sold), -1 takers sold
    ret = 1e4 * np.log(c / c.shift(1))                  # 1m log-return in bps

    vol_ma60, vol_ma360 = v.rolling(60).mean(), v.rolling(360).mean()
    rv5, rv15, rv60 = ret.rolling(5).std(), ret.rolling(15).std(), ret.rolling(60).std()
    avg_size = v / nt.where(nt > 0)
    rng_bps = 1e4 * np.log(h / l)
    svn = sv / (vol_ma60 + EPS)                         # signed volume in units of "normal" volume
    sgn = np.sign(ofi)
    clv = ((c - l) / (h - l)).where(h > l, 0.5)         # close location value in [0, 1]

    F = {}
    # ---------- symmetric (side-agnostic) toxicity / regime features ----------
    F["abs_ofi"] = ofi.abs()
    for w in (10, 30, 60):                              # VPIN-style proxy on time bars
        F[f"vpin_{w}"] = sv.abs().rolling(w).sum() / v.rolling(w).sum()
    F["abs_sv_shock"] = sv.abs() / (vol_ma60 + EPS)
    F["log_vol_shock_60"] = np.log((v + EPS) / (vol_ma60 + EPS))
    F["log_vol_shock_360"] = np.log((v + EPS) / (vol_ma360 + EPS))
    F["log_trade_rate_shock_60"] = np.log((nt + EPS) / (nt.rolling(60).mean() + EPS))
    F["log_trade_size_shock_60"] = np.log((avg_size + EPS) / (avg_size.rolling(60).mean() + EPS))
    F["log_rv_15"] = np.log(rv15 + EPS)
    F["log_rv_60"] = np.log(rv60 + EPS)
    F["log_rv_ratio_5_60"] = np.log((rv5 + EPS) / (rv60 + EPS))
    F["log_range_shock_60"] = np.log((rng_bps + EPS) / (rng_bps.rolling(60).mean() + EPS))
    F["log_amihud_30"] = np.log((ret.abs() / (qv / 1e6 + EPS)).rolling(30).mean() + EPS)
    F["kyle_lambda_60"] = ret.rolling(60).cov(svn) / (svn.rolling(60).var() + EPS)   # price impact per unit flow
    F["ofi_x_volshock"] = F["abs_ofi"] * F["log_vol_shock_60"]
    mod = np.asarray(df.index.hour * 60 + df.index.minute, dtype=float)
    F["tod_sin"] = pd.Series(np.sin(2 * np.pi * mod / 1440), index=df.index)
    F["tod_cos"] = pd.Series(np.cos(2 * np.pi * mod / 1440), index=df.index)
    F["is_weekend"] = pd.Series(np.asarray(df.index.dayofweek >= 5, dtype=float), index=df.index)

    # ---------- directional features, aligned to the side of the flow we just absorbed ----------
    for k in (1, 2, 3):
        F[f"a_ofi_lag{k}"] = side * ofi.shift(k)
    for w in (5, 15, 60):
        F[f"a_cumofi_{w}"] = side * sv.rolling(w).sum() / v.rolling(w).sum()
    for w in (5, 15):
        F[f"a_sign_mean_{w}"] = side * sgn.rolling(w).mean()
    F["a_ret_0"] = side * ret
    F["a_ret_lag1"] = side * ret.shift(1)
    F["a_ret_lag2"] = side * ret.shift(2)
    for w in (5, 15, 60):
        F[f"a_ret_cum_{w}"] = side * 1e4 * np.log(c / c.shift(w))
    F["a_ret_z"] = side * ret / (rv60 + EPS)
    F["a_close_loc"] = side * (2 * clv - 1)

    X = pd.DataFrame(F, index=df.index).replace([np.inf, -np.inf], np.nan)
    aux = pd.DataFrame({"side": side, "close": c, "rv60": rv60, "ofi": ofi}, index=df.index)
    return X, aux


def make_labels(aux: pd.DataFrame, cfg: Config):
    H = cfg.horizon
    fwd_bps = 1e4 * np.log(aux["close"].shift(-H) / aux["close"])
    markout = aux["side"] * fwd_bps                                   # > 0  => adverse for the market maker
    thr = np.maximum(cfg.min_edge_bps, cfg.k_sigma * aux["rv60"] * np.sqrt(H))
    y = (markout > thr).astype(float).where(markout.notna() & thr.notna())
    return pd.DataFrame({"y": y, "markout_bps": markout, "thr_bps": thr})


def build_dataset(df: pd.DataFrame, cfg: Config, with_labels: bool = True):
    X, aux = make_features(df, cfg)
    ok = X.notna().all(axis=1) & aux["side"].notna() & (aux["side"] != 0)
    lab = None
    if with_labels:
        lab = make_labels(aux, cfg)
        ok &= lab["y"].notna()
        lab = lab.loc[ok]
    return X.loc[ok], (lab["y"].astype(int) if with_labels else None), (lab if with_labels else None), aux.loc[ok]


def check_no_lookahead(df: pd.DataFrame, cfg: Config, n_checks: int = 6, window: int = 2500, seed: int = 0):
    """Causality unit-test: features at t computed on data truncated at t must equal features computed on the full data."""
    X_full, _ = make_features(df, cfg)
    rng = np.random.default_rng(seed)
    for i in rng.integers(window, len(df) - 10, size=n_checks):
        sub = df.iloc[i - window: i + 1]
        X_sub, _ = make_features(sub, cfg)
        a, b = X_full.iloc[i].to_numpy(float), X_sub.iloc[-1].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True, rtol=1e-8, atol=1e-8), f"LOOK-AHEAD detected at {df.index[i]}"
    print(f"[OK] causality test passed on {n_checks} random timestamps (no feature uses future data)")


check_no_lookahead(df, cfg)
X, y, lab, aux = build_dataset(df, cfg)
FEATURES = list(X.columns)
print(f"\nModelling rows : {len(X):,}  |  features: {len(FEATURES)}  |  toxic rate: {y.mean():.3f}")
print(f"Label rule     : markout over {cfg.horizon}m > max({cfg.min_edge_bps} bps, {cfg.k_sigma}·σ·√{cfg.horizon})  "
      f"(median threshold {lab['thr_bps'].median():.2f} bps)")


# %% Chronological split with purging (labels look H minutes ahead -> drop rows whose label window crosses a boundary)
def chrono_split(index: pd.DatetimeIndex, cfg: Config):
    n = len(index)
    t_val0 = index[int(n * cfg.train_frac)]
    t_test0 = index[int(n * (cfg.train_frac + cfg.val_frac))]
    purge = pd.Timedelta(minutes=cfg.horizon + 1)
    tr = index < (t_val0 - purge)
    va = (index >= t_val0) & (index < (t_test0 - purge))
    te = index >= t_test0
    return tr, va, te


tr, va, te = chrono_split(X.index, cfg)
X_tr, y_tr, X_va, y_va, X_te, y_te = X[tr], y[tr], X[va], y[va], X[te], y[te]
for name, a in [("train", X_tr), ("val", X_va), ("test", X_te)]:
    yy = y[a.index]
    print(f"{name:5s}: {len(a):>8,} rows  {a.index[0]} -> {a.index[-1]}   toxic rate {yy.mean():.3f}")


# %% Model: Winsorizer -> StandardScaler -> LogisticRegression, tuned with purged walk-forward CV
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to [q, 1-q] quantiles learned on the training data only (crypto features are fat-tailed)."""
    def __init__(self, q=0.005):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.q, axis=0)
        self.hi_ = np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


pipe = Pipeline([
    ("winsor", Winsorizer(q=cfg.winsor_q)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, random_state=cfg.seed)),   # L2; no class_weight -> calibrated probs
])
cv = TimeSeriesSplit(n_splits=cfg.cv_splits, gap=cfg.horizon + 1)
search = GridSearchCV(pipe, {"clf__C": list(cfg.c_grid)}, cv=cv, scoring="neg_log_loss",
                      n_jobs=-1, refit=True, return_train_score=False)
t0 = time.time()
search.fit(X_tr, y_tr)
print(f"Walk-forward CV done in {time.time() - t0:.0f}s   best C = {search.best_params_['clf__C']}")
print(pd.DataFrame({"C": cfg.c_grid, "cv_logloss": -search.cv_results_["mean_test_score"],
                    "std": search.cv_results_["std_test_score"]}).to_string(index=False))
model = search.best_estimator_


# %% Evaluation helpers
def ece_and_table(y, p, n_bins=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"], n_bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), p=("p", "mean"), obs=("y", "mean"))
    return float((g["n"] / g["n"].sum() * (g["p"] - g["obs"]).abs()).sum()), g


def evaluate(y, p, thr, p0):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec, rec, spec = tp / max(tp + fp, 1), tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ll, ll0 = log_loss(y, p), log_loss(y, np.full_like(p, p0))
    br, br0 = np.mean((y - p) ** 2), np.mean((y - p0) ** 2)
    return {"n": len(y), "toxic_rate": y.mean(), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "logloss": ll, "logloss_prior": ll0,
            "logloss_skill": 1 - ll / ll0, "brier": br, "brier_skill": 1 - br / br0,
            "ece": ece_and_table(y, p)[0], "flag_rate": pred.mean(), "precision": prec,
            "recall": rec, "specificity": spec, "bal_acc": (rec + spec) / 2,
            "lift": prec / y.mean(), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def block_bootstrap_auc(y, p, block=120, n_boot=200, seed=0):
    """Moving-block bootstrap CI for AUC (labels are overlapping & clustered -> iid bootstrap is too optimistic)."""
    y, p, n = np.asarray(y), np.asarray(p), len(y)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block, size=nb)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        if 0 < y[idx].sum() < len(idx):
            out.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(out, [2.5, 97.5])


# %% Fit diagnostics + out-of-sample TEST evaluation
p_tr, p_va, p_te = (model.predict_proba(a)[:, 1] for a in (X_tr, X_va, X_te))
p0 = float(y_tr.mean())                                                   # constant "prior" benchmark

# decision threshold chosen on VALIDATION only (Youden's J = max(TPR - FPR); prevalence-independent, non-degenerate)
fpr, tpr, ths = roc_curve(y_va, p_va)
THR = float(ths[np.argmax(tpr - fpr)])

res = {n: evaluate(yy, pp, THR, p0) for n, yy, pp in [("train", y_tr, p_tr), ("val", y_va, p_va), ("test", y_te, p_te)]}
auc_lo, auc_hi = block_bootstrap_auc(y_te.to_numpy(), p_te)
vpin_auc = roc_auc_score(y_te, X_te["vpin_30"])                           # classic toxicity heuristic as benchmark

print("\n" + "=" * 78)
print(f"TEST RESULTS  [{cfg.symbol} {cfg.market} 1m | data: {DATA_SOURCE}]  – out-of-sample, never used for fitting/tuning")
print("=" * 78)
T = res["test"]
print(f"Period            : {X_te.index[0]} -> {X_te.index[-1]}   ({T['n']:,} bars)")
print(f"Toxic base rate   : {T['toxic_rate']:.3f}")
print(f"ROC-AUC           : {T['roc_auc']:.4f}   95% block-bootstrap CI [{auc_lo:.4f}, {auc_hi:.4f}]")
print(f"PR-AUC            : {T['pr_auc']:.4f}   (random = {T['toxic_rate']:.4f})")
print(f"Log-loss          : {T['logloss']:.4f}   vs prior-only {T['logloss_prior']:.4f}   -> skill {T['logloss_skill']*100:+.2f}%")
print(f"Brier             : {T['brier']:.4f}   skill {T['brier_skill']*100:+.2f}%")
print(f"Calibration (ECE) : {T['ece']:.4f}")
print(f"Benchmark         : raw VPIN(30) heuristic AUC = {vpin_auc:.4f}")
print(f"\nDecision rule     : flag toxic if P >= {THR:.4f}  (threshold chosen on validation)")
print(f"  flagged {T['flag_rate']*100:.1f}% of bars | precision {T['precision']:.3f} (lift {T['lift']:.2f}x) | "
      f"recall {T['recall']:.3f} | specificity {T['specificity']:.3f} | balanced acc {T['bal_acc']:.3f}")
print(f"  confusion matrix  [[TN {T['TN']:,}  FP {T['FP']:,}]  [FN {T['FN']:,}  TP {T['TP']:,}]]")

summary = pd.DataFrame(res).T[["n", "toxic_rate", "roc_auc", "pr_auc", "logloss", "logloss_skill", "brier_skill", "ece"]]
print("\nTrain / Val / Test side by side (a big train->test AUC drop = overfit / regime shift):")
print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

# ---- economic view: adverse markout by predicted-risk decile (this is what a MM actually cares about) ----
mk_te = lab.loc[X_te.index, "markout_bps"]
dec = pd.DataFrame({"p": p_te, "y": y_te.to_numpy(), "mk": mk_te.to_numpy()})
dec["decile"] = pd.qcut(dec["p"], 10, labels=False, duplicates="drop") + 1
dtab = dec.groupby("decile").agg(n=("y", "size"), mean_pred=("p", "mean"), toxic_rate=("y", "mean"),
                                 mean_markout_bps=("mk", "mean"), median_markout_bps=("mk", "median"))
print("\nTest decile table (decile 10 = model's highest toxicity score):")
print(dtab.to_string(float_format=lambda x: f"{x:.4f}"))
flag = p_te >= THR
print(f"\nMean adverse markout (bps):  flagged = {dec.loc[flag,'mk'].mean():+.3f} | not flagged = {dec.loc[~flag,'mk'].mean():+.3f} "
      f"| all = {dec['mk'].mean():+.3f}   (positive = price moved against the quoter)")

# ---- practical policy: widen/pull quotes only on the riskiest 10% of bars (cut-off taken from VALIDATION scores) ----
thr90 = float(np.quantile(p_va, 0.90))
f90 = p_te >= thr90
print(f"\nTop-10% policy (P >= {thr90:.4f}): flagged {f90.mean()*100:.1f}% of test bars | toxic rate {dec.loc[f90,'y'].mean():.3f} "
      f"(lift {dec.loc[f90,'y'].mean()/T['toxic_rate']:.2f}x) | mean markout {dec.loc[f90,'mk'].mean():+.3f} bps "
      f"vs {dec.loc[~f90,'mk'].mean():+.3f} bps on the rest")

# ---- stability: weekly AUC on the test period ----
weekly = {}
for wk_id, g in dec.assign(wk=X_te.index.to_period("W")).groupby("wk"):
    if g["y"].nunique() == 2 and len(g) >= 1000:
        weekly[str(wk_id)] = roc_auc_score(g["y"], g["p"])
wk = pd.Series(weekly, dtype=float)
print(f"\nWeekly test AUC: min {wk.min():.3f} | median {wk.median():.3f} | max {wk.max():.3f} | "
      f"weeks with AUC>0.5: {(wk > 0.5).sum()}/{len(wk)}")

# ---- coefficients (standardised features -> directly comparable) ----
coef = pd.Series(model.named_steps["clf"].coef_[0], index=FEATURES)
ctab = pd.DataFrame({"coef_per_1sd": coef, "odds_ratio_per_1sd": np.exp(coef)}).sort_values("coef_per_1sd", key=np.abs, ascending=False)
print("\nTop 15 coefficients (standardised features):")
print(ctab.head(15).to_string(float_format=lambda x: f"{x:+.4f}"))


# %% Plots
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Toxic-flow logistic regression – {cfg.symbol} {cfg.market} 1m  |  data: {DATA_SOURCE}  |  H={cfg.horizon}m",
             fontsize=14, color=("red" if DATA_SOURCE == "SYNTHETIC" else "black"))

fv, tv, _ = roc_curve(y_va, p_va); ft, tt, _ = roc_curve(y_te, p_te)
ax[0, 0].plot(ft, tt, label=f"test AUC {res['test']['roc_auc']:.3f}")
ax[0, 0].plot(fv, tv, alpha=.6, label=f"val AUC {res['val']['roc_auc']:.3f}")
ax[0, 0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0, 0].set(title="ROC", xlabel="FPR", ylabel="TPR"); ax[0, 0].legend()

pr, rc, _ = precision_recall_curve(y_te, p_te)
ax[0, 1].plot(rc, pr); ax[0, 1].axhline(T["toxic_rate"], color="k", ls="--", lw=.8, label="base rate")
ax[0, 1].set(title=f"Precision-Recall (test, AP={T['pr_auc']:.3f})", xlabel="Recall", ylabel="Precision"); ax[0, 1].legend()

_, cal = ece_and_table(y_te, p_te)
lim = [min(cal.p.min(), cal.obs.min()) * .95, max(cal.p.max(), cal.obs.max()) * 1.05]
ax[0, 2].plot(cal["p"], cal["obs"], "o-"); ax[0, 2].plot(lim, lim, "k--", lw=.8)
ax[0, 2].set(title=f"Calibration (test, ECE={T['ece']:.4f})", xlabel="Mean predicted P(toxic)", ylabel="Observed toxic rate")

ax[1, 0].bar(dtab.index, dtab["mean_markout_bps"], color=np.where(dtab["mean_markout_bps"] > 0, "tab:red", "tab:green"))
ax[1, 0].axhline(0, color="k", lw=.8)
ax[1, 0].set(title="Mean adverse markout by predicted-risk decile (test)", xlabel="Risk decile (10 = most toxic)", ylabel="bps")

top = ctab.head(15).iloc[::-1]
ax[1, 1].barh(top.index, top["coef_per_1sd"], color=np.where(top["coef_per_1sd"] > 0, "tab:red", "tab:blue"))
ax[1, 1].set(title="Top coefficients (per +1 SD)", xlabel="log-odds")

ax[1, 2].bar(range(len(wk)), wk.values, color="tab:blue"); ax[1, 2].axhline(0.5, color="k", ls="--", lw=.8)
ax[1, 2].set(title="Weekly AUC on test period", xlabel="week #", ylabel="AUC", ylim=(max(0.4, wk.min() - .03), wk.max() + .03))
plt.tight_layout()
fig.savefig(os.path.join(cfg.out_dir, "toxic_flow_report.png"), dpi=130)
plt.show()


# %% Persist the model + a live-scoring helper (same feature code path => no train/serve skew)
# NOTE: the pipeline contains the custom `Winsorizer` class. To reload the model in another session, define that class
# (or move it into an importable module) BEFORE calling joblib.load(...), together with make_features()/Config.
bundle = {"pipeline": model, "features": FEATURES, "threshold": THR, "config": asdict(cfg),
          "data_source": DATA_SOURCE, "test_metrics": {k: float(v) for k, v in res["test"].items()}}
joblib.dump(bundle, os.path.join(cfg.out_dir, "toxic_flow_logreg.joblib"))
with open(os.path.join(cfg.out_dir, "test_metrics.json"), "w") as f:
    json.dump(bundle["test_metrics"], f, indent=2)
ctab.to_csv(os.path.join(cfg.out_dir, "coefficients.csv"))


def score_latest(bundle: dict, recent_klines: pd.DataFrame) -> dict:
    """Pass >= 400 most recent COMPLETE 1m klines (same columns as `df`). Returns P(toxic) for the latest closed bar."""
    c = Config(**bundle["config"])
    Xl, aux_l = make_features(recent_klines, c)
    x, side = Xl.iloc[[-1]][bundle["features"]], aux_l["side"].iloc[-1]
    if x.isna().any(axis=None) or not np.isfinite(side) or side == 0:
        return {"ts": recent_klines.index[-1], "p_toxic": None, "reason": "insufficient history / no net flow"}
    p = float(bundle["pipeline"].predict_proba(x)[0, 1])
    return {"ts": recent_klines.index[-1], "flow_side": "takers BOUGHT (we sold)" if side > 0 else "takers SOLD (we bought)",
            "p_toxic": round(p, 4), "flag": bool(p >= bundle["threshold"])}


print("\nSaved to", os.path.abspath(cfg.out_dir), "->", os.listdir(cfg.out_dir))
print("Live-scoring demo on the most recent complete bars:", score_latest(bundle, df.iloc[-1500:]))
