# %% [markdown]
# # 1-minute crypto ATR forecaster v2: linear model, diagnostics, and a step-by-step improvement ladder
#
# Same target as v1: forward ATR = mean true range of the next `HORIZON` minutes / current close (log space).
#
# What is new, and which weakness of v1 each item targets:
# 1. **Adaptive seasonality** (same time yesterday / last week, causal)   -> intraday pattern drifts, static dummies can't follow
# 2. **Scheduled macro events** (FOMC, CPI, jobs report; DST-safe)         -> spikes that are known in advance
# 3. **Spline terms** on key features (still a linear regression)          -> nonlinear vol dynamics / fat-tailed errors
# 4. **Vol-adaptive prediction bands** instead of constant width            -> 90% band covered only 87.7% in v1
# 5. **DST-safe clock** for time-of-day buckets                             -> US/EU clocks change Oct 25 / Nov 1 2026
# 6. **Diagnostics**: where the model loses (by vol regime, hour, spikes, worst misses)
#
# Each addition is evaluated cumulatively on the same untouched test set, so you can keep what helps and drop what doesn't.

# %%
import time, warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# ============================== CONFIG ==============================
SYMBOL           = "BTCUSDT"
DAYS             = 180          # >= 400 lets the model see both DST regimes (needs ~8+ GB RAM)
ATR_PERIOD       = 14           # period of the persistence baseline (Wilder)
HORIZON          = 14           # forecast the ATR of the next HORIZON minutes
CSV_PATH         = None         # optional Binance-format 1m klines CSV instead of the API
TRAIN_FRAC       = 0.70
VAL_FRAC         = 0.10         # remaining 20% = untouched test
ALPHAS           = [1, 10, 100, 1e3, 3e3, 1e4, 3e4, 1e5]
WF_STEP_DAYS     = 7
RUN_WALK_FORWARD = True
CLOCK_TZ         = "America/New_York"   # clock for time-of-day buckets. DST-aware. Use "UTC" to reproduce v1 behaviour.
EVENT_TZ         = "America/New_York"   # macro releases are scheduled in US Eastern time
EPS              = 1e-6

# %% [markdown]
# ## 1. Data (identical to v1)

# %%
BASE_URLS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api.binance.us"]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
NUM_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote"]


def fetch_klines(symbol, days):
    end = (int(time.time() // 60) - 1) * 60_000
    start = end - days * 1440 * 60_000
    sess, errors = requests.Session(), []
    for base in BASE_URLS:
        rows, t = [], start
        try:
            while t <= end:
                for attempt in range(4):
                    r = sess.get(f"{base}/api/v3/klines", timeout=20,
                                 params=dict(symbol=symbol, interval="1m", startTime=t, endTime=end, limit=1000))
                    if r.status_code in (418, 429):
                        time.sleep(2 * (attempt + 1)); continue
                    r.raise_for_status(); break
                else:
                    raise RuntimeError("rate limited")
                chunk = r.json()
                if not chunk:
                    break
                rows += chunk
                t = chunk[-1][0] + 60_000
                time.sleep(0.03)
            if rows:
                print(f"fetched {len(rows):,} bars from {base}")
                return pd.DataFrame(rows, columns=KLINE_COLS)
        except Exception as e:
            errors.append(f"{base}: {e}")
    raise RuntimeError("Could not download klines:\n  " + "\n  ".join(errors) +
                       "\nSet CSV_PATH to a Binance-format 1m klines CSV instead.")


def tidy(d):
    t = d["open_time"].astype("int64")
    unit = "us" if t.iloc[0] > 1e14 else "ms"
    d = d.copy()
    d["open_time"] = pd.to_datetime(t, unit=unit, utc=True)
    d[NUM_COLS] = d[NUM_COLS].astype(float)
    return d.drop_duplicates("open_time").set_index("open_time").sort_index()[NUM_COLS]


def load_csv(path):
    d = pd.read_csv(path)
    if "open_time" not in d.columns:
        d = pd.read_csv(path, header=None, names=KLINE_COLS)
    return d


raw = tidy(load_csv(CSV_PATH) if CSV_PATH else fetch_klines(SYMBOL, DAYS))
bars = raw.reindex(pd.date_range(raw.index[0], raw.index[-1], freq="1min"))
print(f"{SYMBOL}: {len(bars):,} minutes  {bars.index[0]} -> {bars.index[-1]}  |  missing bars: {int(bars['close'].isna().sum())}")

# %% [markdown]
# ## 2. Scheduled macro events
# Dates come from the official Federal Reserve FOMC calendar and the BLS release schedules (CPI, Employment Situation).
# FOMC: statement 14:00 ET, press conference 14:30 ET. CPI and jobs report: 08:30 ET. Times are converted with the
# US Eastern timezone rules, so the UTC time shifts automatically when US daylight saving ends (Nov 1, 2026).
# **Extend these lists yourself for dates beyond the ones shown** (a mismatch only means "no event flagged").

# %%
FOMC_DECISIONS = ["2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29",
                  "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]
CPI_DATES = ["2025-12-18", "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10",
             "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10"]
NFP_DATES = ["2025-12-16", "2026-01-09", "2026-02-11", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05",
             "2026-07-02", "2026-08-07", "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04"]


def _utc(dates, hhmm):
    return [pd.Timestamp(f"{d} {hhmm}", tz=EVENT_TZ).tz_convert("UTC") for d in dates]


EVENT_CLASSES = {"fomc": _utc(FOMC_DECISIONS, "14:00") + _utc(FOMC_DECISIONS, "14:30"),
                 "data": _utc(CPI_DATES, "08:30") + _utc(NFP_DATES, "08:30")}
for k, ts in EVENT_CLASSES.items():
    print(f"macro events '{k}' inside the data range: {sum(bars.index[0] <= t <= bars.index[-1] for t in ts)}")
if bars.index[0] < pd.Timestamp("2025-12-16", tz="UTC"):
    print("WARNING: data starts before 2025-12-16; CPI/jobs dates before that are not in the lists above, "
          "so those rows are treated as 'no event'. Add the earlier dates for best accuracy.")

# %% [markdown]
# ## 3. Features
# v1 feature set, plus: adaptive seasonality (`seas_*`), macro events (`ev_*`), DST-safe clock.
# Everything uses information available at the close of bar *t*; calendar and event features are known in advance.

# %%
def rwin(x, w, how="mean"):
    mp = w if w <= 60 else int(0.8 * w)
    return getattr(x.rolling(w, min_periods=mp), how)()


def calendar_marks(idx, H):
    ext = idx.append(pd.date_range(idx[-1] + pd.Timedelta(minutes=1), periods=H, freq="1min"))
    m, hr = np.asarray(ext.minute), np.asarray(ext.hour)
    marks = {"hour": m == 0, "qtr": m % 15 == 0, "fund": (m == 0) & np.isin(hr, [0, 8, 16])}   # funding marks are UTC-fixed
    out = {}
    for k, v in marks.items():
        roll = pd.Series(v.astype(float), index=ext).rolling(H).sum()
        out[f"{k}_mark_back"] = roll.reindex(idx)
        out[f"{k}_mark_fwd"] = roll.shift(-H).reindex(idx)
    return out


def event_features(idx, H, pre=60, post=60):
    """For each event class: event inside the forecast window / in the hour AFTER the window / in the last hour."""
    ext = idx.append(pd.date_range(idx[-1] + pd.Timedelta(minutes=1), periods=H + pre, freq="1min"))
    out = {}
    for k, times in EVENT_CLASSES.items():
        s = pd.Series(ext.isin(pd.DatetimeIndex(times)).astype(float), index=ext)
        out[f"ev_{k}_fwd"] = s.rolling(H).sum().shift(-H).reindex(idx)
        out[f"ev_{k}_pre"] = s.rolling(pre).sum().shift(-(H + pre)).reindex(idx)
        out[f"ev_{k}_post"] = s.rolling(post).sum().reindex(idx)
    return out


def tod_block(ts, per_bucket, prefix, index):
    ts = ts.tz_convert(CLOCK_TZ)                                   # DST-aware local clock
    nb = 1440 // per_bucket
    b = np.asarray((ts.hour * 60 + ts.minute) // per_bucket)
    wk = (np.asarray(ts.dayofweek) >= 5).astype(int)
    M = np.zeros((len(ts), 2 * nb), dtype=np.float32)
    M[np.arange(len(ts)), b + nb * wk] = 1.0
    cols = [f"{prefix}{i}_wd" for i in range(nb)] + [f"{prefix}{i}_we" for i in range(nb)]
    return pd.DataFrame(M, index=index, columns=cols)


def build_features(d, H=HORIZON, P=ATR_PERIOD):
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    v, n, tbv = d["volume"], d["trades"], d["taker_buy_base"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1, skipna=False)
    ntr = tr / pc
    r = np.log(c / pc)
    F = {}

    fwd_tr = tr.rolling(H, min_periods=H).mean().shift(-H)
    true_rel = fwd_tr / c
    meta = pd.DataFrame({"close": c, "true_rel": true_rel, "y": np.log(true_rel + EPS),
                         "base_wilder": tr.ewm(alpha=1 / P, adjust=False).mean() / c})

    # 1) HAR block
    for w in (1, 3, 5, 14, 30, 60, 120, 240, 720, 1440, 4320, 10080):
        F[f"ltr_{w}"] = np.log(rwin(ntr, w) + EPS)
    F["ltr_wilder"] = np.log(ntr.ewm(alpha=1 / P, adjust=False).mean() + EPS)
    F["ltr_ewm120"] = np.log(ntr.ewm(span=120, adjust=False).mean() + EPS)

    # 2) range / return based estimators
    hl, co = np.log(h / l), np.log(c / o)
    park = hl ** 2 / (4 * np.log(2))
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    for w in (5, 15, 60, 240, 1440):
        F[f"lpark_{w}"] = 0.5 * np.log(rwin(park, w).clip(lower=0) + 1e-14)
    for w in (15, 60, 240):
        F[f"lgk_{w}"] = 0.5 * np.log(rwin(gk, w).clip(lower=0) + 1e-14)
    bv = (np.pi / 2) * r.abs() * r.abs().shift(1)
    for w in (5, 15, 60, 240, 1440):
        F[f"lrv_{w}"] = 0.5 * np.log(rwin(r ** 2, w) + 1e-14)
    for w in (60, 240):
        F[f"lbv_{w}"] = 0.5 * np.log(rwin(bv, w) + 1e-14)
        F[f"ljump_{w}"] = F[f"lrv_{w}"] - F[f"lbv_{w}"]

    # 3) spikes, vol-of-vol, ranges
    lt1 = np.log(ntr + EPS)
    for w in (30, 120):
        F[f"lmax_{w}"] = np.log(rwin(ntr, w, "max") + EPS) - F[f"ltr_{w}"]
    for w in (60, 240):
        F[f"vov_{w}"] = rwin(lt1, w, "std")
    for w in (15, 60, 240, 1440):
        hi, lo = rwin(h, w, "max"), rwin(l, w, "min")
        F[f"lrange_{w}"] = np.log((hi - lo) / c + EPS)
        if w in (60, 240):
            F[f"rpos_{w}"] = (c - lo) / (hi - lo)

    # 4) volume / trades / order flow
    v1440, n1440 = rwin(v, 1440), rwin(n, 1440)
    for w in (1, 5, 15, 60):
        F[f"vol_rel_{w}"] = np.log((rwin(v, w) / v1440).clip(lower=1e-3))
        F[f"trd_rel_{w}"] = np.log((rwin(n, w) / n1440).clip(lower=1e-3))
    for w in (15, 60):
        F[f"size_rel_{w}"] = (np.log((rwin(v, w) / rwin(n, w)).clip(lower=1e-12))
                              - np.log((v1440 / n1440).clip(lower=1e-12)))
    for w in (5, 15, 60):
        sv = v.rolling(w, min_periods=w).sum()
        ofi = (2 * tbv.rolling(w, min_periods=w).sum() - sv) / sv
        F[f"ofi_{w}"], F[f"absofi_{w}"] = ofi, ofi.abs()

    # 5) returns / leverage effect
    vol240 = rwin(ntr, 240)
    for w in (1, 5, 15, 60, 240):
        rw = np.log(c / c.shift(w))
        F[f"ret_{w}"], F[f"negret_{w}"] = rw, rw.clip(upper=0)
        F[f"absz_{w}"] = rw.abs() / (vol240 * np.sqrt(w))

    # 6) NEW: adaptive seasonality = what the forecast-window level actually was at this clock time yesterday / last week.
    #    y[t-k] is a label whose window ended before t, so it is fully known at t (no leakage).
    ysm = meta["y"].rolling(31, center=True, min_periods=21).mean()      # +-15 min smoothing
    F["seas_d1_fwd"] = ysm.shift(1440)
    F["seas_d3_fwd"] = (ysm.shift(1440) + ysm.shift(2880) + ysm.shift(4320)) / 3
    F["seas_w1_fwd"] = ysm.shift(10080)
    F["seas_d1_back"] = ysm.shift(1440 + H)
    F["seas_w1_back"] = ysm.shift(10080 + H)

    # 7) calendar + macro events
    idx = d.index
    F.update(calendar_marks(idx, H))
    F.update(event_features(idx, H))
    X = pd.DataFrame(F, index=idx)
    ts_f = idx + pd.Timedelta(minutes=(H + 1) // 2)
    ts_b = idx - pd.Timedelta(minutes=H // 2)
    dow = np.eye(7, dtype=np.float32)[np.asarray(ts_f.tz_convert(CLOCK_TZ).dayofweek)]
    X = pd.concat([X,
                   pd.DataFrame(dow, index=idx, columns=[f"dow_{i}" for i in range(7)]),
                   tod_block(ts_f, 15, "todf", idx),
                   tod_block(ts_b, 60, "todb", idx)], axis=1)
    return X.replace([np.inf, -np.inf], np.nan), meta


X_full, meta_full = build_features(bars)
ok = X_full.notna().all(axis=1) & meta_full[["y", "base_wilder", "true_rel"]].notna().all(axis=1)
x_last = X_full.iloc[[-1]].copy()                       # latest bar, used for the live forecast at the end
X, meta = X_full[ok], meta_full[ok]
del X_full, meta_full
cols = list(X.columns)
no_clip = np.asarray(X.columns.str.startswith(("todf", "todb", "dow_", "ev_")))   # never winsorise 0/1 flags
is_dummy = np.asarray(X.columns.str.startswith(("todf", "todb", "dow_")))
Xn, yn, idx_all = X.to_numpy(np.float64), meta["y"].to_numpy(), X.index
del X
print(f"features: {len(cols)}  |  usable rows: {len(idx_all):,}")

# %% [markdown]
# ## 4. Splits, model components, tuning helpers
# Chronological split with a `HORIZON`-row purge before every boundary (labels look `HORIZON` minutes ahead).

# %%
H, n = HORIZON, len(idx_all)
i_tr, i_va = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
sl_tr, sl_va, sl_fit, sl_te = slice(0, i_tr - H), slice(i_tr, i_va - H), slice(0, i_va - H), slice(i_va, n)
print(f"train {i_tr:,} | val {i_va - i_tr:,} | test {n - i_va:,} rows | test: {idx_all[i_va]} -> {idx_all[-1]}")

SPLINE_FEATURES = ["ltr_1", "ltr_5", "ltr_14", "ltr_60", "ltr_240", "ltr_1440", "lpark_15", "vol_rel_5",
                   "trd_rel_5", "absz_5", "absz_15", "ljump_60", "lmax_30", "vov_60", "ret_15", "ofi_15"]


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lo=0.001, hi=0.999, skip=None):
        self.lo, self.hi, self.skip = lo, hi, skip

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lo_, self.hi_ = np.quantile(X, self.lo, axis=0), np.quantile(X, self.hi, axis=0)
        if self.skip is not None:
            self.lo_[self.skip], self.hi_[self.skip] = -np.inf, np.inf
        return self

    def transform(self, X):
        return np.clip(np.asarray(X), self.lo_, self.hi_)


class SplineAugment(BaseEstimator, TransformerMixin):
    """Append cubic B-spline bases of selected columns (quantile knots). The model stays linear in its parameters."""
    def __init__(self, idx=None, n_knots=6, degree=3):
        self.idx, self.n_knots, self.degree = idx, n_knots, degree

    def fit(self, X, y=None):
        self.st_ = SplineTransformer(n_knots=self.n_knots, degree=self.degree, knots="quantile",
                                     extrapolation="constant", include_bias=False).fit(np.asarray(X)[:, self.idx])
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.hstack([X, self.st_.transform(X[:, self.idx])])


def make_prep(cidx, spl):
    names = [cols[i] for i in cidx]
    steps = [("clip", Winsorizer(skip=no_clip[list(cidx)]))]
    if spl:
        steps.append(("spline", SplineAugment(idx=[names.index(c) for c in SPLINE_FEATURES])))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def make_pipe(est, cidx, spl=False):
    return Pipeline(make_prep(cidx, spl).steps + [("est", est)])


def clip_y(y):
    lo, hi = np.quantile(y, [0.0005, 0.9995])
    return np.clip(y, lo, hi)


def to_level(yhat, smear):
    """log forecast -> ATR/price. `smear` (Duan) converts a median-type forecast into an unbiased mean forecast."""
    return np.maximum(np.exp(yhat) * smear - EPS, 1e-8)


def tune_alpha(cidx, spl):
    prep = make_prep(cidx, spl)
    Ztr, Zva = prep.fit_transform(Xn[sl_tr][:, cidx]), prep.transform(Xn[sl_va][:, cidx])
    yc = clip_y(yn[sl_tr])
    G, b = Ztr.T @ Ztr, Ztr.T @ (yc - yc.mean())
    scores = {a: np.sqrt(np.mean((yn[sl_va] - (Zva @ np.linalg.solve(G + a * np.eye(len(G)), b) + yc.mean())) ** 2))
              for a in ALPHAS}
    return min(scores, key=scores.get), scores


def fit_predict(model, cidx, fit_sl, eval_sl):
    Xf, yf = Xn[fit_sl][:, cidx], yn[fit_sl]
    model.fit(Xf, clip_y(yf))
    res = yf - model.predict(Xf)
    return model, np.mean(np.exp(res)), res, model.predict(Xn[eval_sl][:, cidx])


def fit_full(cidx, spl, alpha, fit_sl, eval_sl, bands=False):
    model, smear, res, yhat = fit_predict(make_pipe(Ridge(alpha=alpha), cidx, spl), cidx, fit_sl, eval_sl)
    out = dict(model=model, smear=smear, res=res, yhat=yhat, alpha=alpha, cidx=cidx, spl=spl)
    if bands:                                   # scale model: how big is the error likely to be at THIS moment?
        sm = make_pipe(Ridge(alpha=alpha), cidx, False)
        Xf, Xe = Xn[fit_sl][:, cidx], Xn[eval_sl][:, cidx]
        sm.fit(Xf, clip_y(np.log(np.abs(res) + 1e-3)))
        z = res / np.exp(sm.predict(Xf))
        out.update(scale=sm, zq=np.quantile(z, [0.05, 0.95]), sigma=np.exp(sm.predict(Xe)))
    return out


def metrics(true_rel, pred_rel, close):
    t = np.maximum(np.asarray(true_rel, float), 1e-8)
    p = np.maximum(np.asarray(pred_rel, float), 1e-8)
    c, err = np.asarray(close, float), p - t
    lt, lp = np.log(t), np.log(p)
    return {"R2_log": 1 - np.sum((lt - lp) ** 2) / np.sum((lt - lt.mean()) ** 2),
            "RMSE_log": np.sqrt(np.mean((lt - lp) ** 2)),
            "R2_level": 1 - np.sum(err ** 2) / np.sum((t - t.mean()) ** 2),
            "MAE_bps": np.mean(np.abs(err)) * 1e4, "RMSE_bps": np.sqrt(np.mean(err ** 2)) * 1e4,
            "MAPE_%": np.mean(np.abs(err) / t) * 100,
            "QLIKE": np.mean(t / p - np.log(t / p) - 1),
            "Spearman": pd.Series(t).corr(pd.Series(p), method="spearman"),
            "Bias_%": (p.mean() / t.mean() - 1) * 100,
            "MAE_price": np.mean(np.abs(err) * c), "RMSE_price": np.sqrt(np.mean((err * c) ** 2))}


# %% [markdown]
# ## 5. Improvement ladder (cumulative, same untouched test set)
# Each row adds one thing to the row above. Ridge strength is re-tuned on the validation block for every row.

# %%
all_idx = list(range(len(cols)))
base_idx = [i for i, c in enumerate(cols) if not c.startswith(("seas_", "ev_"))]
seas_idx = [i for i, c in enumerate(cols) if not c.startswith("ev_")]

LADDER = [("Ridge, v1 features (previous model)", base_idx, False),
          ("+ adaptive seasonality", seas_idx, False),
          ("+ macro events", all_idx, False),
          ("+ spline terms  (= v2 model)", all_idx, True)]
runs = {}
for name, cidx, spl in LADDER:
    t0 = time.time()
    alpha, scores = tune_alpha(cidx, spl)
    runs[name] = fit_full(cidx, spl, alpha, sl_fit, sl_te, bands=(name == LADDER[-1][0]))
    print(f"{name:40s} features={len(cidx):4d}  best alpha={alpha:g}  ({time.time() - t0:.0f}s)")

te_meta = meta.iloc[sl_te]
te_idx, te_true, te_close = te_meta.index, te_meta["true_rel"].to_numpy(), te_meta["close"].to_numpy()
V2 = LADDER[-1][0]
pred = {"Persistence (Wilder ATR14)": te_meta["base_wilder"].to_numpy()}
for name, _, _ in LADDER:
    pred[name] = to_level(runs[name]["yhat"], runs[name]["smear"])
pred["v2 median forecast (no smearing)"] = to_level(runs[V2]["yhat"], 1.0)

if RUN_WALK_FORWARD:
    print("walk-forward refits (v2 model):")
    r2 = runs[V2]
    wf_y, wf_s = np.full(n, np.nan), np.full(n, np.nan)
    for s in range(i_va, n, WF_STEP_DAYS * 1440):
        e, f_end = min(s + WF_STEP_DAYS * 1440, n), s - H
        _, sm_, _, pr_ = fit_predict(make_pipe(Ridge(alpha=r2["alpha"]), r2["cidx"], r2["spl"]), r2["cidx"],
                                     slice(0, f_end), slice(s, e))
        wf_y[s:e], wf_s[s:e] = pr_, sm_
        print(f"  fold {idx_all[s].date()} -> {idx_all[e - 1].date()}")
    pred["v2 + weekly walk-forward"] = to_level(wf_y[sl_te], wf_s[sl_te])

res = pd.DataFrame({k: metrics(te_true, p, te_close) for k, p in pred.items()}).T

# %% [markdown]
# ## 6. Test results

# %%
print(f"\n=== TEST: {SYMBOL}, forward ATR over next {H} min | {te_idx[0]:%Y-%m-%d} -> {te_idx[-1]:%Y-%m-%d} | {len(te_idx):,} bars ===")
print(res.round(4).to_string())

v1, per = "Ridge, v1 features (previous model)", "Persistence (Wilder ATR14)"
err_cols = ["RMSE_log", "MAE_bps", "RMSE_bps", "QLIKE"]
imp = pd.concat({"% better than persistence": 100 * (1 - res[err_cols] / res.loc[per, err_cols]),
                 "% better than v1": 100 * (1 - res[err_cols] / res.loc[v1, err_cols])}, axis=1)
print("\nImprovement (positive = lower error):")
print(imp.drop(index=per).round(1).to_string())

# %% [markdown]
# ## 7. Where does it lose?  (diagnostics on the v2 model)
# Note: the spike / collapse split uses the realised outcome, so it describes errors; it is not a tradable signal.

# %%
fin = runs[V2]
p, t, b = pred[V2], te_true, pred[per]
lt, lp, lb = np.log(np.maximum(t, 1e-8)), np.log(p), np.log(np.maximum(b, 1e-8))

print("\n--- A) By forecast-volatility decile (1 = quietest forecasts) ---")
dec, tot = pd.qcut(p, 10, labels=False), np.sum((p - t) ** 2)
tab = pd.DataFrame([{"decile": d + 1, "fcst_bps": p[dec == d].mean() * 1e4, "actual_bps": t[dec == d].mean() * 1e4,
                     "RMSE_log": np.sqrt(np.mean((lt - lp)[dec == d] ** 2)),
                     "RMSE_log_persist": np.sqrt(np.mean((lt - lb)[dec == d] ** 2)),
                     "share_of_sq_err_%": 100 * np.sum((p - t)[dec == d] ** 2) / tot} for d in range(10)])
print(tab.round(3).to_string(index=False))

print("\n--- B) Spikes and collapses relative to what persistence expected ---")
ratio, rows = t / np.maximum(b, 1e-8), []
for nm, m in {"spike    (actual > 2x persistence)": ratio > 2, "collapse (actual < 0.5x persistence)": ratio < 0.5,
              "normal": (ratio >= 0.5) & (ratio <= 2)}.items():
    rows.append({"regime": nm, "share_of_bars_%": 100 * m.mean(), "share_of_sq_err_%": 100 * np.sum((p - t)[m] ** 2) / tot,
                 "model_fcst/actual": p[m].mean() / t[m].mean(), "persist_fcst/actual": b[m].mean() / t[m].mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n--- C) Hours (UTC) with the largest model error ---")
byh = pd.DataFrame({"model": (lt - lp) ** 2, "persist": (lt - lb) ** 2}).groupby(np.asarray(te_idx.hour)).mean().pow(0.5)
byh["gain_vs_persist_%"] = 100 * (1 - byh["model"] / byh["persist"])
print(byh.sort_values("model", ascending=False).head(6).round(3).to_string())

print("\n--- D) Ten worst under-forecasts (distinct events, >60 min apart) ---")
err_log, kept = lt - lp, []
for i in np.argsort(-err_log):
    if all(abs(i - j) > 60 for j in kept):
        kept.append(i)
    if len(kept) == 10:
        break
ev_ci = [i for i, c in enumerate(cols) if c.startswith("ev_") and (c.endswith("_fwd") or c.endswith("_post"))]
Xte = Xn[sl_te]
print(pd.DataFrame({"time_utc": [te_idx[i] for i in kept], "actual_bps": t[kept] * 1e4, "model_bps": p[kept] * 1e4,
                    "persist_bps": b[kept] * 1e4,
                    "scheduled_event_nearby": [bool((Xte[i, ev_ci] > 0).any()) for i in kept]}).round(2).to_string(index=False))

# %% [markdown]
# ## 8. Prediction bands: constant width vs volatility-adaptive

# %%
yte, yh = yn[sl_te], fin["yhat"]
q_lo, q_hi = np.quantile(fin["res"], [0.05, 0.95])
const_in = (yte >= yh + q_lo) & (yte <= yh + q_hi)
zl, zh = fin["zq"]
adapt_lo, adapt_hi = yh + fin["sigma"] * zl, yh + fin["sigma"] * zh
adapt_in = (yte >= adapt_lo) & (yte <= adapt_hi)
terc = pd.qcut(yh, 3, labels=["low-vol fcst", "mid-vol fcst", "high-vol fcst"])
cov = pd.DataFrame({"constant": const_in, "adaptive": adapt_in}).groupby(np.asarray(terc)).mean() * 100
cov.loc["ALL"] = [100 * const_in.mean(), 100 * adapt_in.mean()]
print("90% band empirical coverage on test (%):")
print(cov.round(1).to_string())
print(f"mean band width in log units: constant={q_hi - q_lo:.3f}  adaptive={np.mean(adapt_hi - adapt_lo):.3f}")
slope, icpt = np.polyfit(yh, yte, 1)
print(f"Mincer-Zarnowitz (log): actual = {icpt:.3f} + {slope:.3f} * forecast   (ideal 0 + 1)")

# %%
fig, ax = plt.subplots(2, 3, figsize=(19, 10))
fig.suptitle(f"{SYMBOL} 1m forward-ATR({H}) | v2 linear model | test set", fontsize=14)
k = 3 * 1440
a = ax[0, 0]
a.plot(te_idx[-k:], t[-k:] * 1e4, lw=1, color="k", label="actual")
a.plot(te_idx[-k:], p[-k:] * 1e4, lw=1, color="tab:red", label="v2")
a.fill_between(te_idx[-k:], np.exp(adapt_lo)[-k:] * 1e4, np.exp(adapt_hi)[-k:] * 1e4,
               color="tab:red", alpha=0.15, label="90% adaptive band")
a.plot(te_idx[-k:], b[-k:] * 1e4, lw=.8, color="tab:blue", alpha=.6, label="persistence")
a.set_title("Last 3 days of test (bps of price)"); a.legend(); a.tick_params(axis="x", rotation=30)

a = ax[0, 1]
a.hexbin(yh, yte, gridsize=70, bins="log", cmap="viridis")
lim = [min(yte.min(), yh.min()), max(yte.max(), yh.max())]
a.plot(lim, lim, "r--", lw=1); a.set_xlabel("predicted log ATR%"); a.set_ylabel("actual log ATR%")
a.set_title(f"Predicted vs actual (log)  R2={res.loc[V2, 'R2_log']:.3f}")

a = ax[0, 2]
cal = pd.DataFrame({"pred": p * 1e4, "actual": t * 1e4}).groupby(dec).mean()
a.plot(cal["pred"], cal["actual"], "o-"); a.plot(cal["pred"], cal["pred"], "r--", lw=1)
a.set_xlabel("mean predicted (bps)"); a.set_ylabel("mean actual (bps)"); a.set_title("Calibration by forecast decile")

a = ax[1, 0]
byh[["model", "persist"]].plot(ax=a, marker="o"); a.set_title("RMSE (log) by hour, UTC"); a.set_xlabel("hour")

a = ax[1, 1]
lad = res.loc[[per] + [x[0] for x in LADDER], "RMSE_log"]
a.barh(range(len(lad)), lad.values[::-1], color=(["tab:blue"] + ["tab:gray"] * (len(lad) - 2) + ["tab:red"])[::-1])
a.set_yticks(range(len(lad))); a.set_yticklabels([s[:44] for s in lad.index[::-1]], fontsize=8)
a.set_xlim(lad.min() * 0.97, lad.max() * 1.01); a.set_title("RMSE (log) along the ladder, lower is better")

a = ax[1, 2]
coef = pd.Series(fin["model"].named_steps["est"].coef_[:len(cols)], index=cols)[~is_dummy]
coef.reindex(coef.abs().sort_values(ascending=False).index[:20])[::-1].plot.barh(ax=a)
a.set_title("Top-20 standardised linear coefficients")
plt.tight_layout(); plt.savefig("atr_test_results_v2.png", dpi=130); plt.show()

# %% [markdown]
# ## 9. Production fit + live forecast
# Refit on all data (with purge gap), save the model, forecast the next `HORIZON` minutes from the latest completed bar.

# %%
import joblib
prod = fit_full(fin["cidx"], fin["spl"], fin["alpha"], slice(0, n - H), slice(n - H, n), bands=True)
joblib.dump({"model": prod["model"], "smear": prod["smear"], "scale_model": prod["scale"], "z_q05_q95": prod["zq"],
             "columns": cols, "column_idx": fin["cidx"], "horizon": H, "symbol": SYMBOL, "clock_tz": CLOCK_TZ},
            "atr_model_v2.joblib")

if x_last.notna().all(axis=1).iloc[0]:
    xl = x_last[cols].to_numpy(np.float64)[:, fin["cidx"]]
    yl, sg = prod["model"].predict(xl)[0], float(np.exp(prod["scale"].predict(xl))[0])
    px = float(bars["close"].iloc[-1])
    lo, hi = np.exp(yl + sg * prod["zq"][0]) * px, np.exp(yl + sg * prod["zq"][1]) * px
    pt = to_level(yl, prod["smear"]) * px
    print(f"Bar {x_last.index[0]}  close={px:,.2f}\n"
          f"Forecast ATR over next {H} min: {pt:,.2f}  ({pt / px * 1e4:.2f} bps)   90% adaptive band: [{lo:,.2f}, {hi:,.2f}]")
    now = x_last.index[0]
    up = sorted((t_, k_) for k_, ts in EVENT_CLASSES.items() for t_ in ts if now < t_ <= now + pd.Timedelta(days=7))
    print("Scheduled macro events in the next 7 days (UTC):", [(f"{t_:%a %m-%d %H:%M}", k_) for t_, k_ in up] or "none")
else:
    print("Latest bar has NaN features (data gap): no live forecast.")
print("Saved: atr_model_v2.joblib, atr_test_results_v2.png")
