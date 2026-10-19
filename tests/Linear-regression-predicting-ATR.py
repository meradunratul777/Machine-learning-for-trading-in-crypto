# %% [markdown]
# # 1-minute crypto ATR forecaster: regularised linear regression (HAR + seasonality)
#
# **Target.** Forward ATR = average true range of the *next* `HORIZON` one-minute bars, divided by
# the current close, in log space. Predicting the ATR you already know (the trailing one) is trivial,
# and it is not what a market maker needs: you quote *now* for the volatility that arrives *next*.
#
# **Model.** Ridge-regularised linear regression on ~330 leak-free features:
# multi-horizon log-volatility (HAR), range/return/jump estimators, volume and order-flow,
# leverage-effect terms, and calendar features known in advance (15-min time-of-day x weekday/weekend,
# hour/funding marks inside the forecast window).
#
# **Evaluation.** Chronological train / validation / test with a purge gap, compared against
# persistence (current Wilder ATR) and a classic HAR baseline, plus a weekly walk-forward refit.

# %%
import time, warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

# ============================== CONFIG ==============================
SYMBOL       = "BTCUSDT"   # any Binance spot symbol
DAYS         = 180         # days of 1m history to download (more = better seasonality estimates)
ATR_PERIOD   = 14          # period of the "current ATR" baseline (Wilder)
HORIZON      = 14          # predict the ATR of the next HORIZON minutes
CSV_PATH     = None        # optional: path to your own Binance-format 1m klines CSV instead of the API
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.10        # remaining 20% = untouched test set
ALPHAS       = [0.1, 1, 10, 100, 1e3, 1e4, 1e5]   # ridge strengths searched on the validation block
WF_STEP_DAYS = 7           # walk-forward: refit every N days during the test period
EPS          = 1e-6

# %% [markdown]
# ## 1. Data (Binance 1m klines)
# Tries Binance's public market-data mirror first (works from Colab's US servers), then fallbacks.

# %%
BASE_URLS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api.binance.us"]
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
NUM_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote"]


def fetch_klines(symbol, days):
    end = (int(time.time() // 60) - 1) * 60_000          # open time of last COMPLETED 1m bar
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
    unit = "us" if t.iloc[0] > 1e14 else "ms"             # newer Binance dumps use microseconds
    d = d.copy()
    d["open_time"] = pd.to_datetime(t, unit=unit, utc=True)
    d[NUM_COLS] = d[NUM_COLS].astype(float)
    return d.drop_duplicates("open_time").set_index("open_time").sort_index()[NUM_COLS]


def load_csv(path):
    d = pd.read_csv(path)
    if "open_time" not in d.columns:                      # header-less Binance dump
        d = pd.read_csv(path, header=None, names=KLINE_COLS)
    return d


raw = tidy(load_csv(CSV_PATH) if CSV_PATH else fetch_klines(SYMBOL, DAYS))

# put on a strict 1-minute grid; missing minutes become NaN (rows touching them get dropped later)
full = pd.date_range(raw.index[0], raw.index[-1], freq="1min")
bars = raw.reindex(full)
print(f"{SYMBOL}: {len(bars):,} minutes  {bars.index[0]} -> {bars.index[-1]}  |  missing bars: {int(bars['close'].isna().sum())}")

# %% [markdown]
# ## 2. Feature engineering
# Everything below uses information available at the close of bar *t* only. The target uses bars *t+1 ... t+H*.
# Calendar features are deterministic functions of the clock, so they are known in advance.

# %%
def rwin(x, w, how="mean"):
    """Rolling stat; short windows must be complete, long ones tolerate 20% missing."""
    mp = w if w <= 60 else int(0.8 * w)
    return getattr(x.rolling(w, min_periods=mp), how)()


def calendar_marks(idx, H):
    """# of hour / quarter-hour / funding (00,08,16 UTC) bars inside the BACKWARD and FORWARD H-minute windows."""
    ext = idx.append(pd.date_range(idx[-1] + pd.Timedelta(minutes=1), periods=H, freq="1min"))
    m, hr = np.asarray(ext.minute), np.asarray(ext.hour)
    marks = {"hour": m == 0, "qtr": m % 15 == 0, "fund": (m == 0) & np.isin(hr, [0, 8, 16])}
    out = {}
    for k, v in marks.items():
        roll = pd.Series(v.astype(float), index=ext).rolling(H).sum()
        out[f"{k}_mark_back"] = roll.reindex(idx)
        out[f"{k}_mark_fwd"] = roll.shift(-H).reindex(idx)
    return out


def tod_block(ts, per_bucket, prefix, index):
    """One-hot time-of-day bucket x (weekday | weekend)."""
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
    ntr = tr / pc                                          # true range as a fraction of price
    r = np.log(c / pc)
    F = {}

    # ---- target + baseline (kept out of X) ----
    fwd_tr = tr.rolling(H, min_periods=H).mean().shift(-H)  # mean TR of bars t+1..t+H  (forward ATR, price units)
    true_rel = fwd_tr / c
    meta = pd.DataFrame({"close": c, "fwd_atr_price": fwd_tr, "true_rel": true_rel,
                         "y": np.log(true_rel + EPS),
                         "base_wilder": tr.ewm(alpha=1 / P, adjust=False).mean() / c})

    # ---- 1) HAR block: log average true range over many horizons (1m .. 1 week) ----
    for w in (1, 3, 5, 14, 30, 60, 120, 240, 720, 1440, 4320, 10080):
        F[f"ltr_{w}"] = np.log(rwin(ntr, w) + EPS)
    F["ltr_wilder"] = np.log(ntr.ewm(alpha=1 / P, adjust=False).mean() + EPS)
    F["ltr_ewm120"] = np.log(ntr.ewm(span=120, adjust=False).mean() + EPS)

    # ---- 2) efficient range-based and return-based vol estimators ----
    hl, co = np.log(h / l), np.log(c / o)
    park = hl ** 2 / (4 * np.log(2))
    gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
    for w in (5, 15, 60, 240, 1440):
        F[f"lpark_{w}"] = 0.5 * np.log(rwin(park, w).clip(lower=0) + 1e-14)
    for w in (15, 60, 240):
        F[f"lgk_{w}"] = 0.5 * np.log(rwin(gk, w).clip(lower=0) + 1e-14)
    bv = (np.pi / 2) * r.abs() * r.abs().shift(1)           # bipower variation (jump-robust)
    for w in (5, 15, 60, 240, 1440):
        F[f"lrv_{w}"] = 0.5 * np.log(rwin(r ** 2, w) + 1e-14)
    for w in (60, 240):
        F[f"lbv_{w}"] = 0.5 * np.log(rwin(bv, w) + 1e-14)
        F[f"ljump_{w}"] = F[f"lrv_{w}"] - F[f"lbv_{w}"]      # >0 means recent jumps

    # ---- 3) spikes, vol-of-vol, range / position inside range ----
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

    # ---- 4) volume, trade count, trade size, taker order-flow imbalance ----
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

    # ---- 5) returns: leverage effect (negative returns raise vol) and vol-standardised shocks ----
    vol240 = rwin(ntr, 240)
    for w in (1, 5, 15, 60, 240):
        rw = np.log(c / c.shift(w))
        F[f"ret_{w}"], F[f"negret_{w}"] = rw, rw.clip(upper=0)
        F[f"absz_{w}"] = rw.abs() / (vol240 * np.sqrt(w))

    # ---- 6) calendar (known in advance) ----
    idx = d.index
    F.update(calendar_marks(idx, H))
    X = pd.DataFrame(F, index=idx)
    ts_f = idx + pd.Timedelta(minutes=(H + 1) // 2)         # centre of the FORWARD window
    ts_b = idx - pd.Timedelta(minutes=H // 2)               # centre of the BACKWARD window
    dow = np.eye(7, dtype=np.float32)[np.asarray(ts_f.dayofweek)]
    X = pd.concat([X,
                   pd.DataFrame(dow, index=idx, columns=[f"dow_{i}" for i in range(7)]),
                   tod_block(ts_f, 15, "todf", idx),        # 15-min buckets, forward window
                   tod_block(ts_b, 60, "todb", idx)],       # 60-min buckets, backward window
                  axis=1)
    return X.replace([np.inf, -np.inf], np.nan), meta


X_full, meta_full = build_features(bars)
ok = X_full.notna().all(axis=1) & meta_full[["y", "base_wilder", "true_rel"]].notna().all(axis=1)
X, meta = X_full[ok], meta_full[ok]
is_dummy = np.asarray(X.columns.str.startswith(("todf", "todb", "dow_")))
cols = list(X.columns)
Xn, yn = X.to_numpy(np.float64), meta["y"].to_numpy()
print(f"features: {X.shape[1]}  |  usable rows: {len(X):,}  (dropped {len(X_full) - len(X):,} for warm-up / gaps / tail)")

# %% [markdown]
# ## 3. Splits, preprocessing, ridge-strength search
# Chronological split. A purge gap of `HORIZON` rows is removed before every boundary, because a
# training label at row *t* looks `HORIZON` minutes into the future.

# %%
H = HORIZON
n = len(X)
i_tr, i_va = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
sl_tr, sl_va, sl_fit, sl_te = slice(0, i_tr - H), slice(i_tr, i_va - H), slice(0, i_va - H), slice(i_va, n)
print(f"train {i_tr:,} | val {i_va - i_tr:,} | test {n - i_va:,} rows")
print(f"test period: {X.index[i_va]}  ->  {X.index[-1]}")


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip continuous features to training quantiles (crypto has flash-crash outliers)."""
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


def make_pipe(est):
    return make_pipeline(Winsorizer(skip=is_dummy), StandardScaler(), est)


def clip_y(y):  # mild target winsorisation for fitting only
    lo, hi = np.quantile(y, [0.0005, 0.9995])
    return np.clip(y, lo, hi)


def to_level(yhat, smear):
    """log-forecast -> ATR/price. `smear` (Duan) turns the median-type forecast into an unbiased mean forecast."""
    return np.maximum(np.exp(yhat) * smear - EPS, 1e-8)


# ridge-strength search on the validation block (closed-form for all alphas at once)
prep = make_pipeline(Winsorizer(skip=is_dummy), StandardScaler())
Ztr, Zva = prep.fit_transform(Xn[sl_tr]), prep.transform(Xn[sl_va])
yc = clip_y(yn[sl_tr])
G, b = Ztr.T @ Ztr, Ztr.T @ (yc - yc.mean())
val_rows = []
for a in ALPHAS:
    w = np.linalg.solve(G + a * np.eye(G.shape[0]), b)
    val_rows.append((a, np.sqrt(np.mean((yn[sl_va] - (Zva @ w + yc.mean())) ** 2))))
val_tbl = pd.DataFrame(val_rows, columns=["alpha", "val_RMSE_log"])
best_alpha = float(val_tbl.loc[val_tbl["val_RMSE_log"].idxmin(), "alpha"])
print(val_tbl.to_string(index=False), f"\n-> best alpha = {best_alpha:g}")
del Ztr, Zva, G

# %% [markdown]
# ## 4. Fit final models on train + validation and predict the untouched test set

# %%
def fit_predict(model, col_idx, fit_sl, eval_sl):
    Xf, yf = Xn[fit_sl][:, col_idx], yn[fit_sl]
    model.fit(Xf, clip_y(yf))
    res = yf - model.predict(Xf)
    return model, np.mean(np.exp(res)), res, model.predict(Xn[eval_sl][:, col_idx])


all_idx = list(range(len(cols)))
HAR_COLS = ["ltr_5", "ltr_30", "ltr_240", "ltr_1440"]
har_idx = [cols.index(c) for c in HAR_COLS]

har_m, har_sm, _, har_yhat = fit_predict(LinearRegression(), har_idx, sl_fit, sl_te)
ols_m, ols_sm, _, ols_yhat = fit_predict(make_pipe(LinearRegression()), all_idx, sl_fit, sl_te)
rdg_m, rdg_sm, rdg_res, rdg_yhat = fit_predict(make_pipe(Ridge(alpha=best_alpha)), all_idx, sl_fit, sl_te)

# weekly walk-forward refit (expanding window) over the same test period
def walk_forward(alpha, step):
    yhat, sm = np.full(n, np.nan), np.full(n, np.nan)
    for s in range(i_va, n, step):
        e, f_end = min(s + step, n), s - H
        m, smear, _, pred = fit_predict(make_pipe(Ridge(alpha=alpha)), all_idx, slice(0, f_end), slice(s, e))
        yhat[s:e], sm[s:e] = pred, smear
        print(f"  walk-forward fold {X.index[s].date()} -> {X.index[e - 1].date()}  (fit rows: {f_end:,})")
    return yhat[sl_te], sm[sl_te]


print("walk-forward refits:")
wf_yhat, wf_sm = walk_forward(best_alpha, WF_STEP_DAYS * 1440)


def metrics(true_rel, pred_rel, close):
    t = np.maximum(np.asarray(true_rel, float), 1e-8)
    p = np.maximum(np.asarray(pred_rel, float), 1e-8)
    c, err = np.asarray(close, float), p - t
    lt, lp = np.log(t), np.log(p)
    return {
        "R2_log": 1 - np.sum((lt - lp) ** 2) / np.sum((lt - lt.mean()) ** 2),
        "RMSE_log": np.sqrt(np.mean((lt - lp) ** 2)),
        "R2_level": 1 - np.sum(err ** 2) / np.sum((t - t.mean()) ** 2),
        "MAE_bps": np.mean(np.abs(err)) * 1e4,
        "RMSE_bps": np.sqrt(np.mean(err ** 2)) * 1e4,
        "MAPE_%": np.mean(np.abs(err) / t) * 100,
        "QLIKE": np.mean(t / p - np.log(t / p) - 1),
        "Spearman": pd.Series(t).corr(pd.Series(p), method="spearman"),
        "Bias_%": (p.mean() / t.mean() - 1) * 100,
        "MAE_price": np.mean(np.abs(err) * c),
        "RMSE_price": np.sqrt(np.mean((err * c) ** 2)),
    }


te_meta = meta.iloc[sl_te]
te_idx, te_true, te_close = te_meta.index, te_meta["true_rel"].to_numpy(), te_meta["close"].to_numpy()
pred = {
    "Persistence (Wilder ATR14)": te_meta["base_wilder"].to_numpy(),
    "HAR-OLS (4 lags)": to_level(har_yhat, har_sm),
    "OLS (all features)": to_level(ols_yhat, ols_sm),
    "Ridge (all features)": to_level(rdg_yhat, rdg_sm),
    "Ridge + weekly walk-forward": to_level(wf_yhat, wf_sm),
}
res = pd.DataFrame({k: metrics(te_true, p, te_close) for k, p in pred.items()}).T

# %% [markdown]
# ## 5. Test results
# Units: `bps` = basis points of price; `price` = quote currency (USDT). `QLIKE` is the standard
# loss for volatility forecasts (lower is better). `Bias_%` should be near 0 for market-making use.

# %%
print(f"\n=== TEST RESULTS: {SYMBOL}, forward ATR over next {H} min | "
      f"{te_idx[0]:%Y-%m-%d} -> {te_idx[-1]:%Y-%m-%d} | {len(te_idx):,} bars ===")
print(res.round(4).to_string())

base, mod = res.loc["Persistence (Wilder ATR14)"], res.loc["Ridge (all features)"]
print(f"\nRidge vs persistence:  RMSE(log) {100 * (1 - mod['RMSE_log'] / base['RMSE_log']):+.1f}%  |  "
      f"MAE(bps) {100 * (1 - mod['MAE_bps'] / base['MAE_bps']):+.1f}%  |  "
      f"QLIKE {100 * (1 - mod['QLIKE'] / base['QLIKE']):+.1f}%   (positive = better)")

yte = yn[sl_te]
slope, icpt = np.polyfit(rdg_yhat, yte, 1)
q_lo, q_hi = np.quantile(rdg_res, [0.05, 0.95])
cover = np.mean((yte >= rdg_yhat + q_lo) & (yte <= rdg_yhat + q_hi))
print(f"Mincer-Zarnowitz (log): actual = {icpt:.3f} + {slope:.3f} * forecast   (ideal: 0 + 1.000)")
print(f"90% band [exp(yhat+q05), exp(yhat+q95)] empirical test coverage: {100 * cover:.1f}%")

# %%
fig, ax = plt.subplots(2, 3, figsize=(19, 10))
fig.suptitle(f"{SYMBOL} 1m forward-ATR({H}) | Ridge linear model | test set", fontsize=14)

k = 3 * 1440
a = ax[0, 0]
a.plot(te_idx[-k:], te_true[-k:] * 1e4, lw=1, label="actual", color="k")
a.plot(te_idx[-k:], pred["Ridge (all features)"][-k:] * 1e4, lw=1, label="Ridge", color="tab:red")
a.plot(te_idx[-k:], pred["Persistence (Wilder ATR14)"][-k:] * 1e4, lw=.8, label="persistence", color="tab:blue", alpha=.6)
a.set_title("Last 3 days of test (bps of price)"); a.legend(); a.tick_params(axis="x", rotation=30)

a = ax[0, 1]
a.hexbin(rdg_yhat, yte, gridsize=70, bins="log", cmap="viridis")
lim = [min(yte.min(), rdg_yhat.min()), max(yte.max(), rdg_yhat.max())]
a.plot(lim, lim, "r--", lw=1); a.set_xlabel("predicted log ATR%"); a.set_ylabel("actual log ATR%")
a.set_title(f"Predicted vs actual (log)  R2={res.loc['Ridge (all features)', 'R2_log']:.3f}")

a = ax[0, 2]
dec = pd.qcut(pred["Ridge (all features)"], 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"pred": pred["Ridge (all features)"] * 1e4, "actual": te_true * 1e4}).groupby(dec).mean()
a.plot(cal["pred"], cal["actual"], "o-", label="Ridge")
a.plot([cal["pred"].min(), cal["pred"].max()], [cal["pred"].min(), cal["pred"].max()], "r--", lw=1, label="perfect")
a.set_xlabel("mean predicted (bps)"); a.set_ylabel("mean actual (bps)"); a.set_title("Calibration by forecast decile"); a.legend()

a = ax[1, 0]
sq = pd.DataFrame({"Ridge": (yte - rdg_yhat) ** 2,
                   "Persistence": (yte - np.log(pred["Persistence (Wilder ATR14)"] + EPS)) ** 2}, index=te_idx)
sq.groupby(np.asarray(te_idx.hour)).mean().pow(0.5).plot(ax=a, marker="o")
a.set_title("RMSE (log) by hour of day, UTC"); a.set_xlabel("hour")

a = ax[1, 1]
sq.resample("D").mean().pow(0.5).plot(ax=a, marker=".")
a.set_title("Daily RMSE (log): stability over the test period"); a.set_xlabel("")

a = ax[1, 2]
coef = pd.Series(rdg_m.named_steps["ridge"].coef_, index=cols)[~is_dummy]
coef.reindex(coef.abs().sort_values(ascending=False).index[:20])[::-1].plot.barh(ax=a)
a.set_title("Top-20 standardised coefficients (collinear windows share weight)")

plt.tight_layout(); plt.savefig("atr_test_results.png", dpi=130); plt.show()

# %% [markdown]
# ## 6. Production fit + live forecast
# Refits on *all* data (with the purge gap), saves the model, and forecasts the ATR of the next
# `HORIZON` minutes from the latest completed bar.

# %%
import joblib
prod, prod_sm, prod_res, _ = fit_predict(make_pipe(Ridge(alpha=best_alpha)), all_idx, slice(0, n - H), slice(n - H, n))
lo_q, hi_q = np.quantile(prod_res, [0.05, 0.95])
joblib.dump({"model": prod, "smear": prod_sm, "q05": lo_q, "q95": hi_q, "columns": cols,
             "horizon": H, "symbol": SYMBOL}, "atr_model.joblib")

x_last = X_full.iloc[[-1]]
if x_last.notna().all(axis=1).iloc[0]:
    yh = prod.predict(x_last[cols].to_numpy(np.float64))[0]
    px = float(bars["close"].iloc[-1])
    point = to_level(yh, prod_sm) * px
    band = (np.exp(yh + lo_q) * px, np.exp(yh + hi_q) * px)
    print(f"Bar {x_last.index[0]}  close={px:,.2f}\n"
          f"Forecast ATR over next {H} min: {point:,.2f}  ({point / px * 1e4:.2f} bps)   "
          f"90% band: [{band[0]:,.2f}, {band[1]:,.2f}]")
else:
    print("Latest bar has NaN features (gap in data) - no live forecast.")
print("Saved: atr_model.joblib, atr_test_results.png")
