"""Polynomial regression for var1 and var2.

var2: search over feature subsets and degrees (plain OLS polynomial), pick by validation.
var1: compare plain OLS polynomial against regularized (Ridge / Lasso) polynomial models
      on the same 80/20 splits; regularization strength is chosen by CV inside the 80% part.
Final models are refit on ALL training data and used to predict the test files.
"""
import os
import warnings
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROLL = "BT2024148"                 # must match your csv file names exactly
DATA_DIR = "."                     # folder containing the csv files
SEEDS = [42, 0, 1, 2, 3]           # several 80/20 splits, results averaged
MAX_TERMS = 500                    # skip plain-OLS models with more terms than this
RUN_PREFIX_SWEEP = True            # "first k features" sweep for var2 (fast); reused if file exists
os.makedirs("figures", exist_ok=True)

ALPHAS = np.logspace(-5, 2, 30)


# ---------------------------------------------------------------------------
# model builders (all of them are polynomial regression)
# ---------------------------------------------------------------------------
def ols(d):
    return make_pipeline(PolynomialFeatures(d, include_bias=False), LinearRegression())


def ridge(d):
    # scale AFTER the polynomial expansion so the penalty treats every term fairly
    return make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(),
                         RidgeCV(alphas=ALPHAS))


def lasso(d):
    return make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(),
                         LassoCV(alphas=ALPHAS, cv=5, max_iter=100000, n_jobs=-1))


def load(prb):
    tr = pd.read_csv(f"{DATA_DIR}/{ROLL}_train_var{prb}.csv")
    te = pd.read_csv(f"{DATA_DIR}/{ROLL}_test_var{prb}.csv")
    print(f"\n===== var{prb} =====")
    print("train shape:", tr.shape, "| test shape:", te.shape)
    print("missing:", int(tr.isna().sum().sum()), "| duplicates:", int(tr.duplicated().sum()))
    print(tr.describe().T[["min", "max", "mean", "std"]])
    return tr, te


def save_predictions(model, te, feats, prb):
    pred = model.predict(te[feats].values)
    out = pd.DataFrame({"y": pred})
    assert len(out) == len(te), "row count mismatch"
    out.to_csv(f"{ROLL}_pred_var{prb}.csv", index=False)      # one column 'y', no index
    print(f"saved {ROLL}_pred_var{prb}.csv ({len(out)} rows)")


def report_and_plots(make, tr, feats, prb, label):
    """Train-vs-validation numbers and plots for the final model (seed 42 split)."""
    tr80, val20 = train_test_split(tr, test_size=0.2, random_state=42)
    m = make().fit(tr80[feats].values, tr80["y"].values)
    p_tr, p_val = m.predict(tr80[feats].values), m.predict(val20[feats].values)
    print(f"\n--- Final model ({label}): train vs validation (80/20, seed 42) ---")
    print(f"Train (80%)      : MSE={mean_squared_error(tr80['y'], p_tr):.5f}  R2={r2_score(tr80['y'], p_tr):.5f}")
    print(f"Validation (20%) : MSE={mean_squared_error(val20['y'], p_val):.5f}  R2={r2_score(val20['y'], p_val):.5f}")

    plt.figure(figsize=(5, 5))
    plt.scatter(val20["y"], p_val, s=12, alpha=0.7)
    lo, hi = val20["y"].min(), val20["y"].max()
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.xlabel("Actual y"); plt.ylabel("Predicted y")
    plt.title(f"var{prb}: predicted vs actual (validation)")
    plt.tight_layout(); plt.savefig(f"figures/var{prb}_pred_vs_actual.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(p_val, val20["y"] - p_val, s=12, alpha=0.7)
    plt.axhline(0, color="r", ls="--")
    plt.xlabel("Predicted y"); plt.ylabel("Residual")
    plt.title(f"var{prb}: residuals (validation)")
    plt.tight_layout(); plt.savefig(f"figures/var{prb}_residuals.png", dpi=150); plt.close()


# ---------------------------------------------------------------------------
# var2: first-k sweep + subset/degree search (plain OLS polynomial)
# ---------------------------------------------------------------------------
def prefix_sweep(df, feats_all, degrees):
    rows = []
    for seed in SEEDS:
        tr, val = train_test_split(df, test_size=0.2, random_state=seed)
        for k in range(1, len(feats_all) + 1):
            f = feats_all[:k]
            for d in degrees:
                m = ols(d).fit(tr[f].values, tr["y"].values)
                p = m.predict(val[f].values)
                rows.append(dict(seed=seed, k=k, deg=d,
                                 mse=mean_squared_error(val["y"], p), r2=r2_score(val["y"], p)))
    return pd.DataFrame(rows).groupby(["k", "deg"])[["mse", "r2"]].mean().reset_index()


def plot_sweep(summary, prb):
    plt.figure(figsize=(7, 4.5))
    for k, g in summary.groupby("k"):
        plt.plot(g["deg"], g["mse"], marker="o", label=f"first {k} feature(s)")
    plt.yscale("log")
    plt.xlabel("Polynomial degree"); plt.ylabel("Validation MSE (log scale)")
    plt.title(f"var{prb}: validation MSE vs degree")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"figures/var{prb}_degree_sweep.png", dpi=150); plt.close()


def subset_search(df, feats_all, degrees):
    rows = []
    for seed in SEEDS:
        tr, val = train_test_split(df, test_size=0.2, random_state=seed)
        for r in range(1, len(feats_all) + 1):
            for f in combinations(feats_all, r):
                f = list(f)
                for d in degrees:
                    terms = comb(r + d, d) - 1
                    if terms > MAX_TERMS:
                        continue
                    m = ols(d).fit(tr[f].values, tr["y"].values)
                    p = m.predict(val[f].values)
                    rows.append(dict(seed=seed, feats="+".join(f), n_feats=r, deg=d, terms=terms,
                                     mse=mean_squared_error(val["y"], p), r2=r2_score(val["y"], p)))
    res = pd.DataFrame(rows)
    return (res.groupby(["feats", "n_feats", "deg", "terms"])[["mse", "r2"]]
               .mean().reset_index().sort_values("mse").reset_index(drop=True))


def run_var2():
    tr, te = load(2)
    feats_all = [c for c in tr.columns if c.startswith("x")][:3]

    pfile = "figures/var2_sweep_table.csv"
    if RUN_PREFIX_SWEEP or not os.path.exists(pfile):
        summary = prefix_sweep(tr, feats_all, range(1, 21))
        summary.to_csv(pfile, index=False)
        plot_sweep(summary, 2)

    cand = subset_search(tr, feats_all, range(1, 21))
    cand.to_csv("figures/var2_subset_search.csv", index=False)
    print("\nTop 10 (feature subset, degree) by validation MSE:")
    print(cand.head(10).to_string(index=False))

    best = cand.iloc[0]
    near = cand[cand["mse"] <= best["mse"] * 1.02].sort_values(["terms", "mse"])
    ch = near.iloc[0]
    feats, d = ch["feats"].split("+"), int(ch["deg"])
    print(f"\nFINAL choice: {feats}, degree {d} ({int(ch['terms'])} terms), "
          f"MSE={ch['mse']:.5f}, R2={ch['r2']:.5f}")

    report_and_plots(lambda: ols(d), tr, feats, 2, f"OLS deg {d}")
    final = ols(d).fit(tr[feats].values, tr["y"].values)
    save_predictions(final, te, feats, 2)


# ---------------------------------------------------------------------------
# var1: plain OLS vs regularized polynomial models
# ---------------------------------------------------------------------------
VAR1_CANDIDATES = {
    "OLS deg4 (baseline)": lambda: ols(4),
    "Ridge deg4": lambda: ridge(4),
    "Ridge deg5": lambda: ridge(5),
    "Ridge deg6": lambda: ridge(6),
    "Lasso deg4": lambda: lasso(4),
    "Lasso deg5": lambda: lasso(5),
}


def run_var1():
    tr, te = load(1)
    feats = [f"x{i}" for i in range(1, 7)]       # subset search showed all 6 are needed
    X, y = tr[feats].values, tr["y"].values

    rows = []
    for name, make in VAR1_CANDIDATES.items():
        for seed in SEEDS:
            Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=seed)
            m = make().fit(Xa, ya)
            rows.append(dict(model=name, seed=seed,
                             train_mse=mean_squared_error(ya, m.predict(Xa)),
                             val_mse=mean_squared_error(yb, m.predict(Xb)),
                             val_r2=r2_score(yb, m.predict(Xb))))
        print(f"done: {name}")

    summ = (pd.DataFrame(rows).groupby("model")[["train_mse", "val_mse", "val_r2"]].mean()
              .assign(gap=lambda d: d.val_mse - d.train_mse).sort_values("val_mse"))
    summ.to_csv("figures/var1_model_comparison.csv")
    print(f"\nAverage over {len(SEEDS)} splits (lower val_mse is better; smaller gap = less overfitting):")
    print(summ.round(4).to_string())

    best_name = summ.index[0]
    print(f"\nFINAL choice: {best_name}")
    report_and_plots(VAR1_CANDIDATES[best_name], tr, feats, 1, best_name)

    final = VAR1_CANDIDATES[best_name]().fit(X, y)
    last = final[-1]
    if hasattr(last, "alpha_"):
        print("chosen alpha:", last.alpha_)
    if hasattr(last, "coef_"):
        print("non-zero coefficients:", int((np.abs(last.coef_) > 1e-8).sum()), "of", len(last.coef_))
    save_predictions(final, te, feats, 1)


if __name__ == "__main__":
    run_var1()
    run_var2()