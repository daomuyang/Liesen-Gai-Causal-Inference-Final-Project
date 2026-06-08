#!/usr/bin/env python3
"""
Causal analysis: physical activity -> CVD.
Reproducible pipeline for the final project report.
Run: python code/analysis.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cardio_train.csv"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "output"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

CONFOUNDERS = ["age_years", "gender", "smoke", "alco", "cholesterol", "gluc", "bmi"]
MEDIATOR = "bmi"
RNG = np.random.default_rng(42)


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    df["age_years"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    mask = (
        df["ap_hi"].between(80, 200)
        & df["ap_lo"].between(40, 120)
        & (df["ap_hi"] > df["ap_lo"])
        & df["bmi"].between(15, 60)
        & df["height"].between(140, 220)
    )
    df = df.loc[mask].copy()
    df["A"] = df["active"].astype(int)
    df["Y"] = df["cardio"].astype(int)
    df["M"] = df[MEDIATOR]
    df["age_group"] = pd.qcut(df["age_years"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return df


def fit_ps(df: pd.DataFrame, confounders: list[str]) -> np.ndarray:
    ps = sm.Logit(df["A"], sm.add_constant(df[confounders])).fit(disp=0, maxiter=200).predict(
        sm.add_constant(df[confounders])
    )
    return np.clip(ps, 0.01, 0.99)


def smd(x_t: np.ndarray, x_c: np.ndarray) -> float:
    s = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2)
    return 0.0 if s == 0 else float((x_t.mean() - x_c.mean()) / s)


def bootstrap_se(stat_fn, df: pd.DataFrame, n_boot: int = 500) -> float:
    n = len(df)
    vals = []
    for _ in range(n_boot):
        vals.append(stat_fn(df.iloc[RNG.integers(0, n, n)]))
    return float(np.std(vals, ddof=1))


def ipw_rd(df: pd.DataFrame, confounders: list[str]) -> float:
    ps = fit_ps(df, confounders)
    a, y = df["A"].values, df["Y"].values
    pa = a.mean()
    w = np.where(a == 1, pa / ps, (1 - pa) / (1 - ps))
    mu1 = np.sum(w * a * y) / np.sum(w * a)
    mu0 = np.sum(w * (1 - a) * y) / np.sum(w * (1 - a))
    return float(mu1 - mu0)


def ipw_ace(df: pd.DataFrame, confounders: list[str], n_boot: int = 500) -> dict:
    rd = ipw_rd(df, confounders)
    se = bootstrap_se(lambda d: ipw_rd(d, confounders), df, n_boot)
    return _rd_result("IPW", rd, se)


def lpm_ace(df: pd.DataFrame, confounders: list[str]) -> dict:
    x = sm.add_constant(df[["A"] + confounders])
    m = sm.OLS(df["Y"], x).fit(cov_type="HC1")
    rd, se = float(m.params["A"]), float(m.bse["A"])
    return _rd_result("LPM", rd, se)


def dr_ace(df: pd.DataFrame, confounders: list[str], n_boot: int = 400) -> dict:
    ps = fit_ps(df, confounders)
    a, y = df["A"].values.astype(float), df["Y"].values.astype(float)
    x = sm.add_constant(df[["A"] + confounders])
    out = sm.OLS(df["Y"], x).fit()
    x1, x0 = x.copy(), x.copy()
    x1["A"], x0["A"] = 1, 0
    mu1, mu0 = out.predict(x1), out.predict(x0)
    psi1 = mu1 + a * (y - mu1) / ps
    psi0 = mu0 + (1 - a) * (y - mu0) / (1 - ps)
    rd = float(psi1.mean() - psi0.mean())

    def dr_stat(d: pd.DataFrame) -> float:
        ps_b = fit_ps(d, confounders)
        ab = d["A"].values.astype(float)
        yb = d["Y"].values.astype(float)
        xb = sm.add_constant(d[["A"] + confounders])
        ob = sm.OLS(d["Y"], xb).fit()
        xb1, xb0 = xb.copy(), xb.copy()
        xb1["A"], xb0["A"] = 1, 0
        m1, m0 = ob.predict(xb1), ob.predict(xb0)
        return float((m1 + ab * (yb - m1) / ps_b).mean() - (m0 + (1 - ab) * (yb - m0) / (1 - ps_b)).mean())

    se = bootstrap_se(dr_stat, df, n_boot)
    return _rd_result("doubly_robust", rd, se)


def gformula_ace(df: pd.DataFrame, confounders: list[str], n_boot: int = 300) -> dict:
    cols = ["A"] + confounders
    x = sm.add_constant(df[cols])
    mod = sm.Logit(df["Y"], x).fit(disp=0, maxiter=200)
    x1, x0 = x.copy(), x.copy()
    x1["A"], x0["A"] = 1, 0
    rd = float(mod.predict(x1).mean() - mod.predict(x0).mean())

    def gf(d: pd.DataFrame) -> float:
        xb = sm.add_constant(d[cols])
        mb = sm.Logit(d["Y"], xb).fit(disp=0, maxiter=200)
        xb1, xb0 = xb.copy(), xb.copy()
        xb1["A"], xb0["A"] = 1, 0
        return float(mb.predict(xb1).mean() - mb.predict(xb0).mean())

    se = bootstrap_se(gf, df, n_boot)
    return _rd_result("g-formula", rd, se)


def ps_match_ace(df: pd.DataFrame, ps: np.ndarray) -> dict:
    d = df.copy()
    d["ps"] = ps
    tr = d[d["A"] == 1].reset_index(drop=True)
    ct = d[d["A"] == 0].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=1).fit(ct[["ps"]])
    dist, idx = nn.kneighbors(tr[["ps"]])
    keep = dist.ravel() <= 0.2 * d["ps"].std()
    ty = tr.loc[keep, "Y"].values
    cy = ct.loc[idx.ravel()[keep], "Y"].values
    rd = float(ty.mean() - cy.mean())
    se = float(np.sqrt(ty.var(ddof=1) / len(ty) + cy.var(ddof=1) / len(cy)))
    return _rd_result("PS_matching", rd, se)


def ps_strat_ace(df: pd.DataFrame, ps: np.ndarray, n_strata: int = 5) -> dict:
    d = df.copy()
    d["ps"] = ps
    d["stratum"] = pd.qcut(d["ps"], n_strata, labels=False, duplicates="drop")
    rds, ns = [], []
    for s in sorted(d["stratum"].dropna().unique()):
        ds = d[d["stratum"] == s]
        if ds["A"].nunique() < 2:
            continue
        rds.append(ds.loc[ds["A"] == 1, "Y"].mean() - ds.loc[ds["A"] == 0, "Y"].mean())
        ns.append(len(ds))
    rds, ns = np.array(rds), np.array(ns)
    rd = float(np.average(rds, weights=ns))
    se = float(np.sqrt(np.sum(ns**2 * (rds - rd) ** 2) / np.sum(ns) ** 2))
    return _rd_result("PS_stratification", rd, se)


def _rd_result(method: str, rd: float, se: float) -> dict:
    return {
        "method": method,
        "ace": rd,
        "se": se,
        "ci_low": rd - 1.96 * se,
        "ci_high": rd + 1.96 * se,
    }


def crude_ace(df: pd.DataFrame) -> dict:
    p1 = df.loc[df["A"] == 1, "Y"].mean()
    p0 = df.loc[df["A"] == 0, "Y"].mean()
    rd = float(p1 - p0)
    se = float(np.sqrt(p1 * (1 - p1) / (df["A"] == 1).sum() + p0 * (1 - p0) / (df["A"] == 0).sum()))
    return _rd_result("crude", rd, se)


def balance_table(df: pd.DataFrame, confounders: list[str], sw: np.ndarray) -> pd.DataFrame:
    rows = []
    for v in confounders:
        xt = df.loc[df["A"] == 1, v].values
        xc = df.loc[df["A"] == 0, v].values
        wt, wc = sw[df["A"] == 1], sw[df["A"] == 0]
        m1, m0 = np.average(xt, weights=wt), np.average(xc, weights=wc)
        v1 = np.average((xt - m1) ** 2, weights=wt)
        v0 = np.average((xc - m0) ** 2, weights=wc)
        s = np.sqrt((v1 + v0) / 2)
        rows.append(
            {
                "variable": v,
                "smd_unweighted": smd(xt, xc),
                "smd_weighted": 0.0 if s == 0 else float((m1 - m0) / s),
            }
        )
    return pd.DataFrame(rows)


def mediation_bmi(df: pd.DataFrame, n_boot: int = 500) -> dict:
    """Regression-based mediation A -> BMI -> Y (VanderWeele 2015)."""
    C = [c for c in CONFOUNDERS if c != MEDIATOR]
    m_mod = sm.OLS(df["M"], sm.add_constant(pd.concat([df[["A"]], df[C]], axis=1))).fit(cov_type="HC1")
    y_mod = sm.OLS(df["Y"], sm.add_constant(pd.concat([df[["A", "M"]], df[C]], axis=1))).fit(cov_type="HC1")
    nde = float(y_mod.params["A"])
    nie = float(m_mod.params["A"] * y_mod.params["M"])
    total = nde + nie
    prop = float(nie / total) if abs(total) > 1e-8 else np.nan

    b_nde, b_nie, b_tot = [], [], []
    n = len(df)
    for _ in range(n_boot):
        b = df.iloc[RNG.integers(0, n, n)]
        mm = sm.OLS(b["M"], sm.add_constant(pd.concat([b[["A"]], b[C]], axis=1))).fit()
        ym = sm.OLS(b["Y"], sm.add_constant(pd.concat([b[["A", "M"]], b[C]], axis=1))).fit()
        b_nie.append(mm.params["A"] * ym.params["M"])
        b_nde.append(ym.params["A"])
        b_tot.append(ym.params["A"] + mm.params["A"] * ym.params["M"])

    return {
        "nde": nde,
        "nie": nie,
        "total": float(total),
        "proportion_mediated": prop,
        "nde_ci": [float(np.percentile(b_nde, 2.5)), float(np.percentile(b_nde, 97.5))],
        "nie_ci": [float(np.percentile(b_nie, 2.5)), float(np.percentile(b_nie, 97.5))],
        "total_ci": [float(np.percentile(b_tot, 2.5)), float(np.percentile(b_tot, 97.5))],
        "a_path": float(m_mod.params["A"]),
        "b_path": float(y_mod.params["M"]),
        "resid_m_sd": float(m_mod.resid.std()),
        "resid_y_sd": float(y_mod.resid.std()),
    }


def mediation_bp(df: pd.DataFrame) -> dict:
    m_mod = sm.OLS(df["ap_hi"], sm.add_constant(pd.concat([df[["A"]], df[CONFOUNDERS]], axis=1))).fit(cov_type="HC1")
    y_mod = sm.OLS(df["Y"], sm.add_constant(pd.concat([df[["A", "ap_hi"]], df[CONFOUNDERS]], axis=1))).fit(
        cov_type="HC1"
    )
    nie = float(m_mod.params["A"] * y_mod.params["ap_hi"])
    nde = float(y_mod.params["A"])
    total = nde + nie
    return {
        "nde": nde,
        "nie": nie,
        "total": float(total),
        "proportion_mediated": float(nie / total) if abs(total) > 1e-8 else np.nan,
    }


def mediation_sensitivity_rho(med: dict, rho_grid: np.ndarray | None = None) -> pd.DataFrame:
    """
    Sensitivity of NIE to correlation rho between mediator/outcome model errors
    (VanderWeele 2015, Ch.4; approximate bias correction for product ab).
    """
    if rho_grid is None:
        rho_grid = np.linspace(-0.9, 0.9, 37)
    sd_m, sd_y = med["resid_m_sd"], med["resid_y_sd"]
    nie0 = med["nie"]
    rows = []
    for rho in rho_grid:
        bias = rho * sd_m * sd_y
        nie_rho = nie0 + bias
        nde_rho = med["total"] - nie_rho
        rows.append({"rho": float(rho), "nie": float(nie_rho), "nde": float(nde_rho)})
    return pd.DataFrame(rows)


def effect_modification(df: pd.DataFrame) -> dict:
    C = ["age_years", "gender", "alco", "cholesterol", "gluc", "bmi"]
    d = df.copy()
    d["AxS"] = d["A"] * d["smoke"]
    m = sm.OLS(d["Y"], sm.add_constant(d[["A", "smoke", "AxS"] + C])).fit(cov_type="HC1")
    ace_s = float(m.params["A"] + m.params["AxS"])
    se_s = float(np.sqrt(m.bse["A"] ** 2 + m.bse["AxS"] ** 2))
    return {
        "interaction_coef": float(m.params["AxS"]),
        "interaction_p": float(m.pvalues["AxS"]),
        "ace_nonsmoker": float(m.params["A"]),
        "ace_smoker": ace_s,
        "ace_nonsmoker_ci": [float(m.params["A"] - 1.96 * m.bse["A"]), float(m.params["A"] + 1.96 * m.bse["A"])],
        "ace_smoker_ci": [ace_s - 1.96 * se_s, ace_s + 1.96 * se_s],
    }


def adjustment_sensitivity(df: pd.DataFrame) -> list[dict]:
    specs = [
        ("main (L incl. BMI)", CONFOUNDERS),
        ("exclude BMI", [c for c in CONFOUNDERS if c != "bmi"]),
        ("add blood pressure", CONFOUNDERS + ["ap_hi", "ap_lo"]),
    ]
    out = []
    for label, conf in specs:
        est = ipw_ace(df, conf, n_boot=400)
        est["specification"] = label
        out.append(est)
    return out


def ps_diagnostics(df: pd.DataFrame, ps: np.ndarray) -> dict:
    """Summary statistics for positivity / overlap assessment."""
    ps0 = ps[df["A"] == 0]
    ps1 = ps[df["A"] == 1]
    overlap_lo = max(ps0.min(), ps1.min())
    overlap_hi = min(ps0.max(), ps1.max())
    in_overlap = ((ps >= overlap_lo) & (ps <= overlap_hi)).mean()
    # common support: controls exist for treated PS values
    treated_supported = np.mean([np.any(np.abs(ps0 - p) < 0.05) for p in ps1])
    return {
        "ps_mean_inactive": float(ps0.mean()),
        "ps_mean_active": float(ps1.mean()),
        "ps_min_inactive": float(ps0.min()),
        "ps_max_inactive": float(ps0.max()),
        "ps_min_active": float(ps1.min()),
        "ps_max_active": float(ps1.max()),
        "overlap_range": [float(overlap_lo), float(overlap_hi)],
        "frac_in_overlap": float(in_overlap),
        "crude_rate_inactive": float(df.loc[df["A"] == 0, "Y"].mean()),
        "crude_rate_active": float(df.loc[df["A"] == 1, "Y"].mean()),
    }


def e_value(rd: float, se: float, p0: float) -> dict:
    p1 = np.clip(p0 + rd, 1e-6, 1 - 1e-6)
    rr = p1 / p0 if rd < 0 else p0 / p1
    rd_ci = rd - 1.96 * se
    p1_ci = np.clip(p0 + rd_ci, 1e-6, 1 - 1e-6)
    rr_ci = p1_ci / p0 if rd_ci < 0 else p0 / p1_ci
    ev = rr + np.sqrt(rr * (rr - 1)) if rr >= 1 else (1 / rr) + np.sqrt((1 / rr) * (1 / rr - 1))
    ev_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1)) if rr_ci >= 1 else (1 / rr_ci) + np.sqrt((1 / rr_ci) * (1 / rr_ci - 1))
    return {"e_value": float(ev), "e_value_ci": float(ev_ci), "p0": float(p0), "approx_rr": float(rr)}


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def write_report_metrics(summary: dict) -> None:
    """Export LaTeX-friendly numeric strings used in the report."""
    est = {r["method"]: r for r in summary["estimates"]}
    med = summary["mediation_bmi"]
    ev = summary["e_value"]
    em = summary["effect_modification"]
    lines = {
        "n": str(summary["n"]),
        "prev_y": fmt(summary["prevalence_y"], 3),
        "prev_a": fmt(summary["prevalence_a"], 3),
        "crude_ace": fmt(est["crude"]["ace"]),
        "crude_ci": f"[{fmt(est['crude']['ci_low'])}, {fmt(est['crude']['ci_high'])}]",
        "ipw_ace": fmt(est["IPW"]["ace"]),
        "ipw_ci": f"[{fmt(est['IPW']['ci_low'])}, {fmt(est['IPW']['ci_high'])}]",
        "dr_ace": fmt(est["doubly_robust"]["ace"]),
        "dr_ci": f"[{fmt(est['doubly_robust']['ci_low'])}, {fmt(est['doubly_robust']['ci_high'])}]",
        "gf_ace": fmt(est["g-formula"]["ace"]),
        "gf_ci": f"[{fmt(est['g-formula']['ci_low'])}, {fmt(est['g-formula']['ci_high'])}]",
        "match_ace": fmt(est["PS_matching"]["ace"]),
        "match_ci": f"[{fmt(est['PS_matching']['ci_low'])}, {fmt(est['PS_matching']['ci_high'])}]",
        "strat_ace": fmt(est["PS_stratification"]["ace"]),
        "strat_ci": f"[{fmt(est['PS_stratification']['ci_low'])}, {fmt(est['PS_stratification']['ci_high'])}]",
        "lpm_ace": fmt(est["LPM"]["ace"]),
        "lpm_ci": f"[{fmt(est['LPM']['ci_low'])}, {fmt(est['LPM']['ci_high'])}]",
        "nde": fmt(med["nde"]),
        "nie": fmt(med["nie"]),
        "prop_med": f"{100 * med['proportion_mediated']:.1f}",
        "nde_ci": f"[{fmt(med['nde_ci'][0])}, {fmt(med['nde_ci'][1])}]",
        "nie_ci": f"[{fmt(med['nie_ci'][0])}, {fmt(med['nie_ci'][1])}]",
        "a_path": fmt(med["a_path"], 3),
        "b_path": fmt(med["b_path"], 3),
        "bp_nie": fmt(summary["mediation_bp"]["nie"], 4),
        "evalue": fmt(ev["e_value"], 2),
        "evalue_ci": fmt(ev["e_value_ci"], 2),
        "int_p": fmt(em["interaction_p"], 3),
        "ace_nonsmoke": fmt(em["ace_nonsmoker"]),
        "ace_smoke": fmt(em["ace_smoker"]),
        "rho_tip_nie0": fmt(summary.get("rho_tipping_nie_zero", np.nan), 4),
        "ps_mean_inactive": fmt(summary["ps_diagnostics"]["ps_mean_inactive"], 3),
        "ps_mean_active": fmt(summary["ps_diagnostics"]["ps_mean_active"], 3),
        "rate_inactive": fmt(summary["ps_diagnostics"]["crude_rate_inactive"], 3),
        "rate_active": fmt(summary["ps_diagnostics"]["crude_rate_active"], 3),
    }
    with open(OUT_DIR / "report_metrics.json", "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=2, ensure_ascii=False)


def make_figures(df, ps, bal, med, em, het, sens_rho, adj_sens):
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    ps0, ps1 = ps[df["A"] == 0], ps[df["A"] == 1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(ps0, bins=50, alpha=0.55, label=f"Inactive (n={(df['A']==0).sum():,})", density=True, color="#c44e52")
    ax.hist(ps1, bins=50, alpha=0.55, label=f"Active (n={(df['A']==1).sum():,})", density=True, color="#4c72b0")
    olo, ohi = max(ps0.min(), ps1.min()), min(ps0.max(), ps1.max())
    ax.axvspan(olo, ohi, alpha=0.12, color="green", label="Common support")
    ax.axvline(ps0.mean(), color="#c44e52", ls="--", lw=1.2)
    ax.axvline(ps1.mean(), color="#4c72b0", ls="--", lw=1.2)
    ax.set_xlabel("Propensity score  $\\hat e(L)=\\widehat{\\P}(A=1\\mid L)$")
    ax.set_ylabel("Density")
    ax.set_title("Positivity check: propensity score overlap by treatment group")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ps_overlap.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(bal))
    ax.scatter(bal["smd_unweighted"], y, marker="o", label="Before IPW")
    ax.scatter(bal["smd_weighted"], y, marker="s", label="After IPW")
    ax.axvline(0, color="gray", lw=0.8)
    ax.axvline(-0.1, color="red", ls="--", lw=0.8)
    ax.axvline(0.1, color="red", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(bal["variable"])
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Covariate balance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "love_plot.pdf")
    plt.close()

    est = pd.read_csv(OUT_DIR / "estimates.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ypos = np.arange(len(est))
    ax.errorbar(est["ace"], ypos, xerr=1.96 * est["se"], fmt="o", capsize=4, color="#4c72b0")
    ax.axvline(0, color="gray", ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(est["method"])
    ax.set_xlabel("ACE (risk difference)")
    ax.set_title("Triangulation across causal estimators")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forest_plot.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["NDE", "NIE"], [med["nde"], med["nie"]], color=["#4c72b0", "#55a868"], width=0.55)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Risk difference")
    ax.set_title("Mediation decomposition via BMI")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mediation_decomp.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sens_rho["rho"], sens_rho["nie"], lw=2, label="NIE")
    ax.plot(sens_rho["rho"], sens_rho["nde"], lw=2, ls="--", label="NDE")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel(r"Sensitivity parameter $\rho$")
    ax.set_ylabel("Effect (risk difference)")
    ax.set_title("Mediation sensitivity to correlated errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mediation_sensitivity.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = [r["specification"] for r in adj_sens]
    vals = [r["ace"] for r in adj_sens]
    ses = [r["se"] for r in adj_sens]
    ypos = np.arange(len(labels))
    ax.errorbar(vals, ypos, xerr=1.96 * np.array(ses), fmt="o", capsize=4)
    ax.axvline(0, color="gray", ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("IPW ACE")
    ax.set_title("Adjustment-set sensitivity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "adjustment_sensitivity.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Non-smoker", "Smoker"]
    vals = [em["ace_nonsmoker"], em["ace_smoker"]]
    cis = [em["ace_nonsmoker_ci"], em["ace_smoker_ci"]]
    ax.errorbar(
        vals,
        labels,
        xerr=[[vals[i] - cis[i][0] for i in range(2)], [cis[i][1] - vals[i] for i in range(2)]],
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, color="gray", ls="--")
    ax.set_xlabel("Activity effect on CVD (LPM)")
    ax.set_title("Effect modification by smoking")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "effect_modification.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(
        het["ace"],
        het["subgroup"],
        xerr=[het["ace"] - het["ci_low"], het["ci_high"] - het["ace"]],
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, color="gray", ls="--")
    ax.set_xlabel("IPW ACE")
    ax.set_title("Age-stratified effects")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "age_heterogeneity.pdf")
    plt.close()


def main() -> None:
    df = load_and_clean()
    ps = fit_ps(df, CONFOUNDERS)
    pa = float(df["A"].mean())
    sw = np.where(df["A"] == 1, pa / ps, (1 - pa) / (1 - ps))

    results = [
        crude_ace(df),
        ipw_ace(df, CONFOUNDERS),
        dr_ace(df, CONFOUNDERS),
        gformula_ace(df, CONFOUNDERS),
        lpm_ace(df, CONFOUNDERS),
        ps_match_ace(df, ps),
        ps_strat_ace(df, ps),
    ]

    ps_diag = ps_diagnostics(df, ps)
    med_bmi = mediation_bmi(df)
    med_bp = mediation_bp(df)
    sens_rho = mediation_sensitivity_rho(med_bmi)
    denom = med_bmi["resid_m_sd"] * med_bmi["resid_y_sd"]
    rho_tip = float(-med_bmi["nie"] / denom) if denom > 0 else np.nan

    em = effect_modification(df)
    adj_sens = adjustment_sensitivity(df)
    p0 = float(df.loc[df["A"] == 0, "Y"].mean())
    ev = e_value(results[1]["ace"], results[1]["se"], p0)

    het_rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df[df["age_group"] == q]
        est = ipw_ace(sub, CONFOUNDERS, n_boot=300)
        est["subgroup"] = q
        het_rows.append(est)
    het = pd.DataFrame(het_rows)

    bal = balance_table(df, CONFOUNDERS, sw)

    pd.DataFrame(results).to_csv(OUT_DIR / "estimates.csv", index=False)
    bal.to_csv(OUT_DIR / "balance.csv", index=False)
    het.to_csv(OUT_DIR / "heterogeneity.csv", index=False)
    sens_rho.to_csv(OUT_DIR / "mediation_sensitivity.csv", index=False)
    pd.DataFrame(adj_sens).to_csv(OUT_DIR / "adjustment_sensitivity.csv", index=False)

    summary = {
        "n": len(df),
        "prevalence_y": float(df["Y"].mean()),
        "prevalence_a": pa,
        "estimates": results,
        "mediation_bmi": med_bmi,
        "mediation_bp": med_bp,
        "effect_modification": em,
        "e_value": ev,
        "heterogeneity": het.to_dict(orient="records"),
        "adjustment_sensitivity": adj_sens,
        "rho_tipping_nie_zero": float(rho_tip),
        "ps_diagnostics": ps_diag,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_report_metrics(summary)
    make_figures(df, ps, bal, med_bmi, em, het, sens_rho, adj_sens)

    print(f"N={len(df)}")
    for r in results:
        print(f"{r['method']:18s} ACE={r['ace']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    print(f"NDE={med_bmi['nde']:+.4f} {med_bmi['nde_ci']}, NIE={med_bmi['nie']:+.4f} {med_bmi['nie_ci']}")
    print(f"E-value={ev['e_value']:.2f}, rho_tip(NIE=0)={rho_tip:.4f}")


if __name__ == "__main__":
    main()
