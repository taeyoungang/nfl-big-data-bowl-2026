"""
Expected yards after catch (YAC), completion probability, pass rush and receiver opportunity.

Reads processed/receiver_features.csv (python yac_features.py) and writes
    processed/yac_scores.csv   out-of-fold scores for every eligible receiver on every thrown pass
    figures/yac_*.png
    yac_report.md              every number quoted in the write-up

Every score is out-of-fold: 5-fold cross-validation grouped by game, so no game is ever in train and test together.

Two versions of each model:
    analysis  (suffix _a)  also sees the first 0.5 s of ball flight (t5 and flight-based projection). Most accurate
                           for the receiver who was actually thrown to; used for residuals and the coverage analysis.
    decision  (suffix _d)  sees only what was knowable at release. Defined identically for every eligible receiver,
                           so it is the one used to score the receivers who were NOT thrown to.

The coverage scheme is deliberately NOT a model feature, so a scheme effect shows up in the residuals instead of
being absorbed by the model.

Usage:
    python yac_analysis.py
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, r2_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, "processed")
FIG = os.path.join(ROOT, "figures")
KEYS = ["gameId", "playId"]
CAP = 30
QS = [0.10, 0.25, 0.50, 0.75, 0.90]
QN = [f"q{int(q * 100)}" for q in QS]
CATS = ["pos_lined_up", "pos_group", "offenseFormation", "pff_passCoverage", "pff_passCoverageType"]
BLUE, RED, GREY, GREEN, ORANGE = "#4B8BBE", "#d9534f", "#888888", "#2ca02c", "#ff7f0e"
SEED = 0
REPORT = []


def say(text=""):
    print(text)
    REPORT.append(text)


def table(df, **kw):
    say("```")
    say(df.to_string(**kw))
    say("```")


# ── feature families, split by when the information exists ───────────────────

def cols(names, tags):
    return [f"{n}_{t}" for t in tags for n in names]


SEP = ["sep1", "sep2", "n_def5", "close_speed"]
OPEN = ["n_def10", "t_close", "cone_dist", "n_ahead10"]
LEV = ["lev_dx", "lev_inside", "lev_cos"]
BLK = ["n_block10", "block_net"]
POS = ["x_rel", "sideline", "goal_dist"]
KIN = ["spd", "acc", "vx", "vy", "face_x"]

REL = {  # known at release (the pass_forward frame)
    "Separation": cols(SEP, ["t0"]),
    "Open field": cols(OPEN, ["t0"]),
    "Leverage": cols(LEV, ["t0"]),
    "Blockers ahead": cols(BLK, ["t0"]),
    "Receiver motion and location": cols(KIN, ["t0"]) + cols(POS, ["t0"]),
    "Decision projection": cols(SEP + OPEN + LEV + BLK + POS, ["d"]) + ["tau_d", "pass_dist_d"],
    "Throw": ["throw_dist", "lateral_from_qb"],
    "Pass rush and timing": ["time_to_throw", "qb_depth", "qb_speed", "qb_nearest_def", "n_rush", "rusher_dist", "n_rush5", "rusher_close"],
    "Receiver role and body": ["pos_lined_up", "pos_group", "weight", "height_in"],
    "Game context": ["down", "yardsToGo", "offenseFormation", "pff_playAction", "defendersInBox"],
}
POST = {  # needs the first 0.5 s of ball flight, so only meaningful for the receiver the ball was thrown to
    "Separation": cols(SEP, ["t5"]) + ["sep1_delta"],
    "Open field": cols(OPEN, ["t5"]),
    "Leverage": cols(LEV, ["t5"]),
    "Blockers ahead": cols(BLK, ["t5"]),
    "Receiver motion and location": cols(KIN, ["t5"]) + cols(POS, ["t5"]),
    "Flight-based projection": ["tau", "rem_dist"] + cols(SEP + OPEN + LEV + BLK + POS, ["p"]),
    "Throw": ["v_ball"],
}
ALL_FAMS = list(REL) + ["Flight-based projection"]
GEO_REL = ["Separation", "Open field", "Leverage", "Blockers ahead", "Receiver motion and location", "Decision projection"]


def fam_cols(fams, post):
    out = []
    for n in fams:
        out += REL.get(n, [])
        if post:
            out += POST.get(n, [])
    return out


BASE = ["Game context", "Receiver role and body"]
YAC_LADDER = [
    ("1. Context + receiver role", BASE, True),
    ("2. + Separation", BASE + ["Separation"], True),
    ("3. + Open field, leverage, blockers", BASE + ["Separation", "Open field", "Leverage", "Blockers ahead"], True),
    ("4. + Receiver motion and location", BASE + ["Separation", "Open field", "Leverage", "Blockers ahead", "Receiver motion and location"], True),
    ("5. + Projected arrival and throw", BASE + GEO_REL + ["Flight-based projection", "Throw"], True),
    ("6. + Pass rush and timing  [analysis model]", ALL_FAMS, True),
    ("7. Release-time only  [decision model]", ALL_FAMS, False),
]
COMP_LADDER = [
    ("C1. Context, role, throw distance", BASE + ["Throw"], False),
    ("C2. + receiver geometry at release", BASE + ["Throw"] + GEO_REL, False),
    ("C3. + pass rush and timing  [decision model]", ALL_FAMS, False),
    ("C4. + first 0.5 s of ball flight  [analysis model]", ALL_FAMS, True),
]
PRESSURE_ROW = "C5. + PFF pressure flag (charged after the play)"


# ── models and metrics ───────────────────────────────────────────────────────

def reg(**kw):
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30,
                                         l2_regularization=1.0, categorical_features="from_dtype", random_state=SEED, **kw)


def clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                          l2_regularization=1.0, categorical_features="from_dtype", random_state=SEED)


def rho(a, b):
    return np.nan if np.ptp(b) == 0 else spearmanr(a, b)[0]


def pinball(y, p, q):
    d = y - p
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def calib(p, y, bins=10):
    d = pd.DataFrame({"p": np.asarray(p), "y": np.asarray(y)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))


def permute_family(X, cc, rng):
    Xp = X.copy()
    perm = rng.permutation(len(X))
    for c in cc:
        Xp[c] = X[c].iloc[perm].values  # .values keeps categorical dtype
    return Xp


# ── game-cluster bootstrap and within-game permutation ───────────────────────

def boot_group_means(df, label, value, n_boot=1000):
    labs = sorted(df[label].dropna().unique())
    li = {l: i for i, l in enumerate(labs)}
    games = df["gameId"].unique()
    gi = {g: i for i, g in enumerate(games)}
    g_idx, l_idx, v = df["gameId"].map(gi).to_numpy(), df[label].map(li).to_numpy(), df[value].to_numpy(float)
    S, N = np.zeros((len(games), len(labs))), np.zeros((len(games), len(labs)))
    np.add.at(S, (g_idx, l_idx), v)
    np.add.at(N, (g_idx, l_idx), 1.0)
    W = np.random.default_rng(SEED).multinomial(len(games), np.ones(len(games)) / len(games), size=n_boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        return labs, S.sum(0) / N.sum(0), (W @ S) / (W @ N), N.sum(0)


def ci_table(df, label, value, min_n=1, n_boot=1000):
    labs, obs, draws, n = boot_group_means(df, label, value, n_boot)
    out = pd.DataFrame({"n": n.astype(int), "mean": obs, "lo": np.nanpercentile(draws, 2.5, axis=0),
                        "hi": np.nanpercentile(draws, 97.5, axis=0)}, index=labs)
    return out[out["n"] >= min_n]


def scheme_test(codes, resid, games, n_perm=2000):
    """z-score of each scheme's mean residual, with family-wise p-values from shuffling scheme labels WITHIN each game
    (so game-level effects such as opponent, weather or script cannot create a false difference)."""
    K = codes.max() + 1
    sd = resid.std(ddof=1)

    def z(c):
        cnt = np.bincount(c, minlength=K)
        return (np.bincount(c, resid, minlength=K) / cnt) / (sd / np.sqrt(cnt))
    obs = z(codes)
    groups = [np.where(games == g)[0] for g in np.unique(games)]
    rng = np.random.default_rng(SEED)
    null = np.empty(n_perm)
    for i in range(n_perm):
        pc = codes.copy()
        for ix in groups:
            pc[ix] = codes[ix][rng.permutation(len(ix))]
        null[i] = np.abs(z(pc)).max()
    return obs, null


# ── data ─────────────────────────────────────────────────────────────────────

def load():
    f = pd.read_csv(os.path.join(PROC, "receiver_features.csv"))
    for c in CATS:
        f[c] = f[c].astype("category")
    f["pff_playAction"] = f["pff_playAction"].astype(float)
    plays = pd.read_csv(os.path.join(ROOT, "data", "plays.csv"), usecols=KEYS + ["prePenaltyPlayResult"])
    return f.merge(plays, on=KEYS, how="left")


def fit_all(f):
    folds = list(GroupKFold(5).split(f, groups=f["gameId"]))
    is_t = f["is_target"].to_numpy()
    comp_rows = is_t & (f["complete"] == 1).to_numpy() & f["yac"].notna().to_numpy()
    y_raw = f["yac"].to_numpy(float)
    y_cap = np.minimum(y_raw, CAP)
    n = len(f)
    nanv = lambda: np.full(n, np.nan)  # noqa: E731
    sc = {f"{k}_{s}": nanv() for s in "ad" for k in ["yac_mean", "p_comp", "p_ge10"] + QN}
    sc["base_mean"] = nanv()
    for q in QN:
        sc["base_" + q] = nanv()
    yl = {name: nanv() for name, _, _ in YAC_LADDER}
    cl = {name: nanv() for name, _, _ in COMP_LADDER}
    cl[PRESSURE_ROW] = nanv()
    imp_y = {k: [] for k in ALL_FAMS}
    imp_c = {k: [] for k in ALL_FAMS}
    rng = np.random.default_rng(SEED)

    for k, (tr, te) in enumerate(folds, 1):
        trm, tem = np.zeros(n, bool), np.zeros(n, bool)
        trm[tr], tem[te] = True, True
        tr_y, te_y, tr_c, te_c = trm & comp_rows, tem & comp_rows, trm & is_t, tem & is_t
        ycomp = f["complete"].fillna(0).astype(int).to_numpy()

        for name, fams, post in YAC_LADDER:
            c = fam_cols(fams, post)
            yl[name][te_y] = reg().fit(f.loc[tr_y, c], y_cap[tr_y]).predict(f.loc[te_y, c])
        sc["base_mean"][te_y] = y_cap[tr_y].mean()
        for q, qn in zip(QS, QN):
            sc["base_" + qn][te_y] = np.quantile(y_raw[tr_y], q)

        for name, fams, post in COMP_LADDER:
            c = fam_cols(fams, post)
            cl[name][te_c] = clf().fit(f.loc[tr_c, c], ycomp[tr_c]).predict_proba(f.loc[te_c, c])[:, 1]
        c = fam_cols(ALL_FAMS, True) + ["pressured"]
        cl[PRESSURE_ROW][te_c] = clf().fit(f.loc[tr_c, c], ycomp[tr_c]).predict_proba(f.loc[te_c, c])[:, 1]

        models = {}
        for s, post in (("a", True), ("d", False)):
            c = fam_cols(ALL_FAMS, post)
            models[s] = (reg().fit(f.loc[tr_y, c], y_cap[tr_y]), clf().fit(f.loc[tr_c, c], ycomp[tr_c]), c)
            sc[f"yac_mean_{s}"][tem] = models[s][0].predict(f.loc[tem, c])
            sc[f"p_comp_{s}"][tem] = models[s][1].predict_proba(f.loc[tem, c])[:, 1]
            for q, qn in zip(QS, QN):
                sc[f"{qn}_{s}"][tem] = reg(loss="quantile", quantile=q).fit(f.loc[tr_y, c], y_raw[tr_y]).predict(f.loc[tem, c])
            sc[f"p_ge10_{s}"][tem] = clf().fit(f.loc[tr_y, c], (y_raw[tr_y] >= 10).astype(int)).predict_proba(f.loc[tem, c])[:, 1]

        # family importance: scramble a whole family on held-out targets, measure the added loss
        m_y, _, c_y = models["a"]
        _, m_c, c_c = models["d"]
        Xy, Xc = f.loc[te_y, c_y], f.loc[te_c, c_c]
        yc_t = ycomp[te_c]
        base_y = mean_absolute_error(y_cap[te_y], m_y.predict(Xy))
        base_c = log_loss(yc_t, m_c.predict_proba(Xc)[:, 1])
        for fam in ALL_FAMS:
            cy, cc = fam_cols([fam], True), fam_cols([fam], False)
            dy = [mean_absolute_error(y_cap[te_y], m_y.predict(permute_family(Xy, cy, rng))) - base_y for _ in range(3)] if cy else [0.0]
            dc = [log_loss(yc_t, m_c.predict_proba(permute_family(Xc, cc, rng))[:, 1]) - base_c for _ in range(3)] if cc else [0.0]
            imp_y[fam].append(np.mean(dy))
            imp_c[fam].append(np.mean(dc))
        print(f"  fold {k}/5 done")
    return sc, yl, cl, imp_y, imp_c, comp_rows, y_raw, y_cap


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    f = load()
    n_plays = f[KEYS].drop_duplicates().shape[0]
    say("# Expected YAC, completion probability, pass rush and receiver opportunity")
    say("")
    say(f"Receiver rows: {len(f):,} on {n_plays:,} thrown passes in {f['gameId'].nunique()} games "
        f"(targets: {int(f['is_target'].sum()):,}; completions with a YAC label: "
        f"{int((f['is_target'] & (f['complete'] == 1) & f['yac'].notna()).sum()):,}). All scores are out-of-fold: 5 folds grouped by game.")
    say("Two model versions. **Analysis** also sees the first 0.5 s of ball flight. **Decision** sees only what was knowable at release "
        "and is defined identically for every eligible receiver, so it is the one used to score receivers who were not thrown to.")
    say("")
    print("Fitting models ...")
    sc, yl, cl, imp_y, imp_c, comp_rows, y_raw, y_cap = fit_all(f)
    is_t = f["is_target"].to_numpy()
    for k, v in sc.items():
        f[k] = v
    # keep each model's quantiles ordered
    for s in "ad":
        srt = np.sort(f[[f"{q}_{s}" for q in QN]].to_numpy(), axis=1)
        for i, q in enumerate(QN):
            f[f"{q}_{s}"] = srt[:, i]

    # ── 1 ──
    say("## 1. Expected YAC (point estimate)")
    say(f"Target: yards after catch, capped at {CAP} (0.7% of catches exceed it). Evaluated on {int(comp_rows.sum()):,} held-out catches.")
    rows = []

    def row(name, p, mask=comp_rows):
        yy = y_cap[mask]
        rows.append({"model": name, "n": int(mask.sum()), "MAE": mean_absolute_error(yy, p),
                     "RMSE": float(np.sqrt(np.mean((yy - p) ** 2))), "R2": r2_score(yy, p), "Spearman": rho(yy, p)})
    row("A. Average YAC", f.loc[comp_rows, "base_mean"].to_numpy())
    nv = comp_rows & f["nfl_xyac"].notna().to_numpy()
    row("B. nflfastR xYAC (public benchmark)", f.loc[nv, "nfl_xyac"].to_numpy(), nv)
    row("   A on the same rows as B", f.loc[nv, "base_mean"].to_numpy(), nv)
    for name, _, _ in YAC_LADDER:
        row(name, yl[name][comp_rows])
    res = pd.DataFrame(rows).round(3)
    table(res, index=False)
    say("Earlier probe, which used nflverse air_yards as an input: MAE 2.93, R2 0.29. These models use no nflverse input except the label.")
    say("")

    # ── 2 ──
    say("## 2. Expected YAC as a range")
    yr = y_raw[comp_rows]
    qtab = {}
    for s, label in (("a", "analysis"), ("d", "decision")):
        rws = []
        for q, qn in zip(QS, QN):
            p, b = f.loc[comp_rows, f"{qn}_{s}"].to_numpy(), f.loc[comp_rows, "base_" + qn].to_numpy()
            rws.append({"quantile": q, "share of catches below": float((yr <= p).mean()), "model pinball": pinball(yr, p, q),
                        "unconditional pinball": pinball(yr, b, q)})
        qt = pd.DataFrame(rws).set_index("quantile")
        qt["improvement"] = 1 - qt["model pinball"] / qt["unconditional pinball"]
        qtab[s] = qt
        say(f"**{label.capitalize()} model**")
        table(qt.round(3))
        lo, q25, q75, hi = (f.loc[comp_rows, f"{x}_{s}"].to_numpy() for x in ("q10", "q25", "q75", "q90"))
        say(f"10th-90th band holds {((yr >= lo) & (yr <= hi)).mean():.1%} of catches (nominal 80%); 25th-75th band holds {((yr >= q25) & (yr <= q75)).mean():.1%} (nominal 50%). "
            f"{(yr <= 0).mean():.1%} of catches gain 0 yd or less, so the low quantiles sit at about 0 and the bands over-cover; the upper quantiles are the informative ones "
            f"(share of catches below the 75th and 90th percentile predictions: {(yr <= f.loc[comp_rows, f'q75_{s}'].to_numpy()).mean():.1%} and {(yr <= f.loc[comp_rows, f'q90_{s}'].to_numpy()).mean():.1%}).")
        ge, pg = (yr >= 10).astype(int), f.loc[comp_rows, f"p_ge10_{s}"].to_numpy()
        top = np.argsort(-pg)[: len(pg) // 10]
        say(f"P(YAC >= 10): base rate {ge.mean():.1%} | AUC {roc_auc_score(ge, pg):.3f} | Brier {brier_score_loss(ge, pg):.4f} vs {brier_score_loss(ge, np.full(len(ge), ge.mean())):.4f} "
            f"for the base rate | top decile: {ge[top].mean():.1%} gain 10 or more.")
        say("")
    ge, pg = (yr >= 10).astype(int), f.loc[comp_rows, "p_ge10_a"].to_numpy()
    cal_ge = calib(pg, ge)
    say("Calibration of P(YAC >= 10), analysis model:")
    table(cal_ge.round(3))
    say("")

    # ── 3 ──
    say("## 3. Completion probability and pass rush")
    ya = f.loc[is_t, "complete"].astype(int).to_numpy()
    br = ya.mean()
    rws = [{"model": "Base rate", "AUC": 0.5, "log loss": log_loss(ya, np.full(len(ya), br)), "Brier": brier_score_loss(ya, np.full(len(ya), br))}]
    for name in cl:
        p = cl[name][is_t]
        rws.append({"model": name, "AUC": roc_auc_score(ya, p), "log loss": log_loss(ya, p), "Brier": brier_score_loss(ya, p)})
    ct = pd.DataFrame(rws).set_index("model").round(4)
    say(f"{len(ya):,} targeted attempts (complete, incomplete, interception); completion rate {br:.1%}.")
    table(ct)
    say("Pass-rush features are measured at release: distance from the quarterback to the nearest rusher, rushers within 5 yd, the nearest rusher's closing speed, "
        "number of rushers, quarterback depth and speed, and time from snap to throw. C2 to C3 isolates their contribution. "
        "The PFF flag is charged after the play and is shown only to size what it adds beyond the geometry.")
    cal_c = calib(f.loc[is_t, "p_comp_d"], ya)
    say("Calibration of the decision-model completion probability:")
    table(cal_c.round(3))

    t = f[is_t].copy()
    t["rush_bin"] = pd.cut(t["rusher_dist"], [0, 2, 4, 6, 8, 100], labels=["< 2 yd", "2-4", "4-6", "6-8", "8+"])
    t["ttt_bin"] = pd.cut(t["time_to_throw"], [0, 2.0, 2.5, 3.0, 3.5, 4.0, 20], labels=["< 2.0 s", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0+"], right=False)
    t["nrush_bin"] = pd.cut(t["n_rush"], [0, 3, 4, 5, 20], labels=["3 or fewer", "4", "5", "6+"])
    t["yac_resid"] = np.where(comp_rows[is_t], y_cap[is_t] - t["yac_mean_a"], np.nan)
    for lab, col in (("Nearest rusher to the QB at release", "rush_bin"), ("Number of pass rushers", "nrush_bin"), ("Time from snap to throw", "ttt_bin")):
        say(f"**{lab}**")
        table(t.groupby(col, observed=True).agg(attempts=("complete", "size"), completion_rate=("complete", "mean"),
                                                model_p_comp=("p_comp_d", "mean"), mean_yac=("yac", "mean")).round(3))
    say("**By the PFF pressure flag (charged after the play)**")
    table(t.groupby("pressured").agg(attempts=("complete", "size"), completion_rate=("complete", "mean"), mean_yac=("yac", "mean"),
                                     yac_over_expected=("yac_resid", "mean")).round(3))
    say("")

    # ── 4 ──
    say("## 4. Which feature families matter")
    imp = pd.DataFrame({"YAC (analysis model): MAE added when scrambled, yd": {k: np.mean(v) for k, v in imp_y.items()},
                        "Completion (decision model): log loss added when scrambled": {k: np.mean(v) for k, v in imp_c.items()}})
    imp = imp.sort_values(imp.columns[0], ascending=False)
    table(imp.round(4))
    say("Scrambling a whole family overstates its unique value when families overlap. The projection features contain the same positions and velocities as the t0 and t5 families, "
        "so the unique gain is the one in the section 1 ladder (step 4 to step 5: 2.94 to 2.82 yd of error), not the scrambled-family figure.")
    say("")

    # ── 5 ──
    say("## 5. Coverage scheme: what happens relative to what the geometry predicts")
    cm = f[comp_rows].copy()
    cm["yac_resid"] = y_cap[comp_rows] - cm["yac_mean_a"]
    cm["cov"] = cm["pff_passCoverage"].astype(str)
    tab = ci_table(cm, "cov", "yac_resid", min_n=100)
    tab["raw mean YAC"] = cm.groupby("cov")["yac"].mean().reindex(tab.index)
    use = cm["cov"].isin(tab.index).to_numpy()
    codes, uniq = pd.factorize(cm.loc[use, "cov"])
    obs_z, null = scheme_test(codes, cm.loc[use, "yac_resid"].to_numpy(), cm.loc[use, "gameId"].to_numpy())
    tab["z"] = pd.Series(obs_z, index=uniq).reindex(tab.index)
    tab["family-wise p"] = [float((null >= abs(z)).mean()) for z in tab["z"]]
    crit = float(np.quantile(null, 0.95))
    say(f"YAC over expected = actual YAC minus the analysis model's expectation, by PFF coverage. Interval = 95%, bootstrap by game. "
        f"Family-wise p comes from shuffling coverage labels within each game 2,000 times and asking how often the largest |z| across all {len(tab)} schemes "
        f"is as big (a |z| above {crit:.2f} is significant at 5% after allowing for {len(tab)} comparisons). Overall mean residual {cm['yac_resid'].mean():+.3f}.")
    table(tab.round(3))
    say("")

    zone = cm[cm["pff_passCoverageType"].astype(str) == "Zone"].copy()
    man = cm[cm["pff_passCoverageType"].astype(str) == "Man"].copy()

    def vs_rest(base, name, mask_name):
        base = base.copy()
        base["grp"] = np.where(base["cov"] == name, name, mask_name)
        labs, obs, draws, _ = boot_group_means(base, "grp", "yac_resid")
        a, b = labs.index(name), labs.index(mask_name)
        d = draws[:, a] - draws[:, b]
        return obs[a] - obs[b], np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)
    d2 = vs_rest(zone, "Cover-2", "Other zone")
    d1 = vs_rest(man, "Cover-1", "Other man")
    say(f"Cover-2 minus other zone coverages (Cover-3, Cover-6, Quarters): {d2[0]:+.2f} yd (95% interval {d2[1]:+.2f} to {d2[2]:+.2f}). "
        f"Cover-1 minus other man coverages (Cover-0, 2-Man): {d1[0]:+.2f} yd (95% interval {d1[1]:+.2f} to {d1[2]:+.2f}).")
    say("")

    def subgroup(name):
        c = cm[cm["cov"] == name].copy()
        c["Depth at arrival"] = pd.cut(c["x_rel_d"], [-100, 3, 8, 100], labels=["behind LOS to 3 yd", "3-8 yd", "8+ yd"])
        c["Down"] = c["down"].clip(upper=3).map({1: "1st", 2: "2nd", 3: "3rd/4th"})
        c["Receiver"] = c["pos_group"].astype(str)
        c["Formation"] = np.where(c["offenseFormation"].astype(str).isin(["SHOTGUN", "EMPTY", "PISTOL"]), "Shotgun/empty/pistol", "Under center")
        c["Rushers"] = np.where(c["n_rush"] >= 5, "5+ rushers", "3-4 rushers")
        c["Time to throw"] = np.where(c["time_to_throw"] >= 2.7, "2.7 s or more", "under 2.7 s")
        out = []
        for g in ["Depth at arrival", "Down", "Receiver", "Formation", "Rushers", "Time to throw"]:
            tt = ci_table(c, g, "yac_resid", min_n=30)
            tt.insert(0, "split", g)
            out.append(tt)
        return pd.concat(out)
    sub2, sub1 = subgroup("Cover-2"), subgroup("Cover-1")
    say(f"**Cover-2 subgroups** (n = {int((cm['cov'] == 'Cover-2').sum())}): mean YAC over expected")
    table(sub2.round(2))
    say(f"**Cover-1 subgroups** (n = {int((cm['cov'] == 'Cover-1').sum())}): mean YAC over expected")
    table(sub1.round(2))

    # does the scheme effect depend on which "expected" we subtract? three baselines with different information
    rb = cm.copy()
    rb["resid_decision"] = y_cap[comp_rows] - rb["yac_mean_d"]
    rb["resid_nfl"] = np.where(rb["nfl_xyac"].notna(), np.minimum(rb["yac"], CAP) - rb["nfl_xyac"], np.nan)
    base_cols = {"analysis model": "yac_resid", "decision model (release only)": "resid_decision", "nflfastR xYAC (no tracking)": "resid_nfl"}
    rob, rob_ci = {}, {}
    for lab, col in base_cols.items():
        d = rb.dropna(subset=[col])
        tt = ci_table(d, "cov", col, min_n=100)
        rob[lab] = tt["mean"]
        rob_ci[lab] = tt
    say("Does the scheme effect depend on the baseline? Mean YAC minus expected YAC, three different expectations:")
    table(pd.DataFrame(rob).round(2))
    say("The nflfastR column is offset downward for every scheme (its expectation is fit to all plays, including the screens that are missing here), "
        "so compare each scheme with the others in that column and not with zero. Cover-2 is the clear low outlier on all three baselines.")
    say("95% intervals for the two flagged schemes: " + "; ".join(
        f"{lab}: Cover-2 {rob_ci[lab].loc['Cover-2', 'mean']:+.2f} ({rob_ci[lab].loc['Cover-2', 'lo']:+.2f} to {rob_ci[lab].loc['Cover-2', 'hi']:+.2f}), "
        f"Cover-1 {rob_ci[lab].loc['Cover-1', 'mean']:+.2f} ({rob_ci[lab].loc['Cover-1', 'lo']:+.2f} to {rob_ci[lab].loc['Cover-1', 'hi']:+.2f})" for lab in base_cols))
    say("")

    gf = ["sep1_p", "n_def10_p", "t_close_p", "cone_dist_p", "n_block10_p", "lev_dx_p", "x_rel_p", "tau", "time_to_throw", "n_rush"]
    z2 = zone.assign(grp=np.where(zone["cov"] == "Cover-2", "Cover-2", "Other zone")).groupby("grp")[gf].mean().T
    z2["difference"] = z2["Cover-2"] - z2["Other zone"]
    say("Is Cover-2 simply a different situation? Average geometry on Cover-2 vs other zone catches:")
    table(z2.round(2))
    m1 = man.assign(grp=np.where(man["cov"] == "Cover-1", "Cover-1", "Other man")).groupby("grp")[gf].mean().T
    m1["difference"] = m1["Cover-1"] - m1["Other man"]
    say("Average geometry on Cover-1 vs other man catches:")
    table(m1.round(2))

    at = f[is_t].copy()
    at["cov"] = at["pff_passCoverage"].astype(str)
    at["comp_resid"] = at["complete"] - at["p_comp_a"]
    ctab = ci_table(at, "cov", "comp_resid", min_n=100)
    ctab["completion rate"] = at.groupby("cov")["complete"].mean().reindex(ctab.index)
    ctab["model p_comp"] = at.groupby("cov")["p_comp_a"].mean().reindex(ctab.index)
    say("")
    say("Completion rate over expected by coverage (actual minus the analysis model's probability):")
    table(ctab.round(3))
    say("")

    # ── 6 ──
    say("## 6. Receiver opportunity: scoring every eligible receiver at release (decision model)")
    s = f.copy()
    s["yac_opp"] = s["p_comp_d"] * s["yac_mean_d"]
    s["ev_yards"] = s["p_comp_d"] * (s["x_rel_d"].clip(lower=-8) + s["yac_mean_d"])
    s["ev_capped"] = s["p_comp_d"] * (s["x_rel_d"].clip(-8, 20) + s["yac_mean_d"])  # credit at most 20 yd of depth
    s["n_elig"] = s.groupby(KEYS)["nflId"].transform("size")
    for col in ("p_comp_d", "yac_opp", "ev_yards", "ev_capped"):
        s["rank_" + col] = s.groupby(KEYS)[col].rank(ascending=False, method="min")
    tg, nt = s[s["is_target"]], s[~s["is_target"]]
    multi = tg[tg["n_elig"] >= 2]
    chk = ["p_comp_d", "yac_mean_d", "sep1_t0", "sep1_d", "t_close_d", "cone_dist_d"]
    say("Sanity check. Targets should look more open than the receivers who were not thrown to, using only release-time measurements:")
    table(pd.DataFrame({"target": tg[chk].mean(), "not targeted": nt[chk].mean()}).round(3))
    rand = (1.0 / multi["n_elig"]).mean()
    rws = []
    for col, lab in (("p_comp_d", "highest completion probability"), ("yac_opp", "highest YAC opportunity (p x E[YAC])"),
                     ("ev_yards", "highest expected yards, uncapped depth (p x (depth + E[YAC]))"),
                     ("ev_capped", "highest expected yards, depth credit capped at 20 yd")):
        r = multi["rank_" + col]
        rws.append({"receiver ranking": lab, "target ranked first": float((r == 1).mean()), "random pick": float(rand),
                    "mean rank of target": float(r.mean()), "mean eligible": float(multi["n_elig"].mean())})
    rr = pd.DataFrame(rws)
    table(rr.set_index("receiver ranking").round(3))
    e2 = s[s["n_elig"] >= 2]
    tgt2 = e2["is_target"]

    def gap_of(col):
        return (e2.groupby(KEYS)[col].transform("max") - e2[col])[tgt2]
    gap, gap_cap, gap_yac = gap_of("ev_yards"), gap_of("ev_capped"), gap_of("yac_opp")
    best_row = e2.loc[e2.groupby(KEYS)["ev_yards"].idxmax()]
    say("Gap between the best option and the receiver thrown to (the model does not see the coverage read, the route concept, the decoys or what the quarterback saw, "
        "so a gap marks a throw worth reviewing and is not evidence of a mistake):")
    table(pd.DataFrame({"mean gap (yd)": [gap.mean(), gap_cap.mean(), gap_yac.mean()], "median": [gap.median(), gap_cap.median(), gap_yac.median()],
                        "share above 2 yd": [(gap > 2).mean(), (gap_cap > 2).mean(), (gap_yac > 2).mean()],
                        "share above 5 yd": [(gap > 5).mean(), (gap_cap > 5).mean(), (gap_yac > 5).mean()]},
                       index=["expected yards, uncapped depth", "expected yards, depth credit capped at 20 yd", "YAC opportunity only (p x E[YAC])"]).round(3))
    say(f"The uncapped version is dominated by deep options: on {(best_row['x_rel_d'] > 25).mean():.1%} of throws its best option is more than 25 yd downfield, "
        "where the model has the least training data. Use the capped and YAC-only versions.")
    trg = tg.dropna(subset=["prePenaltyPlayResult"])
    r_ = trg["ev_yards"].corr(trg["prePenaltyPlayResult"])
    say(f"Does expected yards for the target predict what the throw gained? Correlation {r_:.3f} (R2 {r_ ** 2:.3f}).")
    dec = calib(trg["ev_yards"], trg["prePenaltyPlayResult"])
    dec.columns = ["n", "predicted expected yards", "actual mean yards"]
    table(dec.round(2))

    keep = (["gameId", "playId", "nflId", "displayName", "is_target", "pos_group", "n_elig", "p_comp_d", "yac_mean_d"] + [f"{q}_d" for q in QN] +
            ["p_ge10_d", "x_rel_d", "yac_opp", "ev_yards", "ev_capped", "rank_p_comp_d", "rank_yac_opp", "rank_ev_yards", "rank_ev_capped", "p_comp_a", "yac_mean_a"] +
            [f"{q}_a" for q in QN] + ["p_ge10_a", "complete", "yac", "air_yards", "passResult", "pff_passCoverage"])
    s[keep].rename(columns={"p_comp_d": "p_comp", "yac_mean_d": "yac_mean", "p_ge10_d": "p_ge10", **{f"{q}_d": q for q in QN}}).to_csv(
        os.path.join(PROC, "yac_scores.csv"), index=False)
    say("")
    say("## Limits")
    say("- Scope: the 6,805 thrown passes in plays.csv with a named target and no accepted penalty. plays.csv leaves out about 1,150 completions, mostly passes at or behind the line of "
        "scrimmage that gain more after the catch (mean 8.5 yd vs 4.4), and they have no tracking. Findings apply to passes thrown beyond the line.")
    say("- The catch is never tracked. The analysis model uses the first 0.5 s of ball flight and a constant-velocity projection to the estimated arrival; the decision model uses release only.")
    say("- Opportunity scores for receivers who were not thrown to extrapolate a model trained only on targeted receivers.")
    say("- Eight weeks of one season. Individual receivers and defenses do not have enough catches for stable ratings (see yac_probe.py and yac_defense_probe.py).")
    say(f"Run time {time.time() - t_start:.0f}s.")

    make_figures(f, res, cal_ge, cal_c, imp, tab, ctab, sub2, sub1, t, dec, rr, gap_cap)
    with open(os.path.join(ROOT, "yac_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))
    print("\nsaved processed/yac_scores.csv, yac_report.md and figures/yac_*.png")


# ── figures ──────────────────────────────────────────────────────────────────

def forest(ax, tab, title, xlabel, highlight=None):
    t = tab.iloc[::-1]
    y = np.arange(len(t))
    cl = [RED if (highlight and i == highlight) else BLUE for i in t.index]
    ax.hlines(y, t["lo"], t["hi"], color=cl, lw=2)
    ax.scatter(t["mean"], y, color=cl, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i} (n={int(n)})" for i, n in zip(t.index, t["n"])], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)


def make_figures(f, res, cal_ge, cal_c, imp, tab, ctab, sub2, sub1, t, dec, rr, gap):
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    fig.suptitle("Expected YAC, YAC range, completion probability and pass rush (all scores out-of-fold, grouped by game)", fontsize=14)
    a = ax[0, 0]
    r = res[res["model"].str.match(r"^\d\.|^A\.|^B\.")]
    a.barh([m.strip() for m in r["model"]], r["MAE"], color=[GREY if m[0] in "AB" else BLUE for m in r["model"]])
    a.invert_yaxis()
    for i, v in enumerate(r["MAE"]):
        a.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    a.set_xlim(2.4, r["MAE"].max() + 0.35)
    a.tick_params(axis="y", labelsize=7)
    a.set_title("Average error in yards (lower is better; grey = benchmarks)", fontsize=10)
    a = ax[0, 1]
    srt = f[f["is_target"] & (f["complete"] == 1) & f["yac"].notna()].copy()
    srt["bin"] = pd.qcut(srt["yac_mean_a"].rank(method="first"), 10, labels=False)
    g = srt.groupby("bin").agg(p=("yac_mean_a", "mean"), a=("yac", lambda v: v.clip(upper=CAP).mean()), lo=("yac", lambda v: np.quantile(v, .1)),
                               md=("yac", "median"), hi=("yac", lambda v: np.quantile(v, .9)))
    a.fill_between(g["p"], g["lo"], g["hi"], color=BLUE, alpha=0.2, label="actual 10th-90th percentile")
    a.plot(g["p"], g["a"], "o-", color=BLUE, label="actual mean")
    a.plot(g["p"], g["md"], "s--", color=GREY, label="actual median")
    lim = [0, max(g["p"].max(), g["a"].max()) + 1]
    a.plot(lim, lim, color="k", lw=0.8, ls=":")
    a.set_xlabel("Expected YAC from the model (decile averages)"); a.set_ylabel("Actual YAC (yd)")
    a.set_title("Calibration of expected YAC", fontsize=10); a.legend(fontsize=8)
    a = ax[0, 2]
    a.plot(cal_ge["predicted"], cal_ge["actual"], "o-", color=ORANGE, label="P(YAC >= 10)")
    a.plot(cal_c["predicted"], cal_c["actual"], "s-", color=GREEN, label="P(complete), decision model")
    a.plot([0, 1], [0, 1], color="k", lw=0.8, ls=":")
    a.set_xlabel("Predicted probability (decile averages)"); a.set_ylabel("Actual rate")
    a.set_title("Probability calibration", fontsize=10); a.legend(fontsize=8)
    a = ax[1, 0]
    i2 = imp.sort_values(imp.columns[0])
    a.barh(np.arange(len(i2)), i2.iloc[:, 0], color=BLUE)
    a.set_yticks(np.arange(len(i2))); a.set_yticklabels(i2.index, fontsize=8)
    a.set_title("What drives expected YAC\n(yards of error added when a family is scrambled)", fontsize=10)
    a = ax[1, 1]
    i3 = imp.sort_values(imp.columns[1])
    a.barh(np.arange(len(i3)), i3.iloc[:, 1], color=GREEN)
    a.set_yticks(np.arange(len(i3))); a.set_yticklabels(i3.index, fontsize=8)
    a.set_title("What drives completion probability at release\n(log loss added when a family is scrambled)", fontsize=10)
    a = ax[1, 2]
    g = t.groupby("rush_bin", observed=True).agg(n=("complete", "size"), rate=("complete", "mean"), model=("p_comp_d", "mean"))
    xx = np.arange(len(g))
    a.bar(xx - 0.2, g["rate"] * 100, width=0.4, color=GREEN, label="actual completion rate")
    a.bar(xx + 0.2, g["model"] * 100, width=0.4, color=GREY, label="model probability")
    for i, (v, n) in enumerate(zip(g["rate"] * 100, g["n"])):
        a.text(i - 0.2, v, f"{v:.0f}%\nn={n:,}", ha="center", va="bottom", fontsize=7)
    a.set_xticks(xx); a.set_xticklabels(g.index)
    a.set_xlabel("Nearest pass rusher to the QB at release"); a.set_ylim(0, 100); a.legend(fontsize=8, loc="lower right")
    a.set_title("Pass rush vs completion rate", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "yac_01_models.png"), dpi=110, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(24, 6.5))
    fig.suptitle("Coverage scheme: what happens after the catch compared with what the geometry predicts", fontsize=13)
    forest(ax[0], tab[["n", "mean", "lo", "hi"]], "YAC over expected (95% interval, bootstrap by game)", "yards per catch", highlight="Cover-2")
    forest(ax[1], ctab[["n", "mean", "lo", "hi"]] * np.array([1, 100, 100, 100]), "Completion rate over expected", "percentage points", highlight="Cover-2")
    for a, sub, nm in ((ax[2], sub2, "Cover-2"), (ax[3], sub1, "Cover-1")):
        sb = sub.copy()
        sb.index = [f"{x}: {y}" for x, y in zip(sb["split"], sb.index)]
        forest(a, sb[["n", "mean", "lo", "hi"]], f"{nm} YAC over expected, by subgroup", "yards per catch")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "yac_02_coverage.png"), dpi=110, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("Receiver opportunity at release: was the ball thrown to the best option?", fontsize=13)
    a = ax[0]
    r = rr.set_index("receiver ranking")
    xx = np.arange(len(r))
    a.bar(xx - 0.2, r["target ranked first"] * 100, width=0.4, color=BLUE, label="target ranked first")
    a.bar(xx + 0.2, r["random pick"] * 100, width=0.4, color=GREY, label="random pick")
    short = ["completion\nprobability", "YAC\nopportunity", "expected yards\n(uncapped depth)", "expected yards\n(depth capped\nat 20 yd)"]
    a.set_xticks(xx); a.set_xticklabels(short[: len(r)], fontsize=8)
    a.set_ylabel("% of throws"); a.legend(); a.set_title("How often the target was the model's top option", fontsize=10)
    a = ax[1]
    a.hist(gap.clip(upper=15), bins=np.arange(0, 15.5, 0.5), color=BLUE, edgecolor="white")
    a.axvline(2, color=RED, ls="--", label=f"above 2 yd on {(gap > 2).mean():.0%} of throws")
    a.set_xlabel("Expected yards of best option minus receiver thrown to\n(depth credit capped at 20 yd; clipped at 15)"); a.legend()
    a.set_title("Expected-yard gap to the best option", fontsize=10)
    a = ax[2]
    a.plot(dec["predicted expected yards"], dec["actual mean yards"], "o-", color=ORANGE)
    vals = dec[["predicted expected yards", "actual mean yards"]]
    lim = [min(vals.min().min(), 0) - 0.5, vals.max().max() + 0.5]
    a.plot(lim, lim, color="k", lw=0.8, ls=":")
    a.set_xlabel("Expected yards of the target (decile averages)"); a.set_ylabel("Actual yards gained")
    a.set_title("Does expected yards predict the outcome?", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG, "yac_03_opportunity.png"), dpi=110, facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main()
