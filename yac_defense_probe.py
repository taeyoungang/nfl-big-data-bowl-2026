"""
Follow-up probe on yac_probe.py output (processed/yac_features.csv).

Q1. Does play context that tracking geometry does not contain (PFF coverage, pressure, time to throw,
    formation, play action) improve held-out YAC prediction?
Q2. Do defenses differ reliably in "YAC allowed over expected"? Receivers did not (split-half ~0).
    Tested by coverage scheme and by defensive team, with a permutation test and split-half reliability.

Usage:
    python yac_defense_probe.py
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from yac_probe import FAMILIES, KEYS, cv_predict, report

ROOT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def load():
    m = pd.read_csv(os.path.join(ROOT, "processed", "yac_features.csv"))
    plays = pd.read_csv(os.path.join(ROOT, "data", "plays.csv"), usecols=KEYS + ["defensiveTeam", "possessionTeam"])
    m = m.merge(plays, on=KEYS, how="left")
    for c in ("cov_type", "pos_lined_up", "pos_group", "offenseFormation"):
        m[c] = m[c].astype("category")
    return m


def split_half(d, key, col):
    d = d.sort_values(["gameId", "playId"]).copy()
    d["half"] = d.groupby(key).cumcount() % 2
    s = d.groupby([key, "half"])[col].mean().unstack().dropna()
    r = spearmanr(s[0], s[1])[0]
    return r, 2 * r / (1 + r), len(s)


def perm_test(d, key, col, n=2000):
    obs = d.groupby(key)[col].mean().std()
    sizes = d.groupby(key).size().values
    vals = d[col].values
    null = []
    for _ in range(n):
        p = rng.permutation(vals)
        edges = np.cumsum(sizes)[:-1]
        null.append(np.std([x.mean() for x in np.split(p, edges)], ddof=1))
    null = np.array(null)
    return obs, null.mean(), (null >= obs).mean()


def main():
    m = load()
    y = m["yac_capped"].values
    all_cols = sum(FAMILIES.values(), [])
    context = ["cov_type", "pff_playAction", "defendersInBox", "offenseFormation", "pressured", "time_to_throw"]
    geometry = [c for c in all_cols if c not in context]

    print("=== Q1. Does play context add to geometry? (5-fold CV grouped by game) ===")
    base = cv_predict(m, geometry)
    full = cv_predict(m, all_cols)
    ctx_only = cv_predict(m, context + ["air_yards", "goal_dist_at_catch", "down", "yardsToGo"])
    rows = [report("Context only (+ air yards, down, distance)", y, ctx_only),
            report("Geometry only (no coverage/pressure/formation/time to throw)", y, base),
            report("Geometry + context", y, full)]
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    m["resid"] = m["yac_capped"] - full
    print("\nresidual mean (should be ~0):", round(m["resid"].mean(), 3), "| residual SD:", round(m["resid"].std(), 2))

    print("\n=== Q2a. YAC over expected by PFF coverage scheme (mean residual, 95% bootstrap CI) ===")
    rows = []
    for cov, d in m.groupby("pff_passCoverage"):
        if len(d) < 100:
            continue
        b = [d["resid"].sample(len(d), replace=True, random_state=int(s)).mean() for s in range(500)]
        rows.append((cov, len(d), d["resid"].mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), d["yac"].mean()))
    print(pd.DataFrame(rows, columns=["coverage", "n", "mean resid (yd)", "ci_lo", "ci_hi", "raw mean YAC"]).round(2).to_string(index=False))
    r = m.groupby("pff_passCoverage")["resid"].agg(["size", "mean"])
    r = r[r["size"] >= 100]
    print("raw YAC differs by coverage; residual differs by coverage only if CIs exclude 0 above.")

    for key, label in (("defensiveTeam", "defensive team (YAC allowed over expected)"), ("possessionTeam", "offensive team (YAC gained over expected)")):
        print(f"\n=== Q2b. {label}: {m[key].nunique()} teams, {m.groupby(key).size().median():.0f} catches per team ===")
        obs, null, p = perm_test(m, key, "resid")
        rho, sb, n = split_half(m, key, "resid")
        print(f"SD of team means: observed {obs:.3f} vs {null:.3f} expected from pure noise; permutation p = {p:.3f}")
        print(f"split-half reliability over {n} teams: Spearman {rho:.2f} -> full-sample (Spearman-Brown) {sb:.2f}")
        t = m.groupby(key)["resid"].agg(["size", "mean"]).sort_values("mean")
        print("lowest 3:", t.head(3).round(2).to_dict("index"))
        print("highest 3:", t.tail(3).round(2).to_dict("index"))


if __name__ == "__main__":
    main()
