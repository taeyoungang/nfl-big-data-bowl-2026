"""
Feasibility probe: can yards after catch (YAC) be predicted from tracking at the moment of the throw?

Target     : nflverse `yards_after_catch` for completed passes in plays.csv (no penalty).
Receiver   : found in tracking by matching nflverse `receiver_jersey_number` to the offense.
Features   : measured at the pass_forward frame (t0) and at the last tracked frame (t5 = t0 + 0.5 s),
             because tracking stops 0.5 s after the throw and the catch itself is never observed.
Evaluation : 5-fold cross-validation grouped by game, so a game is never in train and test together.

Writes processed/yac_features.csv for reuse.

Usage:
    python yac_probe.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
NV_PATH = os.path.join(ROOT, "external", "play_by_play_2021.parquet")
OUT = os.path.join(ROOT, "processed")
KEYS = ["gameId", "playId"]
THROW = {"pass_forward", "autoevent_passforward"}
SNAP = {"ball_snap", "autoevent_ballsnap"}
FIELD_W = 53.3
YAC_CAP = 30  # cap on the training target; a few huge runs would otherwise dominate squared error


# ── target table ─────────────────────────────────────────────────────────────

def load_targets():
    plays = pd.read_csv(os.path.join(DATA, "plays.csv"))
    cols = ["old_game_id", "play_id", "complete_pass", "penalty", "yards_after_catch", "air_yards",
            "receiver_jersey_number", "yardline_100", "xyac_mean_yardage"]
    nv = pd.read_parquet(NV_PATH, columns=cols)
    nv["gameId"] = pd.to_numeric(nv["old_game_id"], errors="coerce")
    nv = nv.dropna(subset=["gameId", "play_id"])
    nv["gameId"], nv["playId"] = nv["gameId"].astype("int64"), nv["play_id"].astype("int64")
    nv = nv.drop(columns=["old_game_id", "play_id"]).rename(columns={c: "nv_" + c for c in cols[2:]})
    allc = plays.merge(nv, on=KEYS, how="left")

    # How different are the completions that plays.csv leaves out? (matters for any YAC model)
    full = nv[(nv["nv_complete_pass"] == 1) & (nv["nv_penalty"] != 1) & nv["gameId"].isin(plays["gameId"])]
    full = full.merge(plays[KEYS].assign(inc=True), on=KEYS, how="left")
    full["inc"] = full["inc"].fillna(False).astype(bool)
    print("=== Completions in these games, with and without a row in plays.csv ===")
    for flag, name in ((True, "in plays.csv (has tracking)"), (False, "NOT in plays.csv (no tracking)")):
        x = full[full["inc"] == flag]
        print(f"  {name:32s} n={len(x):5,}  mean YAC {x['nv_yards_after_catch'].mean():.2f}  "
              f"median {x['nv_yards_after_catch'].median():.1f}  mean air yards {x['nv_air_yards'].mean():.1f}")

    t = allc[(allc["passResult"] == "C") & (allc["nv_penalty"] != 1) & allc["nv_yards_after_catch"].notna()
             & allc["nv_receiver_jersey_number"].notna()].copy()
    t["jersey"] = pd.to_numeric(t["nv_receiver_jersey_number"], errors="coerce")
    print(f"\ncompletions in plays.csv with a YAC value and receiver jersey: {len(t):,}")
    return t


# ── per-frame geometry ───────────────────────────────────────────────────────

def vel(s, d, sgn):
    r = np.radians(d)  # NGS convention: degrees clockwise from +y; x = sin, y = cos
    return s * np.sin(r) * sgn, s * np.cos(r) * sgn


def frame_block(fr, rid, defense, sgn, los):
    r = fr[fr["nflId"] == rid]
    if len(r) != 1:
        return None
    r = r.iloc[0]
    d = fr[(fr["team"] == defense) & fr["nflId"].notna()]
    if len(d) < 8:
        return None
    dx, dy = (d["x"].values - r["x"]) * sgn, (d["y"].values - r["y"]) * sgn
    dist = np.hypot(dx, dy)
    order = np.sort(dist)
    rvx, rvy = vel(r["s"], r["dir"], sgn)
    dvx, dvy = vel(d["s"].values, d["dir"].values, sgn)
    i = int(dist.argmin())
    ux, uy = dx[i] / max(dist[i], 1e-6), dy[i] / max(dist[i], 1e-6)
    cone = (dx > 0) & (np.abs(dy) <= dx)  # defenders in a 90-degree wedge straight downfield
    xp = r["x"] if sgn == 1 else 120 - r["x"]
    return {
        "sep1": order[0], "sep2": order[1],
        "n_def5": int((dist < 5).sum()),
        "close_speed": -((dvx[i] - rvx) * ux + (dvy[i] - rvy) * uy),
        "cone_dist": float(dist[cone].min()) if cone.any() else 30.0,
        "n_ahead10": int(((dx > 0) & (dx < 10) & (np.abs(dy) < 5)).sum()),
        "x_rel": (r["x"] - los) * sgn,
        "sideline": min(r["y"], FIELD_W - r["y"]),
        "goal_dist": 110 - xp,
        "speed": r["s"], "acc": r["a"], "vx": rvx, "vy": rvy,
    }


def build_features(t):
    files = {int(re.search(r"tracking_(\d+)", f).group(1)): f for f in glob.glob(os.path.join(DATA, "tracking", "*.csv"))}
    cols = ["gameId", "playId", "nflId", "frameId", "jerseyNumber", "team", "playDirection", "x", "y", "s", "a", "dir", "event"]
    rows, skip = [], {"no throw frame": 0, "receiver not unique": 0, "no frame data": 0}
    for gid, sub in t.groupby("gameId"):
        df = pd.read_csv(files[gid], usecols=cols)
        df = df[df["playId"].isin(sub["playId"])]
        by_play = {pid: g for pid, g in df.groupby("playId")}
        for _, p in sub.iterrows():
            g = by_play.get(p["playId"])
            if g is None:
                skip["no frame data"] += 1
                continue
            ev = g[g["event"].notna()]
            t0s = ev.loc[ev["event"].isin(THROW), "frameId"]
            if t0s.empty:
                skip["no throw frame"] += 1
                continue
            t0 = int(t0s.min())
            snap = ev.loc[ev["event"].isin(SNAP), "frameId"]
            f0 = g[g["frameId"] == t0]
            rec = f0[(f0["team"] == p["possessionTeam"]) & (f0["jerseyNumber"] == p["jersey"])]
            if len(rec) != 1:
                skip["receiver not unique"] += 1
                continue
            rid = rec["nflId"].iloc[0]
            sgn = 1 if g["playDirection"].iloc[0] == "right" else -1
            los = p["absoluteYardlineNumber"]
            b0 = frame_block(f0, rid, p["defensiveTeam"], sgn, los)
            t5 = min(t0 + 5, int(g["frameId"].max()))
            b5 = frame_block(g[g["frameId"] == t5], rid, p["defensiveTeam"], sgn, los)
            if b0 is None or b5 is None:
                skip["no frame data"] += 1
                continue
            ball = f0[f0["team"] == "football"]
            rr = rec.iloc[0]
            rec_row = {"gameId": gid, "playId": p["playId"], "nflId": rid}
            rec_row.update({k + "_t0": v for k, v in b0.items()})
            rec_row.update({k + "_t5": v for k, v in b5.items()})
            rec_row["sep1_delta"] = b5["sep1"] - b0["sep1"]
            rec_row["time_to_throw"] = (t0 - int(snap.min())) / 10 if len(snap) else np.nan
            if len(ball):
                bx, by = ball["x"].iloc[0], ball["y"].iloc[0]
                rec_row["release_depth"] = -(bx - los) * sgn
                rec_row["throw_dist"] = float(np.hypot(rr["x"] - bx, rr["y"] - by))
            rows.append(rec_row)
    print(f"\nreceiver found and features built for {len(rows):,} of {len(t):,} completions; skipped: {skip}")
    return pd.DataFrame(rows)


# ── assemble model table ─────────────────────────────────────────────────────

def assemble(t, feats):
    plays_ctx = t[KEYS + ["down", "yardsToGo", "defendersInBox", "pff_playAction", "pff_passCoverageType", "pff_passCoverage",
                          "offenseFormation", "nv_air_yards", "nv_yardline_100", "nv_yards_after_catch", "nv_xyac_mean_yardage"]]
    m = feats.merge(plays_ctx, on=KEYS)
    pff = pd.read_csv(os.path.join(DATA, "pffScoutingData.csv"),
                      usecols=KEYS + ["nflId", "pff_positionLinedUp", "pff_hit", "pff_hurry", "pff_sack"])
    pr = pff.groupby(KEYS)[["pff_hit", "pff_hurry", "pff_sack"]].sum().sum(axis=1).gt(0).rename("pressured").astype(int).reset_index()
    m = m.merge(pff[KEYS + ["nflId", "pff_positionLinedUp"]], on=KEYS + ["nflId"], how="left").merge(pr, on=KEYS, how="left")
    pl = pd.read_csv(os.path.join(DATA, "players.csv"))
    hh = pl["height"].str.extract(r"^(\d)-(\d{1,2})$").astype(float)
    pl["height_in"] = hh[0] * 12 + hh[1]
    grp = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB"}
    pl["pos_group"] = pl["officialPosition"].map(grp).fillna("other")
    m = m.merge(pl[["nflId", "weight", "height_in", "pos_group"]], on="nflId", how="left")
    m = m.rename(columns={"nv_air_yards": "air_yards", "pff_passCoverageType": "cov_type", "pff_positionLinedUp": "pos_lined_up",
                          "nv_yards_after_catch": "yac", "nv_xyac_mean_yardage": "nfl_xyac"})
    m["goal_dist_at_catch"] = m["nv_yardline_100"] - m["air_yards"]
    for c in ("cov_type", "pos_lined_up", "pos_group", "offenseFormation"):
        m[c] = m[c].astype("category")
    m["yac_capped"] = m["yac"].clip(upper=YAC_CAP)
    return m


FAMILIES = {
    "Throw context (no tracking)": ["air_yards", "goal_dist_at_catch", "down", "yardsToGo", "defendersInBox",
                                    "pff_playAction", "cov_type", "offenseFormation"],
    "Receiver role and body": ["pos_lined_up", "pos_group", "weight", "height_in"],
    "Separation from nearest defenders": ["sep1_t0", "sep2_t0", "n_def5_t0", "close_speed_t0",
                                          "sep1_t5", "sep2_t5", "n_def5_t5", "close_speed_t5", "sep1_delta"],
    "Open field ahead": ["cone_dist_t0", "n_ahead10_t0", "cone_dist_t5", "n_ahead10_t5"],
    "Receiver motion and position": ["speed_t0", "acc_t0", "vx_t0", "vy_t0", "speed_t5", "acc_t5", "vx_t5", "vy_t5",
                                     "x_rel_t0", "sideline_t0", "goal_dist_t0", "x_rel_t5", "sideline_t5", "goal_dist_t5"],
    "Quarterback and pocket": ["time_to_throw", "release_depth", "throw_dist", "pressured"],
}


def model():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=30,
                                         l2_regularization=1.0, categorical_features="from_dtype", random_state=0)


def cv_predict(m, cols, k=5):
    oof = np.zeros(len(m))
    for tr, te in GroupKFold(k).split(m, groups=m["gameId"]):
        mdl = model().fit(m.iloc[tr][cols], m.iloc[tr]["yac_capped"])
        oof[te] = mdl.predict(m.iloc[te][cols])
    return oof


def report(name, y, p):
    rho = np.nan if np.ptp(p) == 0 else spearmanr(y, p)[0]  # a constant prediction has no rank correlation
    return {"model": name, "MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "R2": r2_score(y, p), "Spearman": rho}


def main():
    t = load_targets()
    feats = build_features(t)
    m = assemble(t, feats)
    os.makedirs(OUT, exist_ok=True)
    m.to_csv(os.path.join(OUT, "yac_features.csv"), index=False)

    y = m["yac_capped"].values
    print(f"\n=== Target: yards after catch, n={len(m):,} ===")
    print(m["yac"].describe().round(2).to_dict())
    print(f"share of YAC = 0: {(m['yac'] == 0).mean():.1%} | share above {YAC_CAP}: {(m['yac'] > YAC_CAP).mean():.2%} (capped at {YAC_CAP} for fitting)")
    print("by receiver group (n, mean YAC):", m.groupby("pos_group", observed=True)["yac"].agg(["size", "mean"]).round(2).to_dict("index"))

    print("\n=== Single features vs YAC (Spearman rank correlation, all completions) ===")
    uni = []
    for fam, cols in FAMILIES.items():
        for c in cols:
            if str(m[c].dtype) == "category":
                continue
            uni.append((c, fam, spearmanr(m[c], m["yac"], nan_policy="omit")[0]))
    uni = pd.DataFrame(uni, columns=["feature", "family", "rho"]).sort_values("rho", key=abs, ascending=False)
    print(uni.head(14).round(3).to_string(index=False))

    print("\n=== Out-of-fold prediction, 5-fold CV grouped by game ===")
    res = [report("A. League average YAC", y, np.full(len(y), y.mean()))]
    valid = m["nfl_xyac"].notna()
    r = report("B. nflfastR xYAC model (benchmark)", y[valid], m.loc[valid, "nfl_xyac"].values)
    res.append(r)
    ctx = FAMILIES["Throw context (no tracking)"]
    role = FAMILIES["Receiver role and body"]
    sets = [("C. Throw context + receiver role", ctx + role),
            ("D. C + separation", ctx + role + FAMILIES["Separation from nearest defenders"]),
            ("E. D + open field ahead", ctx + role + FAMILIES["Separation from nearest defenders"] + FAMILIES["Open field ahead"]),
            ("F. E + receiver motion and position", ctx + role + sum([FAMILIES[k] for k in list(FAMILIES)[2:5]], [])),
            ("G. All features", sum(FAMILIES.values(), []))]
    preds = {}
    for name, cols in sets:
        preds[name] = cv_predict(m, cols)
        res.append(report(name, y, preds[name]))
    print(pd.DataFrame(res).round(3).to_string(index=False))

    best_name, best_cols = sets[-1]
    p = preds[best_name]
    print(f"\n=== {best_name}: error by receiver group ===")
    for g, idx in m.groupby("pos_group", observed=True).groups.items():
        ii = m.index.get_indexer(idx)
        if len(ii) >= 100:
            rr = report(str(g), y[ii], p[ii])
            print(f"  {g:6s} n={len(ii):5,}  MAE {rr['MAE']:.2f}  R2 {rr['R2']:.3f}  Spearman {rr['Spearman']:.3f}  (average-only MAE {mean_absolute_error(y[ii], np.full(len(ii), y[ii].mean())):.2f})")

    print("\n=== Which feature families matter? Permute a whole family in held-out games; higher = more important ===")
    rng = np.random.default_rng(0)
    base_mae, fam_inc = [], {k: [] for k in FAMILIES}
    for tr, te in GroupKFold(5).split(m, groups=m["gameId"]):
        mdl = model().fit(m.iloc[tr][best_cols], m.iloc[tr]["yac_capped"])
        Xte, yte = m.iloc[te][best_cols].copy(), m.iloc[te]["yac_capped"].values
        b = mean_absolute_error(yte, mdl.predict(Xte))
        base_mae.append(b)
        for fam, cols in FAMILIES.items():
            inc = []
            for _ in range(5):
                perm = rng.permutation(len(Xte))
                Xp = Xte.copy()
                Xp[cols] = Xte[cols].iloc[perm].values
                inc.append(mean_absolute_error(yte, mdl.predict(Xp)) - b)
            fam_inc[fam].append(np.mean(inc))
    print(pd.Series({k: np.mean(v) for k, v in fam_inc.items()}).sort_values(ascending=False).round(3).rename("MAE increase (yd)").to_string())

    # ── can the model rate individual receivers? ──
    print("\n=== Receiver-level reliability (YAC over expected = actual capped YAC minus out-of-fold prediction) ===")
    d = m[["gameId", "playId", "nflId", "pos_group", "yac_capped"]].copy()
    d["resid"] = d["yac_capped"] - p
    n_by = d.groupby("nflId").size()
    print(f"catches per receiver: median {n_by.median():.0f}, 90th percentile {n_by.quantile(.9):.0f}, max {n_by.max()}; "
          f"receivers with >= 10 catches: {(n_by >= 10).sum()}, >= 20: {(n_by >= 20).sum()}")
    d = d.sort_values(["gameId", "playId"])
    d["half"] = d.groupby("nflId").cumcount() % 2  # alternate catches into two halves
    for min_n in (10, 20):
        ids = n_by[n_by >= min_n].index
        s = d[d["nflId"].isin(ids)].groupby(["nflId", "half"]).agg(raw=("yac_capped", "mean"), over=("resid", "mean")).unstack()
        for col, label in (("raw", "raw mean YAC"), ("over", "YAC over expected")):
            r = spearmanr(s[(col, 0)], s[(col, 1)])[0]
            print(f"  receivers with >= {min_n:2d} catches (n={len(s)}): split-half correlation of {label:18s} = {r:.2f}  "
                  f"(full-sample reliability by Spearman-Brown = {2 * r / (1 + r):.2f})")
    print(f"\nsaved {os.path.join('processed', 'yac_features.csv')}")


if __name__ == "__main__":
    main()
