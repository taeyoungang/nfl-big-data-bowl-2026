"""
Receiver-level feature table for the expected-YAC / completion-probability / opportunity models.

One row per eligible receiver (PFF role "Pass Route", plus the target) on every thrown pass that has a named
target and no accepted penalty. Tracking stops 0.5 s after the throw and the catch is never observed, so every
feature comes from one of three moments:

    t0 : the pass_forward frame
    t5 : the last tracked frame (0.5 s after the throw)
    p  : a projected arrival moment. Players are moved forward at constant velocity from t5 to the time the ball
         is estimated to reach the receiver (ball speed from its last tracked frames).

All coordinates are normalised so the offense always drives toward +x. Angles follow the NGS convention
(degrees clockwise from +y), so the x component of a heading is sin(angle) and the y component is cos(angle).

Output: processed/receiver_features.csv

Usage:
    python yac_features.py                 # full run
    python yac_features.py --max-games 3   # smoke test
"""

import argparse
import glob
import math
import os
import re
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
NV_PATH = os.path.join(ROOT, "external", "play_by_play_2021.parquet")
OUT = os.path.join(ROOT, "processed")
KEYS = ["gameId", "playId"]
THROW = {"pass_forward", "autoevent_passforward"}
SNAP = {"ball_snap", "autoevent_ballsnap"}
FIELD_W = 53.3
CENTER_Y = FIELD_W / 2
MIN_CLOSE_RATE = 3.0  # yd/s floor used when turning a defender's distance into a time-to-close
POS_GROUP = {"WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB"}


# ── inputs ───────────────────────────────────────────────────────────────────

def load_targets(max_games=None):
    plays = pd.read_csv(os.path.join(DATA, "plays.csv"))
    nv_cols = ["old_game_id", "play_id", "penalty", "yards_after_catch", "air_yards", "receiver_jersey_number", "xyac_mean_yardage"]
    nv = pd.read_parquet(NV_PATH, columns=nv_cols)
    nv["gameId"] = pd.to_numeric(nv["old_game_id"], errors="coerce")
    nv = nv.dropna(subset=["gameId", "play_id"])
    nv["gameId"], nv["playId"] = nv["gameId"].astype("int64"), nv["play_id"].astype("int64")
    nv = nv.drop(columns=["old_game_id", "play_id"]).rename(
        columns={"penalty": "nv_penalty", "yards_after_catch": "yac", "air_yards": "air_yards",
                 "receiver_jersey_number": "target_jersey", "xyac_mean_yardage": "nfl_xyac"})
    t = plays.merge(nv, on=KEYS, how="left")
    t["target_jersey"] = pd.to_numeric(t["target_jersey"], errors="coerce")
    n_all = len(t)
    t = t[t["passResult"].isin(["C", "I", "IN"]) & (t["nv_penalty"] != 1) & t["target_jersey"].notna()].copy()
    print(f"plays.csv rows: {n_all:,} -> thrown passes with a named target and no penalty: {len(t):,}")
    if max_games:
        keep = sorted(t["gameId"].unique())[:max_games]
        t = t[t["gameId"].isin(keep)]
    return t


def pff_index():
    pff = pd.read_csv(os.path.join(DATA, "pffScoutingData.csv"),
                      usecols=KEYS + ["nflId", "pff_role", "pff_positionLinedUp", "pff_hit", "pff_hurry", "pff_sack"])
    idx = {}
    for key, d in pff.groupby(KEYS):
        qb = d.loc[d["pff_role"] == "Pass", "nflId"]
        idx[key] = {
            "qb": int(qb.iloc[0]) if len(qb) else None,
            "rush": set(d.loc[d["pff_role"] == "Pass Rush", "nflId"].astype(int)),
            "routes": set(d.loc[d["pff_role"] == "Pass Route", "nflId"].astype(int)),
            "pos": dict(zip(d["nflId"].astype(int), d["pff_positionLinedUp"])),
            "pressured": bool(d[["pff_hit", "pff_hurry", "pff_sack"]].fillna(0).to_numpy().sum() > 0),
        }
    return idx


# ── geometry ─────────────────────────────────────────────────────────────────

def prep(d, flip):
    """Arrays for one team in one frame, with the offense driving toward +x."""
    x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
    dr, o = d["dir"].to_numpy(float), d["o"].to_numpy(float)
    s, a = d["s"].to_numpy(float), d["a"].to_numpy(float)
    if flip:
        x, y = 120.0 - x, FIELD_W - y
        dr, o = (dr + 180.0) % 360.0, (o + 180.0) % 360.0
    r = np.radians(dr)
    return {"id": d["nflId"].to_numpy("int64"), "jersey": d["jerseyNumber"].to_numpy(float), "x": x, "y": y,
            "vx": s * np.sin(r), "vy": s * np.cos(r), "s": s, "a": a, "face_x": np.sin(np.radians(o))}


def take(P, mask):
    return {k: v[mask] for k, v in P.items()}


def advance(P, dt):
    """Constant-velocity move for dt seconds, kept on the field."""
    Q = dict(P)
    Q["x"] = np.clip(P["x"] + P["vx"] * dt, 0.0, 120.0)
    Q["y"] = np.clip(P["y"] + P["vy"] * dt, 0.0, FIELD_W)
    return Q


def geometry(rx, ry, rvx, rvy, D, O, los):
    """How crowded and how open the space around one receiver is, given defenders D and teammates O."""
    ddx, ddy = D["x"] - rx, D["y"] - ry
    dist = np.hypot(ddx, ddy)
    safe = np.maximum(dist, 1e-6)
    ux, uy = ddx / safe, ddy / safe
    closing = -((D["vx"] - rvx) * ux + (D["vy"] - rvy) * uy)  # > 0 when the defender is closing on the receiver
    order = np.sort(dist)
    i = int(dist.argmin())
    cone = (ddx > 0) & (np.abs(ddy) <= ddx)  # 90-degree wedge straight downfield
    sp = math.hypot(rvx, rvy)
    hx, hy = (rvx / sp, rvy / sp) if sp > 1.0 else (1.0, 0.0)
    toward_middle = 1.0 if ry <= CENTER_Y else -1.0
    ox, oy = O["x"] - rx, O["y"] - ry
    odist = np.hypot(ox, oy)
    n_ahead = int((cone & (dist < 10)).sum())
    n_block = int(((ox > 0) & (np.abs(oy) <= ox) & (odist < 10)).sum())
    return {
        "sep1": order[0], "sep2": order[1],
        "n_def5": int((dist < 5).sum()), "n_def10": int((dist < 10).sum()),
        "close_speed": closing[i],
        "t_close": float(np.min(dist / np.maximum(closing, MIN_CLOSE_RATE))),
        "cone_dist": float(dist[cone].min()) if cone.any() else 30.0,
        "n_ahead10": n_ahead, "n_block10": n_block, "block_net": n_block - n_ahead,
        "lev_dx": ddx[i],                                  # > 0: nearest defender is between the receiver and the goal
        "lev_inside": ddy[i] * toward_middle,              # > 0: nearest defender is on the inside shoulder
        "lev_cos": (hx * ddx[i] + hy * ddy[i]) / safe[i],  # 1: directly in the receiver's path, -1: directly behind
        "x_rel": rx - los, "sideline": min(ry, FIELD_W - ry), "goal_dist": 110.0 - rx,
    }


V_RELEASE = 22.0  # yd/s, league-typical ball speed (median of the tracked throws); the same for every receiver


def estimate_arrival_at_release(r0, ball0):
    """Counterfactual arrival time if the ball left the passer's hand at ball0 and flew straight at this receiver.
    Uses only what is known at release (positions and velocities at t0 and a fixed ball speed), so it is
    defined the same way for the receiver who was thrown to and for every other receiver."""
    tau = 0.8
    for _ in range(3):
        px = min(max(r0[0] + r0[2] * tau, 0.0), 120.0)
        py = min(max(r0[1] + r0[3] * tau, 0.0), FIELD_W)
        dist = math.hypot(px - ball0[0], py - ball0[1])
        tau = min(max(dist / V_RELEASE, 0.3), 4.5)
    return tau, dist


def estimate_arrival(r5, ball5, v_ball, elapsed):
    """Seconds after release at which the ball reaches the receiver, iterating a constant-velocity extrapolation."""
    tau = elapsed
    for _ in range(3):
        dt = max(tau - elapsed, 0.0)
        px = min(max(r5["x"] + r5["vx"] * dt, 0.0), 120.0)
        py = min(max(r5["y"] + r5["vy"] * dt, 0.0), FIELD_W)
        rem = math.hypot(px - ball5[0], py - ball5[1])
        tau = elapsed + rem / v_ball
    return min(max(tau, elapsed + 0.1), 4.5), rem


# ── one play ─────────────────────────────────────────────────────────────────

def play_rows(g, p, pf, skip):
    ev = g.loc[g["event"].notna(), ["frameId", "event"]]
    th = ev.loc[ev["event"].isin(THROW), "frameId"]
    if th.empty:
        skip["no throw frame"] += 1
        return []
    t0 = int(th.min())
    sn = ev.loc[ev["event"].isin(SNAP), "frameId"]
    last = int(g["frameId"].max())
    t5 = min(t0 + 5, last)
    t3 = min(t0 + 3, t5)
    flip = g["playDirection"].iloc[0] == "left"
    los = p["absoluteYardlineNumber"] if not flip else 120.0 - p["absoluteYardlineNumber"]
    if pd.isna(los):
        skip["no line of scrimmage"] += 1
        return []
    by_frame = {f: g[g["frameId"] == f] for f in {t0, t3, t5}}

    def split(f):
        fr = by_frame[f]
        pl = fr[fr["nflId"].notna()]
        b = fr[fr["team"] == "football"]
        off = prep(pl[pl["team"] == p["possessionTeam"]], flip)
        dfn = prep(pl[pl["team"] == p["defensiveTeam"]], flip)
        ball = None
        if len(b):
            bx, by = float(b["x"].iloc[0]), float(b["y"].iloc[0])
            ball = (120.0 - bx, FIELD_W - by) if flip else (bx, by)
        return off, dfn, ball

    off0, def0, ball0 = split(t0)
    off5, def5, ball5 = split(t5)
    _, _, ball3 = split(t3)
    if ball0 is None or ball5 is None or ball3 is None or len(def0["id"]) < 8 or len(def5["id"]) < 8:
        skip["incomplete frame"] += 1
        return []

    tgt = off0["id"][off0["jersey"] == p["target_jersey"]]
    if len(tgt) != 1:
        skip["target not unique"] += 1
        return []
    target = int(tgt[0])
    qb_id = pf["qb"]
    if qb_id is None or qb_id not in off0["id"]:
        skip["no qb"] += 1
        return []

    elapsed = (t5 - t0) / 10.0
    span = (t5 - t3) / 10.0
    v_ball = float(np.clip(math.hypot(ball5[0] - ball3[0], ball5[1] - ball3[1]) / span, 10.0, 30.0)) if span > 0 else 18.0

    # quarterback and pass rush at the throw
    qi = int(np.where(off0["id"] == qb_id)[0][0])
    qx, qy = off0["x"][qi], off0["y"][qi]
    rush_mask = np.isin(def0["id"], list(pf["rush"]))
    qdist_all = np.hypot(def0["x"] - qx, def0["y"] - qy)
    if rush_mask.any():
        rd = qdist_all[rush_mask]
        j = int(rd.argmin())
        rv = take(def0, rush_mask)
        ux, uy = (rv["x"][j] - qx) / max(rd[j], 1e-6), (rv["y"][j] - qy) / max(rd[j], 1e-6)
        rusher_close = -((rv["vx"][j] - off0["vx"][qi]) * ux + (rv["vy"][j] - off0["vy"][qi]) * uy)
        rush_feats = {"rusher_dist": float(rd[j]), "n_rush5": int((rd < 5).sum()), "rusher_close": float(rusher_close)}
    else:
        rush_feats = {"rusher_dist": 15.0, "n_rush5": 0, "rusher_close": 0.0}
    qb_feats = {"qb_depth": float(los - qx), "qb_speed": float(off0["s"][qi]), "n_rush": int(rush_mask.sum()),
                "qb_nearest_def": float(qdist_all.min()), "time_to_throw": (t0 - int(sn.min())) / 10.0 if len(sn) else np.nan,
                **rush_feats}

    rows = []
    for rid in sorted((pf["routes"] | {target}) & set(off0["id"].tolist())):
        if rid == qb_id or rid not in off5["id"]:
            continue
        a0, a5 = int(np.where(off0["id"] == rid)[0][0]), int(np.where(off5["id"] == rid)[0][0])
        O0, O5 = take(off0, off0["id"] != rid), take(off5, off5["id"] != rid)
        r0 = (off0["x"][a0], off0["y"][a0], off0["vx"][a0], off0["vy"][a0])
        r5 = {"x": off5["x"][a5], "y": off5["y"][a5], "vx": off5["vx"][a5], "vy": off5["vy"][a5]}
        tau, rem = estimate_arrival(r5, ball5, v_ball, elapsed)
        dt = tau - elapsed
        D_p, O_p = advance(def5, dt), advance(O5, dt)
        rp = advance({k: np.array([v]) for k, v in r5.items()}, dt)
        g0 = geometry(*r0, def0, O0, los)
        g5 = geometry(r5["x"], r5["y"], r5["vx"], r5["vy"], def5, O5, los)
        gp = geometry(float(rp["x"][0]), float(rp["y"][0]), r5["vx"], r5["vy"], D_p, O_p, los)
        # decision-time projection: everything from t0 only, fixed ball speed, same for every receiver
        tau_d, dist_d = estimate_arrival_at_release(r0, ball0)
        rd = advance({"x": np.array([r0[0]]), "y": np.array([r0[1]]), "vx": np.array([r0[2]]), "vy": np.array([r0[3]])}, tau_d)
        gd = geometry(float(rd["x"][0]), float(rd["y"][0]), r0[2], r0[3], advance(def0, tau_d), advance(O0, tau_d), los)
        row = {"gameId": p["gameId"], "playId": p["playId"], "nflId": rid, "is_target": rid == target,
               "pos_lined_up": pf["pos"].get(rid),
               "throw_dist": math.hypot(r0[0] - ball0[0], r0[1] - ball0[1]), "lateral_from_qb": abs(r0[1] - qy),
               "v_ball": v_ball, "tau": tau, "rem_dist": rem, "tau_d": tau_d, "pass_dist_d": dist_d}
        for tag, gg in (("t0", g0), ("t5", g5), ("p", gp), ("d", gd)):
            row.update({f"{k}_{tag}": v for k, v in gg.items()})
        for tag, P, a in (("t0", off0, a0), ("t5", off5, a5)):
            row.update({f"spd_{tag}": P["s"][a], f"acc_{tag}": P["a"][a], f"vx_{tag}": P["vx"][a],
                        f"vy_{tag}": P["vy"][a], f"face_x_{tag}": P["face_x"][a]})
        row["sep1_delta"] = g5["sep1"] - g0["sep1"]
        row.update(qb_feats)
        row["pressured"] = int(pf["pressured"])
        rows.append(row)
    return rows


def build(max_games=None):
    t = load_targets(max_games)
    pff = pff_index()
    files = {int(re.search(r"tracking_(\d+)", f).group(1)): f for f in glob.glob(os.path.join(DATA, "tracking", "*.csv"))}
    cols = ["gameId", "playId", "nflId", "frameId", "jerseyNumber", "team", "playDirection", "x", "y", "s", "a", "o", "dir", "event"]
    skip = {"no throw frame": 0, "no line of scrimmage": 0, "incomplete frame": 0, "target not unique": 0, "no qb": 0, "no pff": 0}
    rows = []
    games = sorted(t["gameId"].unique())
    for i, gid in enumerate(games, 1):
        sub = t[t["gameId"] == gid]
        df = pd.read_csv(files[gid], usecols=cols)
        df = df[df["playId"].isin(sub["playId"])]
        by_play = {pid: d for pid, d in df.groupby("playId")}
        for p in sub.to_dict("records"):
            g, pf = by_play.get(p["playId"]), pff.get((gid, p["playId"]))
            if g is None or pf is None:
                skip["no pff"] += 1
                continue
            rows.extend(play_rows(g, p, pf, skip))
        if i % 20 == 0 or i == len(games):
            print(f"  processed {i}/{len(games)} games, {len(rows):,} receiver rows")
    f = pd.DataFrame(rows)

    ctx = ["gameId", "playId", "down", "yardsToGo", "quarter", "possessionTeam", "defensiveTeam", "pff_passCoverage",
           "pff_passCoverageType", "offenseFormation", "defendersInBox", "pff_playAction", "passResult", "yac",
           "air_yards", "nfl_xyac"]
    f = f.merge(t[ctx], on=KEYS, how="left")
    f["complete"] = np.where(f["is_target"], (f["passResult"] == "C").astype(float), np.nan)
    f.loc[~f["is_target"], ["yac", "air_yards", "nfl_xyac"]] = np.nan
    pl = pd.read_csv(os.path.join(DATA, "players.csv"))
    hh = pl["height"].str.extract(r"^(\d)-(\d{1,2})$").astype(float)
    pl["height_in"] = hh[0] * 12 + hh[1]
    pl["pos_group"] = pl["officialPosition"].map(POS_GROUP).fillna("other")
    f = f.merge(pl[["nflId", "weight", "height_in", "pos_group", "displayName"]], on="nflId", how="left")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "receiver_features.csv")
    f.to_csv(path, index=False)
    print(f"\nskipped plays: {skip}")
    plays_n = f[KEYS].drop_duplicates().shape[0]
    tg = f[f["is_target"]]
    print(f"saved {path}: {len(f):,} receiver rows on {plays_n:,} plays | targets: {len(tg):,} | "
          f"eligible receivers per play: mean {len(f) / plays_n:.1f}")
    return f


def check_projection(f):
    """The projected arrival depth can be checked against nflverse air_yards for targeted completions."""
    c = f[f["is_target"] & f["air_yards"].notna()].copy()
    print(f"\n=== Projection check on {len(c):,} targeted passes with nflverse air_yards ===")
    for name, col in (("receiver depth at throw (t0)", "x_rel_t0"), ("receiver depth 0.5 s later (t5)", "x_rel_t5"),
                      ("projected depth, from the actual ball path (p)", "x_rel_p"),
                      ("projected depth, release-time only (d)", "x_rel_d")):
        err = c[col] - c["air_yards"]
        print(f"  {name:40s} corr with air_yards {c[col].corr(c['air_yards']):.3f} | MAE {err.abs().mean():.2f} yd | bias {err.mean():+.2f}")
    print(f"  estimated time of flight: median {c['tau'].median():.2f} s, 10th-90th percentile "
          f"{c['tau'].quantile(.1):.2f}-{c['tau'].quantile(.9):.2f} s; ball speed median {c['v_ball'].median():.1f} yd/s")
    for lo, hi in ((-20, 0), (0, 5), (5, 10), (10, 20), (20, 60)):
        s = c[(c["air_yards"] >= lo) & (c["air_yards"] < hi)]
        if len(s) > 20:
            print(f"  air_yards {lo:>3d} to {hi:<3d} n={len(s):4d}: projected depth MAE {(s['x_rel_p'] - s['air_yards']).abs().mean():.2f}, "
                  f"t5 depth MAE {(s['x_rel_t5'] - s['air_yards']).abs().mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()
    t_start = time.time()
    table = build(args.max_games)
    check_projection(table)
    print(f"\nDone in {time.time() - t_start:.0f}s")
