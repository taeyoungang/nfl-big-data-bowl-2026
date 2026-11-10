"""
Data validation for the NFL Big Data Bowl dataset.

Three layers of evidence, written to validation/validation_report.md:
  1. Structure   - keys, joins, formats, ranges.
  2. Physics     - tracking values must be physically possible and internally consistent.
  3. Agreement   - files must agree with each other, and with an outside source
                   (nflverse play-by-play, downloaded to external/).

What this CANNOT prove: that the sensors measured the true position. Tracking accuracy can only
be tested for plausibility and consistency, not against ground truth.

Usage:
    python validate_data.py                 # full run
    python validate_data.py --max-games 3   # smoke test
"""

import argparse
import glob
import os
import re
import time
from collections import OrderedDict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TRACK = os.path.join(DATA, "tracking")
NV_PATH = os.path.join(ROOT, "external", "play_by_play_2021.parquet")
OUT = os.path.join(ROOT, "validation")
EXC_DIR = os.path.join(OUT, "exceptions")

HZ = 10.0
KEYS = ["gameId", "playId"]
SNAP = {"ball_snap", "autoevent_ballsnap"}
THROW = {"pass_forward", "autoevent_passforward"}
OFF_ROLES = {"Pass", "Pass Block", "Pass Route"}
DEF_ROLES = {"Coverage", "Pass Rush"}
GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
}

R = []            # check results
EXC = {}          # name -> DataFrame of offending rows
ACC = OrderedDict()  # tracking checks accumulated across files: name -> [bad, total, tol]
MAXV = {"speed": 0.0, "accel": 0.0, "step": 0.0, "ball_speed": 0.0}


def add(section, name, bad, total, tol=0.0, detail=""):
    bad, total = int(bad), int(total)
    if total == 0:
        status, res = "SKIP", "no data"
    else:
        frac = bad / total
        status = "PASS" if bad == 0 else ("WARN" if frac <= tol else "FAIL")
        res = f"{bad:,} of {total:,} ({frac:.2%}) violate"
    R.append(dict(section=section, check=name, status=status, result=res, detail=detail, tol=tol))


def info(section, name, detail):
    R.append(dict(section=section, check=name, status="INFO", result="", detail=detail, tol=None))


def acc(name, bad, total, tol=0.0):
    a = ACC.setdefault(name, [0, 0, tol])
    a[0] += int(bad)
    a[1] += int(total)


def keep(name, df, n=300):
    if df is not None and len(df):
        EXC[name] = df.head(n)


# ── tracking scan (one pass over every file) ────────────────────────────────

def scan_tracking(files):
    cols = ["gameId", "playId", "nflId", "frameId", "time", "jerseyNumber", "team", "playDirection",
            "x", "y", "s", "a", "dis", "o", "dir", "event"]
    members, snaps, pre, plays_info = [], [], [], []
    for i, f in enumerate(files, 1):
        fid = int(re.search(r"tracking_(\d+)", f).group(1))
        df = pd.read_csv(f, usecols=cols)
        df["pid"] = df["nflId"].fillna(-1).astype("int64")
        acc("file name gameId equals gameId column", (df["gameId"] != fid).sum(), len(df))

        # ── structure ──
        per_frame = df.groupby(["playId", "frameId"]).size()
        acc("every frame has exactly 23 rows (22 players + ball)", (per_frame != 23).sum(), len(per_frame))
        acc("no duplicate (play, object, frame) rows", df.duplicated(["playId", "pid", "frameId"]).sum(), len(df))
        gp = df.groupby(["playId", "pid"])["frameId"].agg(["count", "min", "max"])
        acc("frames per object run 1..N with no gaps", ((gp["min"] != 1) | (gp["max"] != gp["count"])).sum(), len(gp))
        acc("all objects in a play have the same frame count",
            (gp.groupby("playId")["count"].nunique() > 1).sum(), gp.index.get_level_values(0).nunique())
        acc("one playDirection per play", (df.groupby("playId")["playDirection"].nunique() != 1).sum(), df["playId"].nunique())
        m = df[["playId", "pid", "team", "jerseyNumber"]].drop_duplicates()
        acc("each object has one team and one jersey number per play", m.duplicated(["playId", "pid"]).sum(), len(m))
        mm = m.assign(gameId=fid)
        members.append(mm)

        # ── events ──
        ev = df.loc[df["event"].notna(), ["playId", "frameId", "event"]].drop_duplicates()
        evf = df[df["event"].notna()].groupby(["playId", "frameId"])["event"].agg(["count", "nunique"])
        acc("an event tag is written on all 23 rows of its frame", (evf["count"] != 23).sum(), len(evf))
        acc("at most one event type per frame", (evf["nunique"] > 1).sum(), len(evf))

        def first_frame(names):
            return ev[ev["event"].isin(names)].groupby("playId")["frameId"].min()

        pl = pd.DataFrame({"n_frames": df.groupby("playId")["frameId"].max()})
        pl["snap"] = first_frame(SNAP)
        pl["snap_a"] = first_frame({"ball_snap"})
        pl["snap_b"] = first_frame({"autoevent_ballsnap"})
        pl["throw"] = first_frame(THROW)
        pl["sack"] = first_frame({"qb_sack"})
        pl["direction"] = df.groupby("playId")["playDirection"].first()
        for e in ["play_action", "run", "qb_sack", "qb_strip_sack", "pass_forward", "autoevent_passforward", "pass_arrived"]:
            pl["ev_" + e] = ev[ev["event"] == e].groupby("playId").size().reindex(pl.index).fillna(0) > 0
        pl["gameId"] = fid
        plays_info.append(pl.reset_index())

        both = pl.dropna(subset=["snap", "throw"])
        acc("snap happens before pass_forward", (both["throw"] <= both["snap"]).sum(), len(both))

        # ── time stamps ──
        tt = df.groupby("playId")["time"].agg(["min", "max"])
        dur = (pd.to_datetime(tt["max"], utc=True) - pd.to_datetime(tt["min"], utc=True)).dt.total_seconds()
        exp = (pl["n_frames"] - 1) / HZ
        acc("play duration from timestamps matches frames / 10 Hz (+/-1 s)", ((dur - exp).abs() > 1.0).sum(), len(pl))
        ball = df[df["pid"] == -1].sort_values(["playId", "frameId"])
        tb = pd.to_datetime(ball["time"], utc=True)
        acc("timestamps never go backwards", (tb.groupby(ball["playId"]).diff().dt.total_seconds() < 0).sum(), len(ball))
        per_sec = ball.groupby(["playId", "time"]).size()
        acc("no more than 10 frames share one timestamp second", (per_sec > 10).sum(), len(per_sec))

        # ── physics (players only) ──
        d = df[df["pid"] > 0].sort_values(["playId", "pid", "frameId"]).copy()
        g = d.groupby(["playId", "pid"], sort=False)
        d["dx"], d["dy"] = g["x"].diff(), g["y"].diff()
        d["step"] = np.hypot(d["dx"], d["dy"])
        v = d["step"].notna()
        acc("x within 0..120 yd", ((d["x"] < 0) | (d["x"] > 120)).sum(), len(d), tol=0.001)
        acc("y within 0..53.3 yd", ((d["y"] < 0) | (d["y"] > 53.3)).sum(), len(d), tol=0.001)
        acc("x, y no more than 1 yd outside the field", ((d["x"] < -1) | (d["x"] > 121) | (d["y"] < -1) | (d["y"] > 54.3)).sum(), len(d))
        acc("dis equals the real position change between frames (+/-0.05 yd)", ((d["step"] - d["dis"]).abs() > 0.05)[v].sum(), v.sum())
        acc("speed s is close to 10 x dis (+/-1.5 yd/s)", ((d["s"] - 10 * d["dis"]).abs() > 1.5)[v].sum(), v.sum(), tol=0.01)
        acc("no teleporting: position change <= 1.5 yd per frame (15 yd/s)", (d["step"] > 1.5)[v].sum(), v.sum())
        acc("player speed <= 12 yd/s (about 27 mph)", (d["s"] > 12).sum(), len(d))
        acc("player acceleration <= 15 yd/s^2", (d["a"] > 15).sum(), len(d))
        acc("speed and acceleration are never negative", ((d["s"] < 0) | (d["a"] < 0)).sum(), len(d))
        acc("orientation o and direction dir within 0..360", (((d["o"] < 0) | (d["o"] > 360)) | ((d["dir"] < 0) | (d["dir"] > 360))).sum(), len(d))
        acc("player rows all have o and dir", (d["o"].isna() | d["dir"].isna()).sum(), len(d))
        mv = v & (d["step"] * HZ > 3) & d["dir"].notna()
        ang = np.degrees(np.arctan2(d["dx"], d["dy"])) % 360
        diff = ((ang - d["dir"] + 180) % 360 - 180).abs()
        acc("dir points along the direction of travel (+/-20 deg when moving > 3 yd/s)", (diff[mv] > 20).sum(), mv.sum(), tol=0.05)
        MAXV["speed"] = max(MAXV["speed"], float(d["s"].max()))
        MAXV["accel"] = max(MAXV["accel"], float(d["a"].max()))
        MAXV["step"] = max(MAXV["step"], float(d["step"].max()))
        MAXV["ball_speed"] = max(MAXV["ball_speed"], float(ball["s"].max()))

        # ── snap-frame rows ──
        sf = pl["snap"].dropna().rename("snap_frame").reset_index()
        if len(sf):
            s = df.merge(sf, on="playId")
            keep_cols = ["gameId", "playId", "pid", "team", "x", "y", "s", "o", "playDirection", "jerseyNumber"]
            snaps.append(s[s["frameId"] == s["snap_frame"]][keep_cols])
            # 0.3 s before the snap tag: the ball has not started to move yet
            pre.append(s[s["frameId"] == s["snap_frame"] - 3][keep_cols])
        if i % 20 == 0 or i == len(files):
            print(f"  scanned {i}/{len(files)} tracking files")
    return (pd.concat(members, ignore_index=True), pd.concat(snaps, ignore_index=True),
            pd.concat(pre, ignore_index=True), pd.concat(plays_info, ignore_index=True))


# ── sections ─────────────────────────────────────────────────────────────────

def sec_keys(games, plays, players, pff, file_ids):
    S = "1. Keys, joins and formats"
    add(S, "games.gameId is unique", games["gameId"].duplicated().sum(), len(games))
    add(S, "plays (gameId, playId) is unique", plays.duplicated(KEYS).sum(), len(plays))
    add(S, "players.nflId is unique", players["nflId"].duplicated().sum(), len(players))
    add(S, "pff (gameId, playId, nflId) is unique", pff.duplicated(KEYS + ["nflId"]).sum(), len(pff))
    add(S, "every play belongs to a game in games.csv", (~plays["gameId"].isin(games["gameId"])).sum(), len(plays))
    add(S, "every game has a tracking file", (~games["gameId"].isin(file_ids)).sum(), len(games))
    add(S, "every tracking file belongs to a game in games.csv", (~pd.Series(sorted(file_ids)).isin(games["gameId"])).sum(), len(file_ids))
    pk = plays[KEYS].drop_duplicates()
    add(S, "every PFF row belongs to a play in plays.csv",
        (~pff.merge(pk, on=KEYS, how="left", indicator=True)["_merge"].eq("both")).sum(), len(pff))
    add(S, "every PFF nflId is in players.csv", (~pff["nflId"].isin(players["nflId"])).sum(), len(pff))
    m = plays.merge(games, on="gameId")
    ok = ((m["possessionTeam"] == m["homeTeamAbbr"]) & (m["defensiveTeam"] == m["visitorTeamAbbr"])) | \
         ((m["possessionTeam"] == m["visitorTeamAbbr"]) & (m["defensiveTeam"] == m["homeTeamAbbr"]))
    add(S, "offense and defense are the two teams of that game", (~ok).sum(), len(m))
    gd = pd.to_datetime(games["gameDate"], format="%m/%d/%Y", errors="coerce")
    wk = ((gd - pd.Timestamp("2021-09-09")).dt.days // 7) + 1
    add(S, "games.week matches the game date (NFL weeks start on Thursday)", (wk != games["week"]).sum(), len(games))
    add(S, "gameId starts with the game date (YYYYMMDD)", (games["gameId"].astype(str).str[:8] != gd.dt.strftime("%Y%m%d")).sum(), len(games))
    add(S, "all games are season 2021", (games["season"] != 2021).sum(), len(games))
    keep("games_week_mismatch", games[wk != games["week"]])


def sec_plays(plays, pi, players, mm):
    S = "2. plays.csv: internal logic"
    p = plays.merge(pi[KEYS + ["direction"]], on=KEYS, how="left")
    has_dir = p["direction"].notna()
    p["right"] = p["direction"] == "right"

    add(S, "down is 1-4 (0 only for two-point tries)", (~p["down"].between(0, 4)).sum(), len(p))
    two = p["playDescription"].str.contains("TWO-POINT", case=False, na=False)
    add(S, "down = 0 exactly on two-point tries", ((p["down"] == 0) != two).sum(), len(p))
    add(S, "quarter is 1-5", (~p["quarter"].between(1, 5)).sum(), len(p))
    clk = p["gameClock"].str.extract(r"^(\d{1,2}):(\d{2})$").astype(float)
    secs = clk[0] * 60 + clk[1]
    add(S, "gameClock is MM:SS between 0:00 and 15:00", (secs.isna() | (secs > 900)).sum(), len(p))
    add(S, "yardsToGo is between 0 and 50", (~p["yardsToGo"].between(0, 50)).sum(), len(p))
    add(S, "defendersInBox is between 1 and 11 when present", (~p["defendersInBox"].dropna().between(1, 11)).sum(), p["defendersInBox"].notna().sum())

    # playId is an identifier, not a timestamp, so test it before relying on it for ordering
    q = p.assign(secs=secs)
    byid = q.sort_values(KEYS).groupby("gameId")
    out_of_order = (byid["quarter"].diff() < 0) | ((byid["quarter"].diff() == 0) & (byid["secs"].diff() > 0))
    info(S, "playId order vs game clock",
         f"{int(out_of_order.sum()):,} of {len(q):,} plays have a playId that is not in clock order, so playId is not a clock. "
         "The score checks below sort by quarter and game clock instead.")
    q = q.sort_values(["gameId", "quarter", "secs", "playId"], ascending=[True, True, False, True])
    gq = q.groupby("gameId")
    add(S, "home score never decreases (plays sorted by quarter and clock)", (gq["preSnapHomeScore"].diff() < 0).sum(), len(q))
    add(S, "visitor score never decreases (plays sorted by quarter and clock)", (gq["preSnapVisitorScore"].diff() < 0).sum(), len(q))
    jump = pd.concat([gq["preSnapHomeScore"].diff(), gq["preSnapVisitorScore"].diff()]).max()
    info(S, "largest score change between consecutive rows",
         f"{jump:.0f} points. Not a defect: plays.csv holds dropbacks only, so consecutive rows can span several scoring drives. "
         "Section 6 checks every row's pre-snap score directly against nflverse.")

    # line of scrimmage: yardline fields vs absoluteYardlineNumber and the tracking direction
    n = p["yardlineNumber"]
    own, opp = p["yardlineSide"] == p["possessionTeam"], p["yardlineSide"] == p["defensiveTeam"]
    expect = np.where(own, np.where(p["right"], 10 + n, 110 - n),
                      np.where(opp, np.where(p["right"], 110 - n, 10 + n), np.where(n == 50, 60, np.nan)))
    chk = has_dir & p["absoluteYardlineNumber"].notna() & ~np.isnan(expect)
    bad = chk & ((expect - p["absoluteYardlineNumber"]).abs() > 0.5)
    add(S, "absoluteYardlineNumber agrees with yardlineSide/Number and tracked play direction", bad.sum(), chk.sum())
    keep("yardline_mismatch", p[bad][KEYS + ["possessionTeam", "yardlineSide", "yardlineNumber", "absoluteYardlineNumber", "direction"]])
    blank = p["yardlineSide"].isna()
    add(S, "yardlineSide is blank only at midfield (yard line 50)", (blank & (n != 50)).sum(), blank.sum())
    goal = np.where(p["right"], 110 - p["absoluteYardlineNumber"], p["absoluteYardlineNumber"] - 10)
    c2 = has_dir & p["absoluteYardlineNumber"].notna()
    bad = c2 & (p["yardsToGo"] > goal + 0.5)
    add(S, "yardsToGo never exceeds the distance to the goal line", bad.sum(), c2.sum())
    keep("ytg_exceeds_goal", p[bad][KEYS + ["down", "yardsToGo", "absoluteYardlineNumber", "direction"]])
    p["yardline_100"] = goal

    # result fields
    d = p["playDescription"].str.lower()
    nopen = p["penaltyYards"].isna()
    fum = d.str.contains("fumble")
    sel = nopen & ~fum
    bad = (p["playResult"] != p["prePenaltyPlayResult"]) & sel
    add(S, "no penalty -> playResult equals prePenaltyPlayResult (fumble plays excluded)", bad.sum(), sel.sum())
    keep("no_penalty_result_differs", p[bad][KEYS + ["prePenaltyPlayResult", "playResult", "playDescription"]])
    pen = ~nopen
    rule = pd.Series(np.select(
        [p["playResult"] == p["penaltyYards"], p["playResult"] == p["prePenaltyPlayResult"],
         p["playResult"] == p["prePenaltyPlayResult"] + p["penaltyYards"]],
        ["penalty yards replace the play", "play yards kept (penalty declined or offsetting)", "play yards + penalty yards"],
        "other"), index=p.index)
    info(S, "how playResult is set on plays with penaltyYards",
         "; ".join(f"{k}: {v:,}" for k, v in rule[pen].value_counts().items()) +
         ". When a penalty cancels the play (No Play), the penalty yards become the result.")
    bad = pen & (rule == "other") & p["foulName2"].isna()
    add(S, "with one penalty, playResult is the penalty yards, the play yards, or their sum", bad.sum(), pen.sum(), tol=0.01)
    keep("penalty_result_unexplained", p[bad][KEYS + ["prePenaltyPlayResult", "penaltyYards", "playResult", "foulName1", "playDescription"]])
    has_foul = p["foulName1"].notna()
    add(S, "a named foul implies penaltyYards is present", (has_foul & nopen).sum(), has_foul.sum())
    info(S, "penaltyYards without a foul name", f"{int((pen & ~has_foul).sum()):,} plays (team penalties such as Illegal Formation have no player to name).")

    # passResult vs play description
    exp_s, exp_in, exp_i, exp_r = d.str.contains("sacked"), d.str.contains("intercepted"), d.str.contains("incomplete"), d.str.contains("scramble")
    for code, cond, label in (("S", exp_s, "sacked"), ("IN", exp_in, "intercepted")):
        sel = p["passResult"] == code
        add(S, f"passResult {code} <-> description says '{label}'", ((sel != cond)).sum(), len(p), tol=0.01)
    sel = p["passResult"] == "I"
    add(S, "passResult I -> description says 'incomplete'", (sel & ~exp_i).sum(), sel.sum(), tol=0.05)
    keep("passresult_I_not_incomplete", p[sel & ~exp_i][KEYS + ["passResult", "playDescription"]])
    sel = p["passResult"] == "C"
    add(S, "passResult C -> description is not incomplete/intercepted/sacked", (sel & (exp_i | exp_in | exp_s)).sum(), sel.sum(), tol=0.01)
    sel = p["passResult"] == "R"
    add(S, "passResult R -> description says 'scramble'", (sel & ~exp_r).sum(), sel.sum(), tol=0.05)

    # yards in description vs prePenaltyPlayResult
    def yards(txt):
        m = re.search(r"\bfor (-?\d+) yards?\b", txt)
        if m:
            return int(m.group(1))
        if re.search(r"\bfor no gain\b", txt):
            return 0
        m = re.search(r"\bfor a loss of (\d+) yards?\b", txt)
        return -int(m.group(1)) if m else np.nan
    dy = p["playDescription"].map(yards)
    clean = p["passResult"].isin(["C", "S", "R"]) & dy.notna() & ~d.str.contains("penalty|fumble|lateral")
    bad = clean & (dy != p["prePenaltyPlayResult"])
    add(S, "yards written in the description equal prePenaltyPlayResult (C/S/R plays without penalty/fumble)", bad.sum(), clean.sum(), tol=0.01)
    keep("description_yards_mismatch", p[bad][KEYS + ["passResult", "prePenaltyPlayResult", "playDescription"]])

    # coverage labels
    man, zone = {"Cover-0", "Cover-1", "2-Man"}, {"Cover-2", "Cover-3", "Quarters", "Cover-6"}
    sel = p["pff_passCoverage"].isin(man | zone)
    exp_t = p["pff_passCoverage"].map(lambda c: "Man" if c in man else "Zone")
    add(S, "coverage type matches coverage name (Cover-0/1/2-Man = Man; Cover-2/3/4/6 = Zone)", (sel & (exp_t != p["pff_passCoverageType"])).sum(), sel.sum(), tol=0.01)

    # personnel strings vs the players actually on the field
    def parse(s):
        return {k: int(v) for v, k in re.findall(r"(\d+) (QB|RB|TE|WR|OL|DL|LB|DB)", str(s))}
    po = p["personnelO"].map(parse)
    tot_o = po.map(lambda c: sum(c.values()) + (5 if "OL" not in c else 0) + (1 if "QB" not in c else 0))
    add(S, "offensive personnel adds up to 11 (QB and 5 OL implied)", (tot_o != 11).sum(), len(p), tol=0.01)
    pd_ = p["personnelD"].map(parse)
    add(S, "defensive personnel adds up to 11", (pd_.map(lambda c: sum(c.values())) != 11).sum(), len(p), tol=0.01)

    pos = players.set_index("nflId")["officialPosition"].map(GROUP)
    on = mm[mm["pid"] > 0].copy()
    on["grp"] = on["pid"].map(pos)
    cnt = on.groupby(KEYS + ["role"])["grp"].value_counts().unstack(fill_value=0).reset_index()
    off = cnt[cnt["role"] == "off"].set_index(KEYS)
    dfn = cnt[cnt["role"] == "def"].set_index(KEYS)
    j = p.set_index(KEYS)
    a = j.index.intersection(off.index)
    po_i, pd_i = po.set_axis(j.index), pd_.set_axis(j.index)
    mism_o = sum(any(po_i.loc[k].get(g, 0) != off.loc[k].get(g, 0) for g in ("RB", "TE", "WR")) for k in a)
    add(S, "personnelO (RB/TE/WR counts) matches the positions of the 11 offensive players", mism_o, len(a), tol=0.10,
        detail="Compared with officialPosition in players.csv. Small gaps are expected because personnel labels follow the play, not the roster.")
    b = j.index.intersection(dfn.index)
    mism_d = sum(any(pd_i.loc[k].get(g, 0) != dfn.loc[k].get(g, 0) for g in ("DL", "LB", "DB")) for k in b)
    add(S, "personnelD (DL/LB/DB counts) matches the positions of the 11 defensive players", mism_d, len(b), tol=0.10)

    # play action flag vs tracking event
    pa = p.merge(pi[KEYS + ["ev_play_action"]], on=KEYS, how="left")
    pa = pa[pa["ev_play_action"].notna()]
    agree = ((pa["pff_playAction"] == 1) == pa["ev_play_action"]).mean()
    info(S, "pff_playAction vs 'play_action' tracking event",
         f"agree on {agree:.1%} of plays. PFF flags {int(pa['pff_playAction'].sum()):,} play-action plays, "
         f"tracking tags {int(pa['ev_play_action'].sum()):,}. Two different labelers, so some disagreement is expected.")
    return p


def sec_players(players, mm):
    S = "3. players.csv"
    hh = players["height"].str.extract(r"^(\d)-(\d{1,2})$").astype(float)
    inches = hh[0] * 12 + hh[1]
    add(S, "height is formatted F-I", inches.isna().sum(), len(players))
    add(S, "height is between 5'0\" and 7'0\"", (~inches.dropna().between(60, 84)).sum(), inches.notna().sum())
    add(S, "weight is between 150 and 400 lb", (~players["weight"].between(150, 400)).sum(), len(players))
    bd = pd.to_datetime(players["birthDate"], errors="coerce")
    age = (pd.Timestamp("2021-09-09") - bd).dt.days / 365.25
    add(S, "age on 2021-09-09 is between 20 and 45", (~age.dropna().between(20, 45)).sum(), age.notna().sum())
    info(S, "missing values", f"birthDate missing for {bd.isna().sum()} players ({bd.isna().mean():.1%}), collegeName missing for {players['collegeName'].isna().sum()}.")
    g = players["officialPosition"].map(GROUP)
    add(S, "officialPosition is a known position", g.isna().sum(), len(players))
    rules = {"QB": (170, 280), "OL": (250, 400), "DB": (150, 260), "WR": (150, 260), "TE": (210, 300), "RB": (160, 280)}
    bad = sum(((players["weight"] < lo) | (players["weight"] > hi))[g == k].sum() for k, (lo, hi) in rules.items())
    tot = sum((g == k).sum() for k in rules)
    add(S, "weights are plausible for the position (QB, OL, DB, WR, TE, RB)", bad, tot, tol=0.01)
    dup = players[players["displayName"].duplicated(keep=False)]
    info(S, "duplicate display names", f"{dup['displayName'].nunique()} names are shared by more than one nflId (different people with the same name are legitimate).")
    if mm is not None:
        add(S, "every player in players.csv appears in some tracking file", (~players["nflId"].isin(mm.loc[mm["pid"] > 0, "pid"])).sum(), len(players))
        add(S, "every tracked nflId is in players.csv", (~mm.loc[mm["pid"] > 0, "pid"].isin(players["nflId"])).sum(), (mm["pid"] > 0).sum())


def sec_pff(plays, pff, players, mm, snaps, pi):
    S = "4. PFF scouting"
    roles = pff.groupby(KEYS)["pff_role"].value_counts().unstack(fill_value=0)
    add(S, "exactly one 'Pass' (QB) row per play", (roles["Pass"] != 1).sum(), len(roles))
    n_off = roles[["Pass", "Pass Block", "Pass Route"]].sum(axis=1)
    n_def = roles[["Coverage", "Pass Rush"]].sum(axis=1)
    add(S, "11 offensive roles per play", (n_off != 11).sum(), len(roles))
    add(S, "11 defensive roles per play", (n_def != 11).sum(), len(roles))
    qb = pff[pff["pff_role"] == "Pass"].merge(players[["nflId", "officialPosition"]], on="nflId", how="left")
    add(S, "the 'Pass' player is listed as a QB in players.csv", (qb["officialPosition"] != "QB").sum(), len(qb), tol=0.005)
    for c in ["pff_hit", "pff_hurry", "pff_sack", "pff_beatenByDefender", "pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed", "pff_backFieldBlock"]:
        vals = pff[c].dropna()
        add(S, f"{c} contains only 0 or 1", (~vals.isin([0, 1])).sum(), len(vals))
    d_flags = pff[["pff_hit", "pff_hurry", "pff_sack"]].notna().any(axis=1)
    add(S, "hit/hurry/sack credited only to defensive roles", (d_flags & ~pff["pff_role"].isin(DEF_ROLES)).sum(), d_flags.sum())
    o_flags = pff[["pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed"]].notna().any(axis=1)
    add(S, "'allowed' flags set only for offensive roles", (o_flags & ~pff["pff_role"].isin(OFF_ROLES)).sum(), o_flags.sum())

    ps = pff.groupby(KEYS).agg(sacks=("pff_sack", "sum"), sack_allowed=("pff_sackAllowed", "sum")).reset_index()
    j = plays.merge(ps, on=KEYS, how="left")
    s = j["passResult"] == "S"
    add(S, "plays ending in a sack have a defender credited with a sack", (s & (j["sacks"] < 1)).sum(), s.sum(), tol=0.05)
    keep("sack_without_pff_credit", j[s & (j["sacks"] < 1)][KEYS + ["passResult", "sacks", "playDescription"]])
    add(S, "a PFF-credited sack means the dropback ended in a sack", ((j["sacks"] >= 1) & ~s).sum(), (j["sacks"] >= 1).sum(), tol=0.05)
    add(S, "defender sack credit and blocker 'sack allowed' appear together", ((j["sacks"] >= 1) != (j["sack_allowed"] >= 1)).sum(), len(j), tol=0.05)

    bl = pff[pff["pff_nflIdBlockedPlayer"].notna()]
    dkeys = pff[pff["pff_role"].isin(DEF_ROLES)][KEYS + ["nflId"]].rename(columns={"nflId": "pff_nflIdBlockedPlayer"})
    dkeys["ok"] = True
    chk = bl.merge(dkeys, on=KEYS + ["pff_nflIdBlockedPlayer"], how="left")
    add(S, "the 'blocked player' is a defender on that play", chk["ok"].isna().sum(), len(chk), tol=0.01)

    # PFF vs tracking
    tk = mm[mm["pid"] > 0][KEYS + ["pid", "team", "role"]].rename(columns={"pid": "nflId"})
    o = pff[KEYS + ["nflId", "pff_role", "pff_positionLinedUp"]].merge(tk, on=KEYS + ["nflId"], how="outer", indicator=True)
    add(S, "every PFF row has a matching tracked player", (o["_merge"] == "left_only").sum(), len(pff))
    add(S, "every tracked player has a PFF row", (o["_merge"] == "right_only").sum(), len(tk))
    b = o[o["_merge"] == "both"]
    bad = (b["pff_role"].isin(OFF_ROLES) & (b["role"] != "off")) | (b["pff_role"].isin(DEF_ROLES) & (b["role"] != "def"))
    add(S, "PFF offense/defense role agrees with the tracked player's team", bad.sum(), len(b))

    # offensive line order at the snap, using playDirection
    sn = snaps.merge(pff[KEYS + ["nflId", "pff_positionLinedUp"]], left_on=KEYS + ["pid"], right_on=KEYS + ["nflId"])
    ol = sn[sn["pff_positionLinedUp"].isin(["LT", "LG", "C", "RG", "RT"])]
    piv = ol.pivot_table(index=KEYS, columns="pff_positionLinedUp", values="y", aggfunc="first").dropna()
    piv = piv.join(pi.set_index(KEYS)["direction"])
    sign = np.where(piv["direction"] == "right", 1.0, -1.0)
    y = piv[["LT", "LG", "C", "RG", "RT"]].mul(sign, axis=0)
    ordered = (y["LT"] > y["LG"]) & (y["LG"] > y["C"]) & (y["C"] > y["RG"]) & (y["RG"] > y["RT"])
    rev = (y["LT"] < y["LG"]) & (y["LG"] < y["C"]) & (y["C"] < y["RG"]) & (y["RG"] < y["RT"])
    add(S, "just before the snap, LT-LG-C-RG-RT are lined up in the right left-to-right order for the tracked direction",
        (~ordered).sum(), len(piv), tol=0.03,
        detail=f"Strictly in expected order on {ordered.mean():.1%} of plays; in the mirror-image order on {rev.mean():.1%}. "
               "This confirms three things at once: PFF alignment labels, tracking coordinates, and playDirection.")
    keep("ol_order_violations", piv[~ordered].reset_index().head(300))

    # the ball must be at the centre at the snap
    ball = snaps[snaps["pid"] == -1].set_index(KEYS)[["x", "y"]].rename(columns={"x": "bx", "y": "by"})
    c = ol[ol["pff_positionLinedUp"] == "C"].set_index(KEYS)[["x", "y"]].join(ball, how="inner")
    dist = np.hypot(c["x"] - c["bx"], c["y"] - c["by"])
    add(S, "ball is within 1.5 yd of the center just before the snap", (dist > 1.5).sum(), len(c), tol=0.03,
        detail=f"median distance {dist.median():.2f} yd.")
    keep("ball_far_from_center_at_snap", c[dist > 1.5].reset_index())


def sec_tracking_vs_plays(plays, pi, mm, snaps, p):
    S = "5. Tracking vs plays.csv"
    pk = plays[KEYS].drop_duplicates()
    add(S, "every play in plays.csv has tracking", (~pk.merge(pi[KEYS], on=KEYS, how="left", indicator=True)["_merge"].eq("both")).sum(), len(pk), tol=0.001)
    miss = pk.merge(pi[KEYS], on=KEYS, how="left", indicator=True)
    keep("plays_without_tracking", miss[miss["_merge"] == "left_only"][KEYS])
    add(S, "every tracked play is in plays.csv", (~pi[KEYS].merge(pk, on=KEYS, how="left", indicator=True)["_merge"].eq("both")).sum(), len(pi))
    add(S, "every play has a snap event", pi["snap"].isna().sum(), len(pi), tol=0.01)
    keep("tracked_plays_without_snap_event", pi[pi["snap"].isna()][KEYS + ["n_frames"]])
    b = pi.dropna(subset=["snap_a", "snap_b"])
    info(S, "ball_snap vs autoevent_ballsnap", f"both tags exist on {len(b):,} plays; frame difference median {(b['snap_a'] - b['snap_b']).median():.0f}, "
         f"{((b['snap_a'] - b['snap_b']).abs() > 2).sum():,} differ by more than 2 frames.")

    # teams: 11 offense, 11 defense, 1 ball
    cnt = mm.groupby(KEYS + ["role"]).size().unstack(fill_value=0)
    ok = (cnt.get("off", 0) == 11) & (cnt.get("def", 0) == 11) & (cnt.get("ball", 0) == 1) & (cnt.get("other", 0) == 0)
    add(S, "each play has 11 offense, 11 defense, 1 ball, and no unknown team code", (~ok).sum(), len(cnt))
    add(S, "no two players on one team share a jersey number in a play",
        mm[mm["pid"] > 0].duplicated(KEYS + ["team", "jerseyNumber"]).sum(), (mm["pid"] > 0).sum(), tol=0.001)

    # event vs passResult
    j = plays.merge(pi, on=KEYS, how="inner")
    thrown = j["throw"].notna()
    cia = j["passResult"].isin(["C", "I", "IN"])
    add(S, "completed/incomplete/intercepted passes have a pass_forward event", (cia & ~thrown).sum(), cia.sum(), tol=0.01)
    add(S, "sacks have no pass_forward event", ((j["passResult"] == "S") & thrown).sum(), (j["passResult"] == "S").sum(), tol=0.01)
    keep("sack_with_pass_forward_event", j[(j["passResult"] == "S") & thrown][KEYS + ["throw", "sack", "playDescription"]])
    sk = j["passResult"] == "S"
    add(S, "sacks have a qb_sack or qb_strip_sack event", (sk & ~(j["ev_qb_sack"] | j["ev_qb_strip_sack"])).sum(), sk.sum(), tol=0.2)
    add(S, "qb_sack events occur only on plays whose passResult is S", ((j["ev_qb_sack"]) & ~sk).sum(), j["ev_qb_sack"].sum(), tol=0.05)

    # direction and line of scrimmage at the snap
    s = snaps.merge(plays[KEYS + ["possessionTeam", "absoluteYardlineNumber"]], on=KEYS)
    sg = np.where(s["playDirection"] == "right", 1.0, -1.0)
    s["x_rel"] = (s["x"] - s["absoluteYardlineNumber"]) * sg
    bl = s[s["pid"] == -1].dropna(subset=["x_rel"])
    add(S, "ball is within 1.5 yd of the line of scrimmage just before the snap", (bl["x_rel"].abs() > 1.5).sum(), len(bl), tol=0.02,
        detail=f"median offset {bl['x_rel'].median():.2f} yd, 95th percentile |offset| {bl['x_rel'].abs().quantile(0.95):.2f} yd.")
    keep("ball_off_los_before_snap", bl[bl["x_rel"].abs() > 1.5][KEYS + ["x_rel"]])
    pl_ = s[s["pid"] > 0].copy()
    pl_["is_off"] = pl_["team"] == pl_["possessionTeam"]
    mo = pl_[pl_["is_off"]].groupby(KEYS)["x_rel"].mean()
    md = pl_[~pl_["is_off"]].groupby(KEYS)["x_rel"].mean()
    both = pd.concat([mo.rename("o"), md.rename("d")], axis=1).dropna()
    add(S, "offense is on the backfield side of the ball and defense on the other side just before the snap (confirms playDirection)", (both["o"] >= both["d"]).sum(), len(both))
    off_ahead = pl_[pl_["is_off"] & (pl_["x_rel"] > 2)].groupby(KEYS).size()
    def_back = pl_[~pl_["is_off"] & (pl_["x_rel"] < -2)].groupby(KEYS).size()
    add(S, "no offensive player is more than 2 yd past the line of scrimmage just before the snap", len(off_ahead), len(both), tol=0.03,
        detail="Whole-yard rounding of the line of scrimmage, motion men and tracking noise can cause a few.")
    add(S, "no defender is more than 2 yd behind the line of scrimmage just before the snap", len(def_back), len(both), tol=0.03)
    keep("offense_past_los_at_snap", off_ahead.reset_index().rename(columns={0: "n_players"}))


def sec_external(plays, games, p):
    S = "6. Cross-check against nflverse play-by-play (independent source)"
    if not os.path.exists(NV_PATH):
        info(S, "nflverse file not found", f"Expected {NV_PATH}. Skipped.")
        return
    cols = ["old_game_id", "play_id", "week", "season_type", "home_team", "away_team", "game_date", "qtr", "down", "ydstogo",
            "posteam", "defteam", "yardline_100", "time", "posteam_score", "defteam_score", "complete_pass",
            "incomplete_pass", "interception", "sack", "qb_scramble", "qb_dropback", "yards_gained", "penalty", "play_type", "desc",
            "air_yards"]
    nv = pd.read_parquet(NV_PATH, columns=cols)
    nv["gameId"] = pd.to_numeric(nv["old_game_id"], errors="coerce")
    nv = nv.dropna(subset=["gameId", "play_id"])
    nv["gameId"], nv["playId"] = nv["gameId"].astype("int64"), nv["play_id"].astype("int64")
    nvg = nv[nv["gameId"].isin(games["gameId"])]

    gm = nvg.groupby("gameId").agg(week=("week", "first"), home=("home_team", "first"), away=("away_team", "first"), date=("game_date", "first")).reset_index()
    g = games.merge(gm, on="gameId", how="left", suffixes=("", "_nv"))
    add(S, "every game exists in nflverse", g["week_nv"].isna().sum(), len(g))
    g = g.dropna(subset=["week_nv"])
    add(S, "week matches nflverse", (g["week"] != g["week_nv"]).sum(), len(g))
    add(S, "home team matches nflverse", (g["homeTeamAbbr"] != g["home"]).sum(), len(g))
    add(S, "visitor team matches nflverse", (g["visitorTeamAbbr"] != g["away"]).sum(), len(g))
    gdt = pd.to_datetime(g["gameDate"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    add(S, "game date matches nflverse", (gdt != g["date"].astype(str).str[:10]).sum(), len(g))

    # prefix every nflverse column so nothing collides with plays.csv columns
    nvp = nvg.drop(columns=["old_game_id", "play_id"]).rename(columns={c: "nv_" + c for c in nvg.columns if c not in KEYS})
    m = plays.merge(p[KEYS + ["yardline_100"]], on=KEYS, how="left").merge(nvp, on=KEYS, how="left", indicator=True)
    add(S, "every play in plays.csv exists in nflverse (same gameId and playId)", (m["_merge"] == "left_only").sum(), len(m))
    keep("plays_not_in_nflverse", m[m["_merge"] == "left_only"][KEYS + ["playDescription"]])
    m = m[m["_merge"] == "both"].copy()
    n = len(m)
    # nflverse writes jersey numbers in front of names ("12-T.Brady"); BDB does not
    norm = lambda s: (s.astype(str).str.replace(r"\b\d{1,3}-(?=[A-Za-z])", "", regex=True)
                      .str.replace(r"\s+", " ", regex=True).str.strip().str.lower())
    bad = norm(m["playDescription"]).str[:60] != norm(m["nv_desc"]).str[:60]
    add(S, "description text matches nflverse (first 60 characters, jersey numbers removed)", bad.sum(), n, tol=0.05)
    keep("description_mismatch", m[bad][KEYS + ["playDescription", "nv_desc"]])
    add(S, "quarter matches", (m["quarter"] != m["nv_qtr"]).sum(), n)
    sel = (m["down"] > 0) & m["nv_down"].notna()
    add(S, "down matches", (m["down"] != m["nv_down"])[sel].sum(), sel.sum())
    sel = m["nv_ydstogo"].notna()
    add(S, "yards to go matches", (m["yardsToGo"] != m["nv_ydstogo"])[sel].sum(), sel.sum())
    sel = m["nv_posteam"].notna()
    bad = (m["possessionTeam"] != m["nv_posteam"]) & sel
    add(S, "possession team matches", bad.sum(), sel.sum())
    keep("possession_team_mismatch", m[bad][KEYS + ["possessionTeam", "nv_posteam", "defensiveTeam", "nv_defteam"]])
    sel = m["nv_defteam"].notna()
    add(S, "defensive team matches", ((m["defensiveTeam"] != m["nv_defteam"]) & sel).sum(), sel.sum())
    sel = m["nv_yardline_100"].notna() & m["yardline_100"].notna()
    bad = ((m["yardline_100"] - m["nv_yardline_100"]).abs() > 0.5) & sel
    add(S, "distance to goal line (from tracking direction + absoluteYardlineNumber) matches nflverse yardline_100", bad.sum(), sel.sum(), tol=0.01,
        detail="An independent check on tracking coordinates, direction, and the line of scrimmage.")
    keep("yardline_100_mismatch", m[bad][KEYS + ["yardline_100", "nv_yardline_100", "absoluteYardlineNumber"]])
    secs = lambda s: s.astype(str).str.extract(r"^(\d+):(\d+)$").astype(float).pipe(lambda x: x[0] * 60 + x[1])
    sel = m["nv_time"].notna()
    add(S, "game clock matches (seconds)", ((secs(m["gameClock"]) - secs(m["nv_time"])).abs() > 1)[sel].sum(), sel.sum(), tol=0.01)
    # nflverse posteam_score / defteam_score are the scores BEFORE the play
    m["nv_home"] = np.where(m["nv_posteam"] == m["nv_home_team"], m["nv_posteam_score"], m["nv_defteam_score"])
    m["nv_away"] = np.where(m["nv_posteam"] == m["nv_home_team"], m["nv_defteam_score"], m["nv_posteam_score"])
    sel = m["nv_posteam_score"].notna() & m["nv_defteam_score"].notna()
    hs = ((m["preSnapHomeScore"] != m["nv_home"]) | (m["preSnapVisitorScore"] != m["nv_away"])) & sel
    add(S, "pre-snap home and visitor scores match nflverse", hs.sum(), sel.sum(), tol=0.01)
    keep("score_mismatch", m[hs][KEYS + ["preSnapHomeScore", "nv_home", "preSnapVisitorScore", "nv_away"]])

    pen_play = m["nv_penalty"] == 1
    info(S, "plays with an accepted penalty", f"{int(pen_play.sum()):,} plays. nflverse leaves complete/incomplete unset when a penalty cancels the play, "
         "while plays.csv still records what the pass did, so those plays are left out of the pass-outcome comparison below.")
    mp = m[~pen_play]
    for code, col in (("C", "nv_complete_pass"), ("I", "nv_incomplete_pass"), ("IN", "nv_interception"), ("S", "nv_sack"), ("R", "nv_qb_scramble")):
        bad = (mp["passResult"] == code) != (mp[col] == 1)
        add(S, f"passResult {code} <-> nflverse {col[3:]} (plays without a penalty)", bad.sum(), len(mp), tol=0.02)
        keep(f"passresult_{code}_mismatch", mp[bad][KEYS + ["passResult", "nv_complete_pass", "nv_incomplete_pass", "nv_sack", "nv_qb_scramble", "playDescription"]])
    sel = (m["nv_penalty"] == 0) & m["nv_yards_gained"].notna()
    bad = (m["prePenaltyPlayResult"] != m["nv_yards_gained"]) & sel
    add(S, "yards gained (plays without a penalty) match nflverse", bad.sum(), sel.sum(), tol=0.01)
    keep("yards_mismatch", m[bad][KEYS + ["prePenaltyPlayResult", "nv_yards_gained", "playDescription"]])

    # completeness: dropbacks nflverse has but plays.csv doesn't
    nd = nvg[(nvg["qb_dropback"] == 1) & (nvg["season_type"] == "REG")]
    miss = nd.merge(plays[KEYS], on=KEYS, how="left", indicator=True)
    miss = miss[miss["_merge"] == "left_only"]
    info(S, "completeness", f"nflverse has {len(nd):,} QB dropbacks in these 122 games; plays.csv has {len(plays):,}. "
         f"{len(miss):,} nflverse dropbacks are absent from plays.csv ({len(miss) / max(len(nd), 1):.1%}); "
         f"{int((miss['penalty'] == 1).sum()):,} of them involve a penalty.")
    keep("nflverse_dropbacks_missing_from_plays", miss[KEYS + ["play_type", "penalty", "desc"]])
    # is the omission random? compare dropbacks (no penalty) that are in plays.csv with those that are not
    cmp = nd[nd["penalty"] != 1].merge(plays[KEYS].assign(inc=True), on=KEYS, how="left")
    cmp["inc"] = cmp["inc"].fillna(False).astype(bool)

    def profile(x):
        att = x[["complete_pass", "incomplete_pass", "interception"]].fillna(0).sum(axis=1).sum()
        thrown = x["air_yards"].dropna()
        return (f"n={len(x):,}; completion rate {x['complete_pass'].sum() / att:.1%}; sack rate {x['sack'].mean():.1%}; "
                f"median air yards {thrown.median():.0f}; thrown at or behind the line (air yards <= 0): {(thrown <= 0).mean():.0%}")
    info(S, "selection: dropbacks IN plays.csv", profile(cmp[cmp["inc"]]))
    info(S, "selection: dropbacks NOT in plays.csv", profile(cmp[~cmp["inc"]]) +
         ". The omitted plays are mostly passes at or behind the line of scrimmage, so plays.csv is not a random sample of dropbacks.")
    nonpass = nvg[(nvg["qb_dropback"] != 1)].merge(plays[KEYS], on=KEYS, how="inner")
    info(S, "plays.csv rows that nflverse does not mark as QB dropbacks",
         f"{len(nonpass):,} (for example two-point tries or plays nullified by penalty are labelled differently).")


# ── report ───────────────────────────────────────────────────────────────────

def write_report(elapsed, n_games):
    os.makedirs(EXC_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(EXC_DIR, "*.csv")):  # drop files left by earlier runs
        os.remove(old)
    for k, df in EXC.items():
        df.to_csv(os.path.join(EXC_DIR, k + ".csv"), index=False)
    cnt = pd.Series([r["status"] for r in R]).value_counts()
    L = ["# Data validation report", "",
         f"Games checked: {n_games}. Run time {elapsed:.0f}s.", "",
         "**Totals:** " + ", ".join(f"{cnt.get(s, 0)} {s}" for s in ("PASS", "WARN", "FAIL", "INFO", "SKIP") if cnt.get(s, 0)), "",
         "PASS = no violations. WARN = violations at or below the allowed rate. FAIL = violations above the allowed rate. "
         "Rows that fail or warn are saved in `validation/exceptions/`.", ""]
    for sec in OrderedDict.fromkeys(r["section"] for r in R):
        L += [f"## {sec}", "", "| Status | Check | Result | Allowed | Note |", "|---|---|---|---|---|"]
        for r in (x for x in R if x["section"] == sec):
            tol = "" if r["tol"] is None else ("0" if r["tol"] == 0 else f"{r['tol']:.1%}")
            L.append(f"| {r['status']} | {r['check']} | {r['result']} | {tol} | {r['detail']} |")
        L.append("")
    path = os.path.join(OUT, "validation_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    games = pd.read_csv(os.path.join(DATA, "games.csv"))
    plays = pd.read_csv(os.path.join(DATA, "plays.csv"))
    players = pd.read_csv(os.path.join(DATA, "players.csv"))
    pff = pd.read_csv(os.path.join(DATA, "pffScoutingData.csv"))
    files = sorted(glob.glob(os.path.join(TRACK, "tracking_*.csv")))
    file_ids = {int(re.search(r"tracking_(\d+)", f).group(1)) for f in files}
    if args.max_games:
        files = files[: args.max_games]
        keep_ids = {int(re.search(r"tracking_(\d+)", f).group(1)) for f in files}
        games, plays, pff = (x[x["gameId"].isin(keep_ids)].copy() for x in (games, plays, pff))
        file_ids = keep_ids

    print("Scanning tracking files ...")
    members, _at_snap, snaps, pi = scan_tracking(files)  # snaps = frame 0.3 s before the snap tag
    pi = pi.rename(columns={"index": "playId"}) if "playId" not in pi.columns else pi
    mm = members.merge(plays[KEYS + ["possessionTeam", "defensiveTeam"]], on=KEYS, how="left")
    mm["role"] = np.where(mm["pid"] == -1, "ball", np.where(mm["team"] == mm["possessionTeam"], "off",
                          np.where(mm["team"] == mm["defensiveTeam"], "def", "other")))

    print("Running checks ...")
    sec_keys(games, plays, players, pff, file_ids)
    p = sec_plays(plays, pi, players, mm)
    sec_players(players, mm if not args.max_games else None)
    sec_pff(plays, pff, players, mm, snaps, pi)
    sec_tracking_vs_plays(plays, pi, mm, snaps, p)
    sec_external(plays, games, p)

    S = "7. Tracking: structure and physics (every row of every file)"
    for name, (bad, total, tol) in ACC.items():
        add(S, name, bad, total, tol=tol)
    info(S, "observed maxima", f"max player speed {MAXV['speed']:.2f} yd/s ({MAXV['speed'] * 2.045:.1f} mph), "
         f"max player acceleration {MAXV['accel']:.2f} yd/s^2, max one-frame move {MAXV['step']:.2f} yd, max ball speed {MAXV['ball_speed']:.1f} yd/s.")

    path = write_report(time.time() - t0, len(file_ids))
    print(f"\nReport: {path}\n")
    for r in R:
        if r["status"] in ("WARN", "FAIL"):
            print(f"[{r['status']}] {r['section'][:2]} {r['check']} -> {r['result']} {r['detail'][:120]}")
    cnt = pd.Series([r["status"] for r in R]).value_counts().to_dict()
    print("\nTotals:", cnt)


if __name__ == "__main__":
    main()
