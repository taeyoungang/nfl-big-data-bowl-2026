"""
Exploratory data analysis for the NFL Big Data Bowl dataset.

Reads everything under data/ (games, plays, players, PFF scouting, tracking/*.csv)
and writes figures to figures/.

Usage:
    python eda.py                  # full run (loads all 122 tracking files, ~8.3M rows)
    python eda.py --max-games 3    # quick smoke test on a few games
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from animate_play import draw_field  # noqa: E402  (reuse the field drawing)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TRACK = os.path.join(DATA, "tracking")
OUT = os.path.join(ROOT, "figures")

HZ = 10.0  # tracking frame rate (verified: speed ~= 10 * per-frame distance)

SNAP = {"ball_snap", "autoevent_ballsnap"}
THROW = {"pass_forward", "autoevent_passforward"}
ARRIVE = {"pass_arrived"}
SACK = {"qb_sack"}

POS_GROUP = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE",
    "T": "OL", "G": "OL", "C": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
}
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"]
GROUP_COLOR = {
    "QB": "#d62728", "RB": "#ff7f0e", "WR": "#e377c2", "TE": "#8c564b",
    "OL": "#7f7f7f", "DL": "#1f77b4", "LB": "#17becf", "DB": "#2ca02c",
}
RESULT_NAME = {"C": "Complete", "I": "Incomplete", "IN": "Interception", "S": "Sack", "R": "Scramble"}
RESULT_COLOR = {"C": "#2ca02c", "I": "#ff7f0e", "IN": "#9467bd", "S": "#d62728", "R": "#1f77b4"}
BLUE, RED, GREY = "#4B8BBE", "#d9534f", "#888888"


# ── helpers ──────────────────────────────────────────────────────────────────

def savefig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", os.path.relpath(path, ROOT))


def barh(ax, labels, values, color=BLUE, fmt="{:,.0f}"):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8)
    ax.margins(x=0.18)


def summarize(df, key):
    g = df.groupby(key).agg(
        n=("playId", "size"), c=("is_c", "sum"), att=("is_att", "sum"),
        sacks=("is_sack", "sum"), yds=("playResult", "mean"),
    )
    g["comp_rate"] = g["c"] / g["att"]
    g["sack_rate"] = g["sacks"] / g["n"]
    return g


def parse_height(h):
    s = str(h)
    try:
        if "-" in s:
            f, i = s.split("-")
            return int(f) * 12 + int(i)
        return float(s)
    except ValueError:
        return np.nan


# ── loading ──────────────────────────────────────────────────────────────────

def load_tracking(max_games=None):
    files = sorted(glob.glob(os.path.join(TRACK, "tracking_*.csv")))
    if max_games:
        files = files[:max_games]
    cols = ["gameId", "playId", "nflId", "frameId", "team", "playDirection",
            "x", "y", "s", "a", "o", "dir", "event"]
    dtypes = {"gameId": "int64", "playId": "int64", "frameId": "int16",
              "x": "float32", "y": "float32", "s": "float32", "a": "float32",
              "o": "float32", "dir": "float32"}
    parts = []
    for i, f in enumerate(files, 1):
        parts.append(pd.read_csv(f, usecols=cols, dtype=dtypes))
        if i % 20 == 0 or i == len(files):
            print(f"  read {i}/{len(files)} tracking files")
    return pd.concat(parts, ignore_index=True)


# ── figures ──────────────────────────────────────────────────────────────────

def fig_overview(plays, games):
    p = plays.merge(games[["gameId", "week"]], on="gameId")
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("1. Games and plays overview (every row in plays.csv is a dropback / pass play)", fontsize=14)

    wk = p.groupby("week").size()
    gw = games.groupby("week").size()
    ax = axes[0, 0]
    ax.bar(wk.index, wk.values, color=BLUE)
    for w, v in wk.items():
        ax.text(w, v, f"{v}\n({gw[w]} g)", ha="center", va="bottom", fontsize=8)
    ax.set_title("Dropback plays per week (games in parentheses)")
    ax.set_xlabel("Week"); ax.set_ylabel("Plays"); ax.margins(y=0.2)

    ax = axes[0, 1]
    per_game = p.groupby("gameId").size()
    ax.hist(per_game, bins=15, color=BLUE, edgecolor="white")
    ax.axvline(per_game.mean(), color=RED, ls="--", label=f"mean {per_game.mean():.0f}")
    ax.set_title("Dropback plays per game"); ax.set_xlabel("Plays"); ax.legend()

    ax = axes[0, 2]
    d = p["down"].value_counts().sort_index()
    labels = ["2-pt try" if k == 0 else f"{k}" for k in d.index]
    ax.bar(labels, d.values, color=BLUE)
    for i, v in enumerate(d.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Plays by down"); ax.margins(y=0.12)

    ax = axes[1, 0]
    ax.hist(p["yardsToGo"], bins=np.arange(0, 31) - 0.5, color=BLUE, edgecolor="white")
    ax.set_title("Yards to go for a first down"); ax.set_xlabel("Yards"); ax.set_xlim(-1, 30)

    ax = axes[1, 1]
    ax.hist(p["playResult"].clip(-20, 60), bins=np.arange(-20, 62, 2), color=BLUE, edgecolor="white")
    ax.axvline(p["playResult"].mean(), color=RED, ls="--", label=f"mean {p['playResult'].mean():.1f}")
    ax.axvline(p["playResult"].median(), color="k", ls=":", label=f"median {p['playResult'].median():.0f}")
    ax.set_title("Net yards gained (playResult; clipped to -20..60)"); ax.set_xlabel("Yards"); ax.legend()

    ax = axes[1, 2]
    q = p["quarter"].value_counts().sort_index()
    ax.bar(["Q1", "Q2", "Q3", "Q4", "OT"][: len(q)], q.values, color=BLUE)
    for i, v in enumerate(q.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Plays by quarter"); ax.margins(y=0.12)

    fig.tight_layout()
    savefig(fig, "01_overview.png")


def fig_context(plays):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("2. Play context: what the offense and defense lined up in", fontsize=14)

    f = plays["offenseFormation"].value_counts()
    barh(axes[0, 0], f.index, f.values)
    axes[0, 0].set_title("Offensive formation")

    cov = plays["pff_passCoverage"].value_counts()
    ctype = plays.groupby("pff_passCoverage")["pff_passCoverageType"].agg(lambda s: s.mode().iloc[0])
    cmap = {"Man": RED, "Zone": BLUE, "Other": GREY}
    barh(axes[0, 1], cov.index, cov.values, color=[cmap[ctype[c]] for c in cov.index])
    axes[0, 1].set_title("PFF pass coverage (red = man, blue = zone, grey = other)")

    db = plays["dropBackType"].fillna("(missing)").value_counts()
    barh(axes[0, 2], db.index, db.values)
    axes[0, 2].set_title("Quarterback dropback type")

    po = plays["personnelO"].value_counts().head(8)
    barh(axes[1, 0], po.index, po.values)
    axes[1, 0].set_title("Top offensive personnel groupings")

    pdn = plays["personnelD"].value_counts().head(8)
    barh(axes[1, 1], pdn.index, pdn.values)
    axes[1, 1].set_title("Top defensive personnel groupings")

    ax = axes[1, 2]
    dib = plays["defendersInBox"].dropna()
    ax.hist(dib, bins=np.arange(0, 12) - 0.5, color=BLUE, edgecolor="white")
    pa = plays["pff_playAction"].mean()
    ax.set_title(f"Defenders in the box (play action on {pa:.0%} of plays)")
    ax.set_xlabel("Defenders")

    fig.tight_layout()
    savefig(fig, "02_play_context.png")


def fig_outcomes(plays):
    d = plays.copy()
    d["is_c"] = d["passResult"] == "C"
    d["is_att"] = d["passResult"].isin(["C", "I", "IN"])
    d["is_sack"] = d["passResult"] == "S"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("3. Outcomes: how results vary with coverage, down and formation", fontsize=14)

    ax = axes[0, 0]
    order = ["C", "I", "S", "R", "IN"]
    r = d["passResult"].value_counts().reindex(order)
    ax.bar([RESULT_NAME[k] for k in order], r.values, color=[RESULT_COLOR[k] for k in order])
    for i, v in enumerate(r.values):
        ax.text(i, v, f"{v:,}\n{v / len(d):.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Dropback outcome (passResult)"); ax.margins(y=0.15)

    cov = summarize(d, "pff_passCoverage")
    cov = cov[cov["n"] >= 100]

    ax = axes[0, 1]
    c = cov.sort_values("comp_rate", ascending=False)
    barh(ax, [f"{i} (n={int(n)})" for i, n in zip(c.index, c["n"])], c["comp_rate"] * 100, fmt="{:.1f}%")
    ax.set_title("Completion rate by coverage (completions / attempts)")

    ax = axes[0, 2]
    c = cov.sort_values("yds", ascending=False)
    barh(ax, [f"{i} (n={int(n)})" for i, n in zip(c.index, c["n"])], c["yds"], fmt="{:.2f}")
    ax.set_title("Mean net yards per dropback by coverage")

    ax = axes[1, 0]
    c = cov.sort_values("sack_rate", ascending=False)
    barh(ax, [f"{i} (n={int(n)})" for i, n in zip(c.index, c["n"])], c["sack_rate"] * 100, color=RED, fmt="{:.1f}%")
    ax.set_title("Sack rate by coverage")

    ax = axes[1, 1]
    dd = d[d["down"].between(1, 4)]
    data = [dd.loc[dd["down"] == k, "playResult"] for k in (1, 2, 3, 4)]
    ax.boxplot(data, tick_labels=[f"Down {k}\n(n={len(x):,})" for k, x in zip((1, 2, 3, 4), data)],
               showfliers=False)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Net yards by down (outliers hidden)"); ax.set_ylabel("Yards")

    ax = axes[1, 2]
    fm = summarize(d, "offenseFormation")
    fm = fm[fm["n"] >= 100].sort_values("yds", ascending=False)
    barh(ax, [f"{i} (n={int(n)})" for i, n in zip(fm.index, fm["n"])], fm["yds"], fmt="{:.2f}")
    ax.set_title("Mean net yards per dropback by formation")

    fig.tight_layout()
    savefig(fig, "03_outcomes.png")


def fig_players(players):
    p = players.copy()
    p["group"] = p["officialPosition"].map(POS_GROUP)
    p["height_in"] = p["height"].map(parse_height)
    p["age"] = (pd.Timestamp("2021-09-09") - pd.to_datetime(p["birthDate"], errors="coerce")).dt.days / 365.25

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("4. Players (players.csv): who is on the field", fontsize=14)

    pc = p["officialPosition"].value_counts()
    barh(axes[0, 0], pc.index, pc.values,
         color=[GROUP_COLOR.get(POS_GROUP.get(k), GREY) for k in pc.index])
    axes[0, 0].set_title(f"Players by official position ({len(p):,} players)")

    for ax, col, title in ((axes[0, 1], "height_in", "Height (inches) by position group"),
                           (axes[1, 0], "weight", "Weight (lb) by position group")):
        data = [p.loc[p["group"] == g, col].dropna() for g in GROUPS]
        bp = ax.boxplot(data, tick_labels=GROUPS, patch_artist=True, showfliers=False)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(GROUP_COLOR[g]); patch.set_alpha(0.7)
        ax.set_title(title)

    ax = axes[1, 1]
    ax.hist(p["age"].dropna(), bins=25, color=BLUE, edgecolor="white")
    ax.set_title(f"Age on 2021-09-09 (birthDate missing for {p['birthDate'].isna().sum()} players)")
    ax.set_xlabel("Years")

    fig.tight_layout()
    savefig(fig, "04_players.png")


def fig_pff(plays, pff):
    rush = pff[pff["pff_role"] == "Pass Rush"].copy()
    rush["pressure"] = rush[["pff_hit", "pff_hurry", "pff_sack"]].fillna(0).sum(axis=1) > 0
    rush["sack"] = rush["pff_sack"].fillna(0) > 0
    blk = pff[pff["pff_role"] == "Pass Block"].copy()
    blk["allowed"] = blk[["pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed"]].fillna(0).sum(axis=1) > 0
    blk["beaten"] = blk["pff_beatenByDefender"].fillna(0) > 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("5. PFF scouting data: roles, pass rush and pass protection", fontsize=14)

    r = pff["pff_role"].value_counts()
    barh(axes[0, 0], r.index, r.values)
    axes[0, 0].set_title("Player-play rows by PFF role")

    ax = axes[0, 1]
    by = rush.groupby("pff_positionLinedUp").agg(n=("playId", "size"), pr=("pressure", "mean"), sk=("sack", "mean"))
    by = by[by["n"] >= 300].sort_values("pr", ascending=False)
    y = np.arange(len(by))
    ax.barh(y - 0.2, by["pr"] * 100, height=0.4, color=BLUE, label="Pressure (hit/hurry/sack)")
    ax.barh(y + 0.2, by["sk"] * 100, height=0.4, color=RED, label="Sack")
    ax.set_yticks(y); ax.set_yticklabels([f"{i} (n={int(n):,})" for i, n in zip(by.index, by["n"])])
    ax.invert_yaxis(); ax.legend(); ax.set_xlabel("% of rushes")
    ax.set_title("Pass-rusher success by alignment")

    ax = axes[0, 2]
    bb = blk.groupby("pff_positionLinedUp").agg(n=("playId", "size"), al=("allowed", "mean"), be=("beaten", "mean"))
    bb = bb[bb["n"] >= 300].sort_values("al", ascending=False)
    y = np.arange(len(bb))
    ax.barh(y - 0.2, bb["al"] * 100, height=0.4, color=RED, label="Pressure allowed")
    ax.barh(y + 0.2, bb["be"] * 100, height=0.4, color=GREY, label="Beaten, no pressure charged")
    ax.set_yticks(y); ax.set_yticklabels([f"{i} (n={int(n):,})" for i, n in zip(bb.index, bb["n"])])
    ax.invert_yaxis(); ax.legend(); ax.set_xlabel("% of pass-block snaps")
    ax.set_title("Pass-protection breakdowns by alignment")

    bt = blk["pff_blockType"].value_counts()
    barh(axes[1, 0], bt.index, bt.values)
    axes[1, 0].set_title("Pass-block type (see README for codes)")

    # play-level: number of rushers vs pressure
    nr = rush.groupby(["gameId", "playId"]).size().rename("n_rush")
    pp = rush.groupby(["gameId", "playId"])["pressure"].any().rename("pressured")
    d = plays.merge(nr.reset_index(), on=["gameId", "playId"], how="left")
    d = d.merge(pp.reset_index(), on=["gameId", "playId"], how="left")
    d["n_rush"] = d["n_rush"].fillna(0).astype(int)
    d["pressured"] = d["pressured"].fillna(False).astype(bool)
    d["rush_bin"] = d["n_rush"].clip(upper=7)

    ax = axes[1, 1]
    g = d.groupby("rush_bin").agg(n=("playId", "size"), pr=("pressured", "mean"),
                                  sk=("passResult", lambda s: (s == "S").mean()))
    g = g[g["n"] >= 30]
    ax.plot(g.index, g["pr"] * 100, "o-", color=BLUE, label="Pressure rate")
    ax.plot(g.index, g["sk"] * 100, "o-", color=RED, label="Sack rate")
    for k, row in g.iterrows():
        ax.text(k, row["pr"] * 100 + 1.5, f"n={int(row['n']):,}", ha="center", fontsize=7)
    ax.set_xticks(g.index)
    ax.set_xticklabels([("7+" if k == 7 else str(k)) for k in g.index])
    ax.set_xlabel("Pass rushers on the play"); ax.set_ylabel("% of dropbacks"); ax.legend()
    ax.set_title("More rushers -> more pressure?")

    ax = axes[1, 2]
    order = ["C", "I", "IN", "S", "R"]
    ct = pd.crosstab(d["pressured"], d["passResult"], normalize="index").reindex(columns=order).fillna(0) * 100
    ct = ct.reindex([False, True])
    bottom = np.zeros(len(ct))
    for k in order:
        ax.bar(["No pressure", "Pressured"], ct[k].values, bottom=bottom, color=RESULT_COLOR[k], label=RESULT_NAME[k])
        for i, (v, b) in enumerate(zip(ct[k].values, bottom)):
            if v > 4:
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9)
        bottom += ct[k].values
    counts = d["pressured"].value_counts()
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"No pressure\n(n={counts.get(False, 0):,})", f"Pressured\n(n={counts.get(True, 0):,})"])
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=8)
    ax.set_title("Outcome mix with and without pressure")

    fig.tight_layout()
    savefig(fig, "05_pff_scouting.png")
    return d[["gameId", "playId", "n_rush", "pressured"]]


def fig_tracking_structure(tr, plays, pe):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("6. Tracking data structure (10 frames per second, one row per player/ball per frame)", fontsize=14)

    ax = axes[0, 0]
    rows = tr.groupby("gameId").size()
    ax.hist(rows / 1e3, bins=20, color=BLUE, edgecolor="white")
    ax.set_title(f"Tracking rows per game (total {len(tr) / 1e6:.2f}M rows, {len(rows)} games)")
    ax.set_xlabel("Thousand rows")

    ax = axes[0, 1]
    ax.hist(pe["n_frames"].clip(upper=120), bins=np.arange(0, 121, 4), color=BLUE, edgecolor="white")
    ax.axvline(pe["n_frames"].median(), color=RED, ls="--",
               label=f"median {pe['n_frames'].median():.0f} frames = {pe['n_frames'].median() / HZ:.1f} s")
    ax.set_title("Frames per play (clipped at 120)"); ax.set_xlabel("Frames"); ax.legend()

    ax = axes[0, 2]
    per_frame = tr.groupby(["gameId", "playId", "frameId"]).size().value_counts().sort_index()
    ax.bar(per_frame.index.astype(str), per_frame.values, color=BLUE)
    for i, v in enumerate(per_frame.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Tracked objects per frame (22 players + 1 ball = 23)")
    ax.set_xlabel("Rows in frame"); ax.set_ylabel("Frames"); ax.margins(y=0.15)

    ax = axes[1, 0]
    ev = (tr.loc[tr["event"].notna(), ["gameId", "playId", "event"]]
          .drop_duplicates().groupby("event").size().sort_values(ascending=False))
    barh(ax, ev.index, ev.values)
    ax.set_title("Plays containing each event tag")

    ax = axes[1, 1]
    t_throw = (pe["throw_frame"] - pe["snap_frame"]) / HZ
    t_sack = (pe["sack_frame"] - pe["snap_frame"]) / HZ
    bins = np.arange(0, 8.01, 0.25)
    ax.hist(t_throw.dropna(), bins=bins, alpha=0.7, color=BLUE, label=f"snap -> pass_forward (median {t_throw.median():.1f}s)")
    ax.hist(t_sack.dropna(), bins=bins, alpha=0.7, color=RED, label=f"snap -> qb_sack (median {t_sack.median():.1f}s)")
    ax.set_title("Time from snap to throw or sack"); ax.set_xlabel("Seconds"); ax.legend()

    ax = axes[1, 2]
    after = (pe["n_frames"] - pe["throw_frame"]).dropna()
    vc = after.clip(upper=15).astype(int).value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=[RED if k == 5 else BLUE for k in vc.index])
    share5 = (after == 5).mean()
    ax.set_title(f"Frames tracked after the throw: {share5:.1%} of throws end at exactly 5 (0.5 s)")
    ax.set_xlabel("Frames after pass_forward"); ax.set_ylabel("Throws")

    fig.tight_layout()
    savefig(fig, "06_tracking_structure.png")


def fig_spatial(tr):
    snap = tr[(tr["fss"] == 0) & tr["group"].notna()].copy()
    sign = np.where(snap["playDirection"] == "right", 1.0, -1.0)
    snap["x_rel"] = (snap["x"] - snap["absoluteYardlineNumber"]) * sign
    snap["y_n"] = np.where(snap["playDirection"] == "right", snap["y"], 53.3 - snap["y"])
    snap["is_off"] = snap["team"] == snap["possessionTeam"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.suptitle("7. Where players line up at the snap (all plays flipped so offense drives left -> right; x = 0 is the line of scrimmage)",
                 fontsize=13)
    xb, yb = np.arange(-15, 15.1, 0.5), np.arange(0, 53.4, 1.0)
    for ax, mask, title, cmap in ((axes[0], snap["is_off"], "Offense (density)", "Reds"),
                                  (axes[1], ~snap["is_off"], "Defense (density)", "Blues")):
        s = snap[mask]
        h = ax.hist2d(s["x_rel"], s["y_n"], bins=[xb, yb], cmap=cmap)
        plt.colorbar(h[3], ax=ax, label="player-snaps")
        ax.axvline(0, color="k", lw=1)
        ax.set_title(title); ax.set_xlabel("Yards downfield of the line of scrimmage"); ax.set_ylabel("Field width (yd)")

    ax = axes[2]
    s = snap.sample(min(40000, len(snap)), random_state=0)
    for g in GROUPS:
        sg = s[s["group"] == g]
        ax.scatter(sg["x_rel"], sg["y_n"], s=3, alpha=0.35, color=GROUP_COLOR[g], label=g)
    ax.set_xlim(-15, 15); ax.set_ylim(0, 53.3); ax.axvline(0, color="k", lw=1)
    ax.legend(markerscale=5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Same snap positions, coloured by position group (40k sample)")
    ax.set_xlabel("Yards downfield of the line of scrimmage")

    fig.tight_layout()
    savefig(fig, "07_snap_alignment.png")
    return snap


def fig_kinematics(tr, players_group, plays_t):
    pl = tr[tr["group"].notna()]
    post = pl[pl["fss"] >= 0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("8. Motion: speed (yd/s), acceleration (yd/s^2) and orientation", fontsize=14)

    ax = axes[0, 0]
    mx = post.groupby(["gameId", "playId", "nflId"])["s"].max().reset_index()
    mx["group"] = mx["nflId"].map(players_group)
    data = [mx.loc[mx["group"] == g, "s"] for g in GROUPS]
    bp = ax.boxplot(data, tick_labels=GROUPS, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], GROUPS):
        patch.set_facecolor(GROUP_COLOR[g]); patch.set_alpha(0.7)
    ax.set_title("Top speed reached per player per play (after snap)"); ax.set_ylabel("yd/s")

    ax = axes[0, 1]
    win = pl[(pl["fss"] >= -10) & (pl["fss"] <= 50)]
    prof = win.groupby(["group", "fss"], observed=True)["s"].mean().unstack(0)
    for g in GROUPS:
        ax.plot(prof.index / HZ, prof[g], color=GROUP_COLOR[g], label=g, lw=2)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_title("Average speed around the snap"); ax.set_xlabel("Seconds from snap"); ax.set_ylabel("yd/s")
    ax.legend(ncol=4, fontsize=8)

    ax = axes[0, 2]
    ax.hist(post["s"], bins=np.arange(0, 12.1, 0.25), color=BLUE, edgecolor="white")
    ax.axvline(post["s"].median(), color=RED, ls="--", label=f"median {post['s'].median():.2f}")
    ax.set_yscale("log"); ax.legend()
    ax.set_title("Speed of all player-frames after the snap"); ax.set_xlabel("yd/s")

    ax = axes[1, 0]
    ax.hist(post["a"].clip(upper=12), bins=np.arange(0, 12.1, 0.25), color=BLUE, edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Acceleration magnitude (always >= 0; clipped at 12)"); ax.set_xlabel("yd/s^2")

    ax = axes[1, 1]
    mv = post[post["s"] > 3]
    diff = np.abs((mv["o"] - mv["dir"] + 180) % 360 - 180)
    data = [diff[(mv["group"] == g).values] for g in GROUPS]
    bp = ax.boxplot(data, tick_labels=GROUPS, patch_artist=True, showfliers=False)
    for patch, g in zip(bp["boxes"], GROUPS):
        patch.set_facecolor(GROUP_COLOR[g]); patch.set_alpha(0.7)
    ax.set_title("|facing direction - movement direction| when moving > 3 yd/s")
    ax.set_ylabel("Degrees (0 = running straight ahead, 180 = backpedalling)")

    ax = axes[1, 2]
    for res in ("C", "I", "IN"):
        s = plays_t.loc[plays_t["passResult"] == res, "release_depth"].dropna()
        ax.hist(s.clip(-5, 15), bins=np.arange(-5, 15.1, 0.5), histtype="step", lw=2, density=True,
                color=RESULT_COLOR[res], label=f"{RESULT_NAME[res]} (n={len(s):,}, median {s.median():.1f} yd)")
    ax.axvline(0, color="k", lw=1)
    ax.set_title("Ball position at release: yards behind the line of scrimmage")
    ax.set_xlabel("Yards behind the line of scrimmage (negative = past it)"); ax.legend(fontsize=8)

    fig.tight_layout()
    savefig(fig, "08_motion.png")


def fig_timing(plays_t):
    d = plays_t.copy()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("9. Time to throw, pressure and results (derived from tracking events + PFF)", fontsize=14)

    ax = axes[0]
    bins = np.arange(0, 8.01, 0.25)
    for res in ("C", "I", "IN", "S"):
        s = d.loc[d["passResult"] == res, "t_event"].dropna()
        lab = "Sack (snap -> sack)" if res == "S" else f"{RESULT_NAME[res]} (snap -> throw)"
        ax.hist(s, bins=bins, histtype="step", lw=2, density=True, color=RESULT_COLOR[res],
                label=f"{lab}, median {s.median():.1f}s, n={len(s):,}")
    ax.set_title("Time from snap to throw/sack"); ax.set_xlabel("Seconds"); ax.legend(fontsize=8)

    thr = d[d["passResult"].isin(["C", "I", "IN"]) & d["t_event"].notna()].copy()
    edges = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 10]
    labels = ["<2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0+"]
    thr["ttt"] = pd.cut(thr["t_event"], edges, labels=labels, right=False)
    g = thr.groupby("ttt", observed=True).agg(n=("playId", "size"), pr=("pressured", "mean"),
                                            c=("is_c", "sum"), att=("is_att", "sum"), yds=("playResult", "mean"))
    g = g[g["n"] >= 50]
    g["comp"] = g["c"] / g["att"]

    ax = axes[1]
    ax.bar(g.index.astype(str), g["pr"] * 100, color=RED)
    for i, (v, n) in enumerate(zip(g["pr"] * 100, g["n"])):
        ax.text(i, v, f"{v:.0f}%\nn={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Pressure rate vs. time to throw (throws only)")
    ax.set_xlabel("Seconds from snap to throw"); ax.set_ylabel("% of throws with pressure"); ax.margins(y=0.2)

    ax = axes[2]
    ax.plot(g.index.astype(str), g["comp"] * 100, "o-", color=BLUE, label="Completion rate (%)")
    ax.set_ylabel("Completion rate (%)", color=BLUE)
    ax2 = ax.twinx()
    ax2.plot(g.index.astype(str), g["yds"], "s-", color=GREY, label="Mean net yards")
    ax2.set_ylabel("Mean net yards", color=GREY)
    ax.set_xlabel("Seconds from snap to throw")
    ax.set_title("Completion rate and yards vs. time to throw")

    fig.tight_layout()
    savefig(fig, "09_time_to_throw.png")


def fig_example_plays(tr, plays, picks):
    fig, axes = plt.subplots(len(picks), 1, figsize=(15, 6.8 * len(picks)))
    axes = np.atleast_1d(axes)
    fig.suptitle("10. What one play looks like in the tracking data (every player's path, snap to end)", fontsize=14)
    for ax, (gid, pid) in zip(axes, picks):
        draw_field(ax)
        d = tr[(tr["gameId"] == gid) & (tr["playId"] == pid)]
        meta = plays[(plays["gameId"] == gid) & (plays["playId"] == pid)].iloc[0]
        off = meta["possessionTeam"]
        snap_f = d["snap_frame"].iloc[0]
        for nid, p in d[d["team"] != "football"].groupby("nflId"):
            p = p.sort_values("frameId")
            c = RED if p["team"].iloc[0] == off else BLUE
            ax.plot(p["x"], p["y"], color=c, lw=1.5, alpha=0.85, zorder=3)
            at_snap = p[p["frameId"] == snap_f]
            if len(at_snap):
                ax.scatter(at_snap["x"], at_snap["y"], s=45, color=c, edgecolor="white", zorder=5)
            ax.scatter(p["x"].iloc[-1], p["y"].iloc[-1], s=45, marker="s", color=c, edgecolor="black", zorder=5)
        b = d[d["team"] == "football"].sort_values("frameId")
        ax.plot(b["x"], b["y"], color="#FFD700", lw=3, zorder=4)
        los = meta["absoluteYardlineNumber"]
        right = d["playDirection"].iloc[0] == "right"
        ax.set_xlim((los - 12, los + 48) if right else (los - 48, los + 12))
        ax.axvline(los, color="yellow", lw=1.5, ls="--", zorder=2)
        ax.set_title(f"{off} offense ({'drives right' if right else 'drives left'}) vs "
                     f"{meta['pff_passCoverage']} ({meta['pff_passCoverageType']})\n"
                     f"{meta['playDescription'][:80]}", fontsize=10, color="black", loc="left")
        ax.text(0.01, 0.02, "circle = position at snap, square = last tracked position, gold = ball, dashed = line of scrimmage",
                transform=ax.transAxes, color="white", fontsize=8)
    fig.tight_layout()
    savefig(fig, "10_example_plays.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    print("Loading general data ...")
    games = pd.read_csv(os.path.join(DATA, "games.csv"))
    plays = pd.read_csv(os.path.join(DATA, "plays.csv"))
    players = pd.read_csv(os.path.join(DATA, "players.csv"))
    pff = pd.read_csv(os.path.join(DATA, "pffScoutingData.csv"))
    print("Loading tracking data ...")
    tr = load_tracking(args.max_games)
    if args.max_games:
        keep = tr["gameId"].unique()
        plays = plays[plays["gameId"].isin(keep)].copy()
        pff = pff[pff["gameId"].isin(keep)].copy()
        games = games[games["gameId"].isin(keep)].copy()
    print(f"  tracking rows: {len(tr):,} | loaded in {time.time() - t0:.0f}s")

    # ── play-level event frames ──
    ev = tr.loc[tr["event"].notna(), ["gameId", "playId", "frameId", "event"]]
    keys = ["gameId", "playId"]

    def first_frame(names, col):
        return ev[ev["event"].isin(names)].groupby(keys)["frameId"].min().rename(col)

    pe = pd.concat([first_frame(SNAP, "snap_frame"), first_frame(THROW, "throw_frame"),
                    first_frame(SACK, "sack_frame")], axis=1)
    pe["n_frames"] = tr.groupby(keys)["frameId"].max()
    pe = pe.astype("float64")

    # ── integrity report ──
    print("\n=== DATA INTEGRITY ===")
    n_tr_plays = len(pe)
    both = plays.merge(pe.reset_index(), on=keys, how="left", indicator=True)
    print(f"plays.csv rows: {len(plays):,} | tracked plays: {n_tr_plays:,} | "
          f"plays without tracking: {(both['_merge'] == 'left_only').sum()}")
    print(f"plays with snap event: {pe['snap_frame'].notna().sum():,} | throw: {pe['throw_frame'].notna().sum():,} | "
          f"qb_sack: {pe['sack_frame'].notna().sum():,}")
    pre = (pe["snap_frame"] - 1).dropna()
    print(f"frames tracked BEFORE snap: median {pre.median():.0f} ({pre.median() / HZ:.1f}s), "
          f"p90 {pre.quantile(0.9):.0f}, max {pre.max():.0f}")
    after_thr = (pe["n_frames"] - pe["throw_frame"]).dropna()
    print(f"frames tracked after throw: median {after_thr.median():.0f}, share exactly 5: {(after_thr == 5).mean():.1%}, "
          f"max {after_thr.max():.0f}")
    print("nulls in tracking (share of rows):")
    print((tr[["nflId", "o", "dir", "event"]].isna().mean() * 100).round(1).to_string())
    ball_rows = (tr["team"] == "football").sum()
    print(f"ball rows: {ball_rows:,} | rows with null nflId: {tr['nflId'].isna().sum():,} | "
          f"rows with null o: {tr['o'].isna().sum():,}")
    print(f"out of bounds (x<0|x>120|y<0|y>53.3): {((tr['x'] < 0) | (tr['x'] > 120) | (tr['y'] < 0) | (tr['y'] > 53.3)).mean():.3%}")
    print("PFF 'Pass' rows (one per play):", (pff["pff_role"] == "Pass").sum(), "| plays:", len(plays))

    # ── enrich tracking ──
    pcols = plays[keys + ["possessionTeam", "absoluteYardlineNumber", "passResult"]]
    tr = tr.merge(pcols, on=keys, how="left")
    tr = tr.merge(pe[["snap_frame"]].reset_index(), on=keys, how="left")
    tr["fss"] = tr["frameId"] - tr["snap_frame"]  # frames since snap
    pos = players.set_index("nflId")["officialPosition"]
    players_group = players.set_index("nflId")["officialPosition"].map(POS_GROUP)
    tr["group"] = tr["nflId"].map(players_group)
    unmatched = tr.loc[tr["team"] != "football", "group"].isna().sum()
    print(f"player rows with no position match in players.csv: {unmatched:,}")

    pl = tr[tr["group"].notna()]
    print("player speed (yd/s) quantiles:", pl["s"].quantile([0.5, 0.99, 0.999]).round(2).to_dict(), "max:", round(float(pl["s"].max()), 2))
    ball = tr[tr["team"] == "football"]
    snapb = ball[ball["fss"] == 0]
    sgn = np.where(snapb["playDirection"] == "right", 1, -1)
    print("ball x relative to absoluteYardlineNumber at snap (median, should be ~0):",
          round(float(np.nanmedian((snapb["x"] - snapb["absoluteYardlineNumber"]) * sgn)), 2))

    # ── play-level table with timing, depth, pressure ──
    d = plays.merge(pe.reset_index(), on=keys, how="left")
    d["is_c"] = d["passResult"] == "C"
    d["is_att"] = d["passResult"].isin(["C", "I", "IN"])
    d["t_throw"] = (d["throw_frame"] - d["snap_frame"]) / HZ
    d["t_sack"] = (d["sack_frame"] - d["snap_frame"]) / HZ
    d["t_event"] = np.where(d["passResult"] == "S", d["t_sack"], d["t_throw"])
    # ball position at the throw frame, as yards BEHIND the line of scrimmage
    rel = ball[["gameId", "playId", "frameId", "x"]].rename(columns={"x": "x_rel", "frameId": "throw_frame"})
    rel["throw_frame"] = rel["throw_frame"].astype("float64")
    d = d.merge(rel, on=keys + ["throw_frame"], how="left")
    right = tr.groupby(keys)["playDirection"].first().eq("right").rename("right").reset_index()
    d = d.merge(right, on=keys, how="left")
    toward_goal = np.where(d["right"].fillna(True).astype(bool), 1.0, -1.0)
    d["release_depth"] = -(d["x_rel"] - d["absoluteYardlineNumber"]) * toward_goal

    print("\n=== FIGURES ===")
    fig_overview(plays, games)
    fig_context(plays)
    fig_outcomes(plays)
    fig_players(players)
    pres = fig_pff(plays, pff)
    d = d.merge(pres, on=keys, how="left")
    fig_tracking_structure(tr, plays, pe)
    fig_spatial(tr)
    fig_kinematics(tr, players_group, d)
    fig_timing(d)
    picks = [(2021090900, 97), (2021090900, 137)]
    picks = [p for p in picks if ((tr["gameId"] == p[0]) & (tr["playId"] == p[1])).any()]
    if picks:
        fig_example_plays(tr, plays, picks)

    # key numbers used in the written summary
    print("\n=== KEY NUMBERS ===")
    att = d[d["is_att"]]
    print(f"completion rate: {att['is_c'].mean():.1%} | sack rate: {(d['passResult'] == 'S').mean():.1%} | "
          f"INT rate: {(d['passResult'] == 'IN').mean():.1%} | scramble: {(d['passResult'] == 'R').mean():.1%}")
    print(f"pressure rate (any rusher hit/hurry/sack): {d['pressured'].mean():.1%}")
    print(f"median time to throw: {d['t_throw'].median():.2f}s | median sack time: {d['t_sack'].median():.2f}s")
    print(f"median ball depth behind LOS at release: {d['release_depth'].median():.1f} yd")
    pr = d[d["is_att"]].groupby("pressured")["is_c"].mean()
    print("completion rate by pressure:", pr.round(3).to_dict())
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
