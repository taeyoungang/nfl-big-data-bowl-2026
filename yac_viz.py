"""
Case-study figure: what the release-time model saw on specific plays.

Each panel is the pass_forward frame. Every eligible receiver is labelled "completion probability | expected YAC",
the receiver thrown to has a gold ring and a translucent 90-degree wedge showing the space downfield of him,
and the title gives what actually happened.

Three kinds of play are shown, two of each:
    created more than expected   biggest actual-minus-expected YAC on a completion (analysis model)
    expected a lot, got little   expected YAC of 6 or more, smallest actual-minus-expected
    biggest YAC-opportunity gap  largest difference in (completion probability x expected YAC) between the best option and the
                                 receiver thrown to (a play to review, not proof of a mistake: the model does not see the coverage read)

Reads processed/yac_scores.csv; writes figures/yac_04_case_studies.png.

Usage:
    python yac_viz.py
"""

import glob
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from animate_play import DEFENSE_COLOR, OFFENSE_COLOR, draw_field  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
KEYS = ["gameId", "playId"]
THROW = {"pass_forward", "autoevent_passforward"}
CYAN = "#00E5FF"


def pick_cases(sc):
    t = sc[sc["is_target"] & (sc["complete"] == 1) & sc["yac"].notna()].copy()
    t["resid"] = np.minimum(t["yac"], 30) - t["yac_mean_a"]
    cases = []
    for _, r in t.nlargest(2, "resid").iterrows():
        cases.append(("Created more than expected", r["gameId"], r["playId"]))
    for _, r in t[t["yac_mean_a"] >= 6].nsmallest(2, "resid").iterrows():
        cases.append(("Expected a lot, got little", r["gameId"], r["playId"]))
    m = sc[sc["n_elig"] >= 2]
    best = m.groupby(KEYS)["yac_opp"].transform("max")
    tg = m[m["is_target"]].assign(gap=(best - m["yac_opp"])[m["is_target"]])
    for _, r in tg.nlargest(2, "gap").iterrows():
        cases.append(("Biggest YAC-opportunity gap (review, not a verdict)", r["gameId"], r["playId"]))
    return [(k, int(g), int(p)) for k, g, p in cases]


def throw_frame_rows(game_id, play_id):
    path = os.path.join(ROOT, "data", "tracking", f"tracking_{game_id}.csv")
    df = pd.read_csv(path, usecols=["playId", "frameId", "nflId", "team", "x", "y", "playDirection", "event", "jerseyNumber"])
    g = df[df["playId"] == play_id]
    f0 = int(g.loc[g["event"].isin(THROW), "frameId"].min())
    return g[g["frameId"] == f0], g["playDirection"].iloc[0]


def draw_case(ax, sc, plays, kind, game_id, play_id):
    fr, direction = throw_frame_rows(game_id, play_id)
    meta = plays[(plays["gameId"] == game_id) & (plays["playId"] == play_id)].iloc[0]
    rows = sc[(sc["gameId"] == game_id) & (sc["playId"] == play_id)].set_index("nflId")
    draw_field(ax)
    sgn = 1 if direction == "right" else -1
    los = float(meta["absoluteYardlineNumber"])
    ax.set_xlim((los - 8, los + 38) if sgn == 1 else (los - 38, los + 8))
    ax.axvline(los, color="yellow", lw=1.2, ls="--", zorder=2)
    off, dfn = fr[fr["team"] == meta["possessionTeam"]], fr[fr["team"] == meta["defensiveTeam"]]
    ax.scatter(dfn["x"], dfn["y"], s=110, color=DEFENSE_COLOR, edgecolors="white", zorder=4)
    ax.scatter(off["x"], off["y"], s=110, color=OFFENSE_COLOR, edgecolors="white", zorder=4)
    ball = fr[fr["team"] == "football"]
    ax.scatter(ball["x"], ball["y"], s=60, color="#8B4513", edgecolors="#FFD700", zorder=6)
    tgt = rows[rows["is_target"]].iloc[0]
    tgt_id = rows[rows["is_target"]].index[0]
    best_id = rows["yac_opp"].idxmax()
    for nid, r in rows.iterrows():
        p = fr[fr["nflId"] == nid]
        if p.empty:
            continue
        x, y = float(p["x"].iloc[0]), float(p["y"].iloc[0])
        ax.text(x, y + 1.7, f"{r['p_comp']:.0%} | {r['yac_mean']:.1f}", color="#FFD700" if nid == tgt_id else "white", fontsize=7.5, fontweight="bold",
                ha="center", va="bottom", zorder=8, bbox=dict(boxstyle="round,pad=0.15", fc="#1a1a2e", ec="none", alpha=0.8))
        if nid == tgt_id:
            ax.scatter([x], [y], s=520, facecolors="none", edgecolors="#FFD700", linewidths=2.5, zorder=7)
            L = 14
            ax.add_patch(mpatches.Polygon([[x, y], [x + sgn * L, y + L], [x + sgn * L, y - L]], closed=True,
                                          facecolor="#FFD700", alpha=0.12, edgecolor="#FFD700", lw=0.8, zorder=3))
        elif nid == best_id and kind.startswith("Biggest"):
            ax.scatter([x], [y], s=520, facecolors="none", edgecolors=CYAN, linewidths=2.5, linestyles="--", zorder=7)
    tg = tgt
    wrap = lambda s, w=58: "\n".join(textwrap.wrap(re.sub(r"\s+", " ", s), w)[:3])  # noqa: E731
    desc = wrap(str(meta["playDescription"]))
    if kind.startswith("Biggest"):
        head = wrap(f"Thrown to: YAC opportunity {tg['yac_opp']:.1f} yd (completion {tg['p_comp']:.0%} x expected YAC {tg['yac_mean']:.1f}); "
                    f"best option (cyan) {rows['yac_opp'].max():.1f} yd. Result: {tg['passResult']}")
    else:
        head = wrap(f"Actual YAC {tg['yac']:.0f} | model expected {tg['yac_mean_a']:.1f} "
                    f"(10th-90th percentile {max(tg['q10_a'], 0):.0f} to {tg['q90_a']:.0f})")
    ax.set_title(f"{kind}\n{meta['possessionTeam']} vs {meta['defensiveTeam']}, Q{int(meta['quarter'])} {meta['gameClock']}, "
                 f"{meta['pff_passCoverage']}\n{head}\n{desc}", fontsize=8.5, color="black", loc="left")


def main():
    sc = pd.read_csv(os.path.join(ROOT, "processed", "yac_scores.csv"))
    plays = pd.read_csv(os.path.join(ROOT, "data", "plays.csv"))
    cases = pick_cases(sc)
    fig, axes = plt.subplots(2, 3, figsize=(21, 15))
    fig.suptitle("What the release-time model saw: label = completion probability | expected YAC (yards)\n"
                 "Gold ring = receiver thrown to; gold wedge = the 90-degree space downfield of him; yellow dashes = line of scrimmage",
                 fontsize=12)
    order = {"Created more than expected": 0, "Expected a lot, got little": 1, "Biggest YAC-opportunity gap (review, not a verdict)": 2}
    seen = {k: 0 for k in order}
    for kind, g, p in cases:
        col, row = order[kind], seen[kind]
        seen[kind] += 1
        draw_case(axes[row, col], sc, plays, kind, g, p)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.01, wspace=0.06, hspace=0.34)
    out = os.path.join(ROOT, "figures", "yac_04_case_studies.png")
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    print("saved", os.path.relpath(out, ROOT))
    print("cases:", cases)


if __name__ == "__main__":
    main()
