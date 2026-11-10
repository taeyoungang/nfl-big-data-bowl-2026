"""
NFL Play Animator
-----------------
Animates all 22 players + the football for any play in the dataset.

Usage:
    python animate_play.py                          # picks a default play (TB vs DAL, play 97)
    python animate_play.py --game 2021090900 --play 137
    python animate_play.py --game 2021090900 --play 97 --save  # saves as GIF

Controls when the window is open:
    Space   pause / resume
    ←  /  →  step back / forward one frame while paused
    q       quit
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRACKING_DIR = os.path.join(DATA_DIR, "tracking")

# ── Colour scheme ────────────────────────────────────────────────────────────
FIELD_GREEN   = "#3a7d44"
FIELD_LINE    = "#ffffff"
ENDZONE_COLOR = "#2d6e38"

# Team dot colours.  Offensive team = warm, Defensive = cool, ball = gold.
OFFENSE_COLOR  = "#FF4B4B"   # red
DEFENSE_COLOR  = "#4B9EFF"   # blue
BALL_COLOR     = "#FFD700"   # gold
FOOTBALL_COLOR = "#8B4513"   # brown

KEY_EVENT_COLORS = {
    "ball_snap":               "#FFFFFF",
    "autoevent_ballsnap":      "#FFFFFF",
    "line_set":                "#CCCCCC",
    "pass_forward":            "#FFD700",
    "autoevent_passforward":   "#FFD700",
    "pass_arrived":            "#FFA500",
    "pass_outcome_caught":     "#00FF7F",
    "pass_outcome_incomplete": "#FF4500",
    "qb_sack":                 "#FF0000",
    "fumble":                  "#FF0000",
    "run":                     "#FF8C00",
    "play_action":             "#DDA0DD",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_tracking(game_id: int) -> pd.DataFrame:
    path = os.path.join(TRACKING_DIR, f"tracking_{game_id}.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: Tracking file not found: {path}")
    return pd.read_csv(path)


def load_support_data():
    plays   = pd.read_csv(os.path.join(DATA_DIR, "plays.csv"))
    players = pd.read_csv(os.path.join(DATA_DIR, "players.csv"))
    games   = pd.read_csv(os.path.join(DATA_DIR, "games.csv"))
    pff     = pd.read_csv(os.path.join(DATA_DIR, "pffScoutingData.csv"))
    return plays, players, games, pff


def get_play_data(tracking: pd.DataFrame, game_id: int, play_id: int):
    df = tracking[(tracking["gameId"] == game_id) & (tracking["playId"] == play_id)].copy()
    if df.empty:
        sys.exit(f"ERROR: No tracking data for game {game_id} play {play_id}")
    return df


# ── Field drawing ─────────────────────────────────────────────────────────────

def draw_field(ax):
    """Draw an NFL field (120 yd x 53⅓ yd in tracking units)."""
    # Field background
    ax.set_facecolor(FIELD_GREEN)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # axis("off") also hides the axes background, so draw the turf explicitly
    ax.add_patch(mpatches.Rectangle((0, 0), 120, 53.3, facecolor=FIELD_GREEN, zorder=-1))

    # End zones
    for x0, x1 in [(0, 10), (110, 120)]:
        ax.add_patch(mpatches.Rectangle((x0, 0), x1 - x0, 53.3,
                                        facecolor=ENDZONE_COLOR, zorder=0))

    # Yard lines every 5 yards (in tracking coords: 10..110)
    for yard in range(10, 111, 5):
        lw = 1.5 if yard % 10 == 0 else 0.6
        ax.plot([yard, yard], [0, 53.3], color=FIELD_LINE, lw=lw, alpha=0.6, zorder=1)

    # Side lines & end lines
    ax.plot([0, 120], [0, 0],    color=FIELD_LINE, lw=2, zorder=1)
    ax.plot([0, 120], [53.3, 53.3], color=FIELD_LINE, lw=2, zorder=1)
    ax.plot([0, 0],   [0, 53.3], color=FIELD_LINE, lw=2, zorder=1)
    ax.plot([120, 120], [0, 53.3], color=FIELD_LINE, lw=2, zorder=1)

    # Yard number labels (10–50 each side)
    for yard_num in range(10, 51, 10):
        x_pos = 10 + yard_num          # tracking x for left half
        x_pos_r = 110 - yard_num       # tracking x for right half  (mirror)
        label = str(yard_num)

        for xp in [x_pos, x_pos_r]:
            ax.text(xp, 2.5, label, color=FIELD_LINE, fontsize=7,
                    ha="center", va="bottom", alpha=0.7, zorder=2)
            ax.text(xp, 50.8, label, color=FIELD_LINE, fontsize=7,
                    ha="center", va="top", alpha=0.7, rotation=180, zorder=2)

    # Hash marks (every yard between 10–110, inner hashes at ~18.5 & ~34.8)
    for yard in range(10, 111):
        for y in [18.5, 34.8]:
            ax.plot([yard, yard], [y - 0.5, y + 0.5],
                    color=FIELD_LINE, lw=0.4, alpha=0.5, zorder=1)


# ── Build frame list ──────────────────────────────────────────────────────────

def build_frames(play_df: pd.DataFrame):
    """Return sorted list of frameIds and a dict {frameId -> sub-df}."""
    frames = sorted(play_df["frameId"].unique())
    frame_map = {f: play_df[play_df["frameId"] == f] for f in frames}
    return frames, frame_map


def get_event_frames(play_df: pd.DataFrame) -> dict:
    """Map frameId -> event string for frames that have a named event."""
    ev = play_df[play_df["event"].notna() & (play_df["event"] != "None")]
    return dict(zip(ev["frameId"], ev["event"]))


# ── Animation ────────────────────────────────────────────────────────────────

class PlayAnimator:
    def __init__(self, game_id, play_id, play_meta, players_df, pff_df, save=False):
        self.game_id   = game_id
        self.play_id   = play_id
        self.meta      = play_meta   # single-row Series from plays.csv
        self.players   = players_df
        self.pff       = pff_df
        self.save      = save

        self.paused    = False
        self.cur_frame = 0

    def run(self, play_df):
        self.frames, self.frame_map = build_frames(play_df)
        self.event_frames = get_event_frames(play_df)

        # Identify possession (offense) team
        self.offense_team = str(self.meta["possessionTeam"]) if self.meta is not None else None
        self.defense_team = str(self.meta["defensiveTeam"])  if self.meta is not None else None

        # Build player-role lookup for dot colouring
        self.player_colors = self._build_player_colors(play_df)

        # Figure setup
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        self.fig.patch.set_facecolor("#1a1a2e")
        self.ax.set_facecolor(FIELD_GREEN)
        draw_field(self.ax)

        # Title
        title_text = self._build_title()
        self.title = self.fig.suptitle(title_text, color="white", fontsize=11, y=0.98)

        # Event banner (shown briefly when a key event fires)
        self.event_banner = self.ax.text(
            60, 51, "", color="yellow", fontsize=13, fontweight="bold",
            ha="center", va="top", zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", alpha=0.75)
        )

        # Frame counter
        self.frame_counter = self.ax.text(
            1, 52, "", color="white", fontsize=8, ha="left", va="top", zorder=10
        )

        # Scatter plots — one per team + ball
        self.scat_off  = self.ax.scatter([], [], s=180, zorder=5, edgecolors="white", linewidths=0.8)
        self.scat_def  = self.ax.scatter([], [], s=180, zorder=5, edgecolors="white", linewidths=0.8)
        self.scat_ball = self.ax.scatter([], [], s=90,  zorder=6, color=FOOTBALL_COLOR,
                                         edgecolors=BALL_COLOR, linewidths=1.5)

        # Jersey number texts — pre-allocate up to 22 players
        self.texts = [self.ax.text(0, 0, "", color="white", fontsize=5.5,
                                   ha="center", va="center", fontweight="bold", zorder=7)
                      for _ in range(23)]

        # Velocity arrows (quiver)
        self.quiver_off = self.ax.quiver([], [], [], [], color=OFFENSE_COLOR,
                                          alpha=0.5, scale=25, width=0.003, zorder=4)
        self.quiver_def = self.ax.quiver([], [], [], [], color=DEFENSE_COLOR,
                                          alpha=0.5, scale=25, width=0.003, zorder=4)

        # Legend
        self._draw_legend()

        # Keyboard handler
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        interval = 50  # ms between frames (~20 fps)
        self.anim = animation.FuncAnimation(
            self.fig, self._update,
            frames=len(self.frames),
            interval=interval,
            blit=False,
            repeat=not self.save
        )

        if self.save:
            out_path = f"play_{self.game_id}_{self.play_id}.gif"
            print(f"Saving animation to {out_path} …")
            writer = animation.PillowWriter(fps=20)
            self.anim.save(out_path, writer=writer)
            print("Done.")
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_player_colors(self, play_df):
        """Return {nflId: color_hex}."""
        colors = {}
        for nfl_id in play_df["nflId"].dropna().unique():
            team_rows = play_df[play_df["nflId"] == nfl_id]["team"]
            if team_rows.empty:
                continue
            team = team_rows.iloc[0]
            if team == "football":
                colors[nfl_id] = FOOTBALL_COLOR
            elif team == self.offense_team:
                colors[nfl_id] = OFFENSE_COLOR
            else:
                colors[nfl_id] = DEFENSE_COLOR
        return colors

    def _build_title(self):
        if self.meta is None:
            return f"Game {self.game_id}  |  Play {self.play_id}"
        m = self.meta
        try:
            desc = str(m["playDescription"])[:90]
            return (f"Game {self.game_id}  |  Play {self.play_id}  |  "
                    f"Q{int(m['quarter'])}  {m['gameClock']}  |  "
                    f"{m['possessionTeam']} vs {m['defensiveTeam']}  |  "
                    f"D{int(m['down'])} & {int(m['yardsToGo'])}\n{desc}")
        except Exception:
            return f"Game {self.game_id}  |  Play {self.play_id}"

    def _draw_legend(self):
        handles = [
            mpatches.Patch(color=OFFENSE_COLOR, label=f"Offense ({self.offense_team or '?'})"),
            mpatches.Patch(color=DEFENSE_COLOR, label=f"Defense ({self.defense_team or '?'})"),
            mpatches.Patch(color=FOOTBALL_COLOR, label="Football"),
        ]
        leg = self.ax.legend(handles=handles, loc="lower left",
                             fontsize=8, framealpha=0.7,
                             facecolor="#1a1a2e", labelcolor="white")

    def _update(self, i):
        if self.paused:
            i = self.cur_frame
        else:
            self.cur_frame = i

        frame_id = self.frames[i]
        df_frame = self.frame_map[frame_id]

        # Separate ball vs players
        ball_rows = df_frame[df_frame["team"] == "football"]
        off_rows  = df_frame[df_frame["team"] == self.offense_team] if self.offense_team else pd.DataFrame()
        def_rows  = df_frame[~df_frame["team"].isin(["football", self.offense_team])]

        # Update scatter
        if not off_rows.empty:
            self.scat_off.set_offsets(off_rows[["x", "y"]].values)
            self.scat_off.set_color(OFFENSE_COLOR)

        if not def_rows.empty:
            self.scat_def.set_offsets(def_rows[["x", "y"]].values)
            self.scat_def.set_color(DEFENSE_COLOR)

        if not ball_rows.empty:
            self.scat_ball.set_offsets(ball_rows[["x", "y"]].values)

        # Velocity arrows
        def _quiver_data(rows):
            if rows.empty:
                return np.array([]), np.array([]), np.array([]), np.array([])
            x = rows["x"].values
            y = rows["y"].values
            # NGS convention: dir is degrees clockwise from +y (the sideline direction),
            # so the x component is sin(dir) and the y component is cos(dir)
            d_rad = np.deg2rad(rows["dir"].fillna(0).values)
            s = rows["s"].fillna(0).values
            u = s * np.sin(d_rad)
            v = s * np.cos(d_rad)
            return x, y, u, v

        x, y, u, v = _quiver_data(off_rows)
        self.quiver_off.set_offsets(np.column_stack([x, y]) if len(x) else np.empty((0, 2)))
        self.quiver_off.set_UVC(u, v)

        x, y, u, v = _quiver_data(def_rows)
        self.quiver_def.set_offsets(np.column_stack([x, y]) if len(x) else np.empty((0, 2)))
        self.quiver_def.set_UVC(u, v)

        # Jersey numbers
        player_rows = pd.concat([off_rows, def_rows]) if not off_rows.empty else def_rows
        for j, (_, row) in enumerate(player_rows.iterrows()):
            if j >= len(self.texts):
                break
            jnum = str(int(row["jerseyNumber"])) if pd.notna(row["jerseyNumber"]) else ""
            self.texts[j].set_position((row["x"], row["y"]))
            self.texts[j].set_text(jnum)
        # Clear unused text slots
        for j in range(len(player_rows), len(self.texts)):
            self.texts[j].set_text("")

        # Frame counter
        self.frame_counter.set_text(f"Frame {frame_id}/{self.frames[-1]}")

        # Event banner
        event = self.event_frames.get(frame_id, "")
        if event and event != "None":
            display = event.replace("autoevent_", "").replace("_", " ").upper()
            color = KEY_EVENT_COLORS.get(event, "yellow")
            self.event_banner.set_text(f"● {display}")
            self.event_banner.set_color(color)
        else:
            # Fade after a few frames
            prev_events = {f: e for f, e in self.event_frames.items() if f <= frame_id}
            if prev_events:
                last_ev_frame = max(prev_events.keys())
                if frame_id - last_ev_frame < 8:
                    pass  # keep showing
                else:
                    self.event_banner.set_text("")
            else:
                self.event_banner.set_text("")

        return (self.scat_off, self.scat_def, self.scat_ball,
                self.event_banner, self.frame_counter, *self.texts)

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "right" and self.paused:
            self.cur_frame = min(self.cur_frame + 1, len(self.frames) - 1)
            self._update(self.cur_frame)
            self.fig.canvas.draw()
        elif event.key == "left" and self.paused:
            self.cur_frame = max(self.cur_frame - 1, 0)
            self._update(self.cur_frame)
            self.fig.canvas.draw()
        elif event.key == "q":
            plt.close(self.fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def list_plays(game_id: int, tracking: pd.DataFrame, plays_df: pd.DataFrame):
    """Print available plays for a game so the user can pick one."""
    play_ids = sorted(tracking["playId"].unique())
    print(f"\nAvailable plays for game {game_id}:")
    for pid in play_ids:
        row = plays_df[(plays_df["gameId"] == game_id) & (plays_df["playId"] == pid)]
        if not row.empty:
            r = row.iloc[0]
            desc = str(r["playDescription"])[:70]
            print(f"  {pid:>5}  Q{int(r['quarter'])} {r['gameClock']}  D{int(r['down'])}&{int(r['yardsToGo'])}  {desc}")
        else:
            print(f"  {pid:>5}")
    print()


def main():
    parser = argparse.ArgumentParser(description="NFL Play Animator")
    parser.add_argument("--game",  type=int,  default=2021090900, help="gameId")
    parser.add_argument("--play",  type=int,  default=97,         help="playId")
    parser.add_argument("--list",  action="store_true",           help="list all plays in the game")
    parser.add_argument("--save",  action="store_true",           help="save animation as GIF")
    args = parser.parse_args()

    print(f"Loading tracking data for game {args.game} …")
    tracking = load_tracking(args.game)

    plays, players, games, pff = load_support_data()

    if args.list:
        list_plays(args.game, tracking, plays)
        return

    print(f"Loading play {args.play} …")
    play_df = get_play_data(tracking, args.game, args.play)

    # Get play metadata
    meta_rows = plays[(plays["gameId"] == args.game) & (plays["playId"] == args.play)]
    meta = meta_rows.iloc[0] if not meta_rows.empty else None

    if meta is not None:
        print(f"\nPlay description: {meta['playDescription']}")
        print(f"Formation: {meta['offenseFormation']}  |  Coverage: {meta['pff_passCoverage']} ({meta['pff_passCoverageType']})")
        print(f"Personnel O: {meta['personnelO']}  |  Personnel D: {meta['personnelD']}")

    # Print key events in this play
    ev_frames = get_event_frames(play_df)
    if ev_frames:
        print(f"\nKey events:")
        for fid, ev in sorted(ev_frames.items()):
            print(f"  Frame {fid:>3}: {ev}")

    print(f"\nTotal frames: {len(sorted(play_df['frameId'].unique()))}")
    print("Opening animation… (Space=pause, ←/→=step, q=quit)\n")

    animator = PlayAnimator(args.game, args.play, meta, players, pff, save=args.save)
    animator.run(play_df)


if __name__ == "__main__":
    main()
