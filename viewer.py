# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
Interactive Residual Stream Trajectory Viewer (3D)

Shows full trajectories (all layers) as 3D paths with directional arrows.
Use arrow keys to cycle through dimension triplets.

Controls:
    Left/Right Arrow  — cycle through dimension triplets
    Up/Down Arrow     — change which dim is on Z axis (rotate triplet)
    Space             — play/pause animation through layers
    1-9               — highlight specific trajectory
    0                 — clear highlight
    C                 — toggle centroids
    F                 — find best dims (auto-detect convergence dims)
    R                 — reset
    P                 — save current frame as PNG
    Q / Escape / Ctrl+C — quit

    Mouse drag        — trackball rotation (natural)
    Scroll            — zoom

Usage:
    python3 residual_attractors.py attractor_data/berlin_multilingual
    python3 residual_attractors.py attractor_data/
    python3 residual_attractors.py attractor_data/ --groups berlin_multilingual paris_multilingual
    python3 residual_attractors.py attractor_data/berlin_multilingual --dims 47,203,512
"""

import sys
import os
import shutil
import subprocess
import signal
import argparse
from pathlib import Path

# =============================================================================
# Auto-restart under `uv run`
# =============================================================================

def _ensure_uv_run():
    if os.environ.get("_UV_RUN_ACTIVE") == "1":
        return
    uv_path = shutil.which("uv")
    if uv_path is None:
        print("ERROR: needs `uv`. Install: curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    script_path = os.path.abspath(__file__)
    cmd = [uv_path, "run", script_path] + sys.argv[1:]
    env = os.environ.copy()
    env["_UV_RUN_ACTIVE"] = "1"
    if sys.platform == "win32":
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    else:
        os.execvpe(uv_path, cmd, env)

_ensure_uv_run()

# =============================================================================
# Force Ctrl+C to kill immediately, regardless of window focus
# =============================================================================
signal.signal(signal.SIGINT, lambda *_: os._exit(0))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Data Loader
# =============================================================================

class TrajectoryData:
    """Loads residual stream trajectory data from CSV output."""

    def __init__(self, data_dir: Path, groups: list[str] = None):
        self.data_dir = data_dir
        self.trajectories = {}
        self.prompts = {}
        self.predictions = {}
        self.n_layers = 0
        self.d_model = 0
        self._load(groups)

    def _load(self, groups: list[str] = None):
        data_dir = self.data_dir

        if (data_dir / "final_token_streams").is_dir():
            group_name = data_dir.name
            self._load_group(group_name, data_dir)
        else:
            if groups:
                candidates = [data_dir / g for g in groups]
            else:
                candidates = sorted([
                    d for d in data_dir.iterdir()
                    if d.is_dir() and d.name != "visualizations"
                ])

            for group_dir in candidates:
                if not group_dir.is_dir():
                    continue
                if (group_dir / "final_token_streams").is_dir():
                    self._load_group(group_dir.name, group_dir)
                else:
                    print(f"  Skipping {group_dir.name}: no final_token_streams/")

        if not self.trajectories:
            print(f"\nERROR: No trajectory data found in {data_dir}")
            sys.exit(1)

        first_group = list(self.trajectories.keys())[0]
        first_traj = self.trajectories[first_group][0]
        self.n_layers, self.d_model = first_traj.shape

        total = sum(len(v) for v in self.trajectories.values())
        print(f"\nLoaded: {len(self.trajectories)} group(s), {total} trajectories")
        print(f"  Layers: {self.n_layers}, Dimensions: {self.d_model}")
        for name, trajs in self.trajectories.items():
            preds = self.predictions.get(name, [])
            pred_str = ", ".join(preds[:5])
            if len(preds) > 5:
                pred_str += "..."
            print(f"  {name}: {len(trajs)} prompts → [{pred_str}]")

    def _load_group(self, group_name: str, group_dir: Path):
        final_dir = group_dir / "final_token_streams"
        all_csv = final_dir / "all_layers_all_prompts.csv"
        if all_csv.exists():
            self._load_combined_csv(group_name, all_csv)
        else:
            self._load_layer_csvs(group_name, final_dir)

        meta_csv = group_dir / "prompts_meta.csv"
        if meta_csv.exists():
            self._load_meta(group_name, meta_csv)

    def _load_combined_csv(self, group_name: str, csv_path: Path):
        import csv
        print(f"  Loading {group_name} from combined CSV...")
        trajectories = {}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            dim_keys = None
            for row in reader:
                if dim_keys is None:
                    dim_keys = sorted([k for k in row.keys() if k.startswith("dim_")])
                layer = int(row["layer"])
                prompt_idx = int(row["prompt_idx"])
                vec = np.array([float(row[k]) for k in dim_keys], dtype=np.float32)
                if prompt_idx not in trajectories:
                    trajectories[prompt_idx] = {}
                trajectories[prompt_idx][layer] = vec

        result = []
        for pidx in sorted(trajectories.keys()):
            layers_dict = trajectories[pidx]
            n_layers = max(layers_dict.keys()) + 1
            d_model = len(dim_keys)
            arr = np.zeros((n_layers, d_model), dtype=np.float32)
            for l, vec in layers_dict.items():
                arr[l] = vec
            result.append(arr)
        self.trajectories[group_name] = result

    def _load_layer_csvs(self, group_name: str, final_dir: Path):
        import csv
        layer_files = sorted(final_dir.glob("layer_*.csv"))
        if not layer_files:
            print(f"  WARNING: No layer CSVs in {final_dir}")
            return

        print(f"  Loading {group_name} from {len(layer_files)} layer CSVs...")
        with open(layer_files[0], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        n_prompts = len(rows)
        dim_keys = sorted([k for k in rows[0].keys() if k.startswith("dim_")])
        d_model = len(dim_keys)
        n_layers = len(layer_files)

        all_trajs = [np.zeros((n_layers, d_model), dtype=np.float32) for _ in range(n_prompts)]
        for layer_idx, layer_file in enumerate(layer_files):
            with open(layer_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for prompt_idx, row in enumerate(reader):
                    vec = np.array([float(row[k]) for k in dim_keys], dtype=np.float32)
                    all_trajs[prompt_idx][layer_idx] = vec
        self.trajectories[group_name] = all_trajs

    def _load_meta(self, group_name: str, meta_path: Path):
        import csv
        prompts = []
        predictions = []
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(row.get("prompt", ""))
                predictions.append(row.get("predicted_next_token", ""))
        self.prompts[group_name] = prompts
        self.predictions[group_name] = predictions

    def get_all_trajectories(self) -> list[tuple[str, int, np.ndarray]]:
        result = []
        for name, trajs in self.trajectories.items():
            for i, traj in enumerate(trajs):
                result.append((name, i, traj))
        return result

    def find_interesting_dims(self, top_k: int = 30) -> list[int]:
        all_final = []
        for trajs in self.trajectories.values():
            for traj in trajs:
                all_final.append(traj[-1])
        X = np.stack(all_final)
        variances = X.var(axis=0)
        return np.argsort(variances)[::-1][:top_k].tolist()

    def find_convergence_dims(self, top_k: int = 30) -> list[int]:
        all_first = []
        all_last = []
        for trajs in self.trajectories.values():
            for traj in trajs:
                all_first.append(traj[0])
                all_last.append(traj[-1])
        var_first = np.stack(all_first).var(axis=0)
        var_last = np.stack(all_last).var(axis=0)
        ratio = var_first / (var_last + 1e-10)
        return np.argsort(ratio)[::-1][:top_k].tolist()


# =============================================================================
# Trackball-style 3D rotation (natural mouse interaction)
# =============================================================================

class Trackball:
    """
    Replaces matplotlib's default 3D mouse rotation with a more natural
    trackball-style interaction:
    - Horizontal drag rotates around the vertical (Z) axis
    - Vertical drag rotates around the horizontal axis
    - Movement is proportional and in the expected direction
    """

    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self._dragging = False
        self._last_x = 0
        self._last_y = 0

        # Disable matplotlib's built-in 3D mouse rotation
        # by disconnecting its button_press/release/motion handlers
        self.ax.disable_mouse_rotation()

        # Connect our own handlers
        self._cid_press = fig.canvas.mpl_connect('button_press_event', self._on_press)
        self._cid_release = fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._cid_motion = fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self._cid_scroll = fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left click
            self._dragging = True
            self._last_x = event.x
            self._last_y = event.y

    def _on_release(self, event):
        if event.button == 1:
            self._dragging = False

    def _on_motion(self, event):
        if not self._dragging:
            return
        if event.x is None or event.y is None:
            return

        dx = event.x - self._last_x
        dy = event.y - self._last_y
        self._last_x = event.x
        self._last_y = event.y

        # Sensitivity
        sensitivity = 0.5

        # Horizontal mouse movement -> azimuth rotation (natural: drag right = rotate right)
        azim = self.ax.azim - dx * sensitivity
        # Vertical mouse movement -> elevation (natural: drag up = look up)
        elev = self.ax.elev + dy * sensitivity

        # Clamp elevation to avoid flipping
        elev = max(-90, min(90, elev))

        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        """Zoom with scroll wheel."""
        if event.inaxes != self.ax:
            return
        # Get current axis limits
        factor = 0.9 if event.button == 'up' else 1.1

        for getter, setter in [(self.ax.get_xlim, self.ax.set_xlim),
                               (self.ax.get_ylim, self.ax.set_ylim),
                               (self.ax.get_zlim, self.ax.set_zlim)]:
            lo, hi = getter()
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * factor
            setter(mid - half, mid + half)

        self.fig.canvas.draw_idle()


# =============================================================================
# 3D Interactive Viewer
# =============================================================================

class Viewer3D:
    """3D viewer: shows full trajectories with arrows. Arrow keys cycle dims."""

    def __init__(self, data: TrajectoryData, initial_dims: list[int] = None):
        self.data = data
        self.n_layers = data.n_layers
        self.d_model = data.d_model

        # State
        self.show_centroids = True
        self.highlighted_idx = None
        self.arrow_step = max(1, data.n_layers // 10)  # Show ~10 arrows per trajectory

        # Build ALL dimension triplet combinations from interesting dims
        interesting = data.find_interesting_dims(top_k=15)
        self.dim_combos = []
        # Generate all unique triplets (combinations, not permutations — order matters for axes)
        from itertools import combinations
        for combo in combinations(interesting[:15], 3):
            self.dim_combos.append(list(combo))
        if not self.dim_combos:
            self.dim_combos = [[0, 1, 2]]

        if initial_dims and len(initial_dims) >= 3:
            self.dims = initial_dims[:3]
            if self.dims not in self.dim_combos:
                self.dim_combos.insert(0, self.dims)
            self.combo_idx = self.dim_combos.index(self.dims)
        else:
            self.combo_idx = 0
            self.dims = self.dim_combos[0]

        # Colors
        group_names = list(data.trajectories.keys())
        cmap = plt.cm.tab10
        self.group_colors = {}
        for i, name in enumerate(group_names):
            self.group_colors[name] = cmap(i % 10)

        # Flat trajectory list
        self.flat_trajs = data.get_all_trajectories()

        # Build figure
        self._build()

    def _build(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title(
            "Residual Stream 3D Viewer — Arrow keys to cycle dims"
        )

        self.ax = self.fig.add_subplot(111, projection='3d')

        # Install trackball rotation
        self.trackball = Trackball(self.ax, self.fig)

        # Connect keyboard
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', lambda _: os._exit(0))

        self._draw()

    def _on_key(self, event):
        if event.key == 'right':
            self.combo_idx = (self.combo_idx + 1) % len(self.dim_combos)
            self.dims = self.dim_combos[self.combo_idx]
            self._draw()
        elif event.key == 'left':
            self.combo_idx = (self.combo_idx - 1) % len(self.dim_combos)
            self.dims = self.dim_combos[self.combo_idx]
            self._draw()
        elif event.key == 'up':
            # Rotate the triplet: [a,b,c] -> [b,c,a]
            self.dims = [self.dims[1], self.dims[2], self.dims[0]]
            # Update in combos list
            self.dim_combos[self.combo_idx] = self.dims
            self._draw()
        elif event.key == 'down':
            # Rotate the triplet: [a,b,c] -> [c,a,b]
            self.dims = [self.dims[2], self.dims[0], self.dims[1]]
            self.dim_combos[self.combo_idx] = self.dims
            self._draw()
        elif event.key == 'c':
            self.show_centroids = not self.show_centroids
            self._draw()
        elif event.key == 'f':
            from itertools import combinations
            conv_dims = self.data.find_convergence_dims(top_k=15)
            self.dim_combos = []
            for combo in combinations(conv_dims[:15], 3):
                self.dim_combos.append(list(combo))
            self.combo_idx = 0
            self.dims = self.dim_combos[0]
            print(f"  Switched to convergence dims. {len(self.dim_combos)} combos available.")
            print(f"  Starting with: {self.dims}")
            self._draw()
        elif event.key == 'r':
            self.combo_idx = 0
            self.dims = self.dim_combos[0]
            self.highlighted_idx = None
            self._draw()
        elif event.key == 'p':
            fname = f"frame_d{self.dims[0]}_{self.dims[1]}_{self.dims[2]}.png"
            self.fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved: {fname}")
        elif event.key in '123456789':
            idx = int(event.key) - 1
            if idx < len(self.flat_trajs):
                self.highlighted_idx = idx
                gname, pidx, _ = self.flat_trajs[idx]
                prompt = self.data.prompts.get(gname, [""])[pidx] if gname in self.data.prompts and pidx < len(self.data.prompts[gname]) else "?"
                pred = self.data.predictions.get(gname, [""])[pidx] if gname in self.data.predictions and pidx < len(self.data.predictions[gname]) else "?"
                print(f"  Highlighted #{idx+1}: [{gname}] '{prompt}' → '{pred}'")
                self._draw()
        elif event.key == '0':
            self.highlighted_idx = None
            self._draw()
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)
            os._exit(0)

    def _draw(self):
        """Redraw the 3D plot with full trajectories and arrows."""
        elev = self.ax.elev
        azim = self.ax.azim

        self.ax.clear()

        d0, d1, d2 = self.dims[0], self.dims[1], self.dims[2]

        # Plot each group's full trajectories
        for group_name, trajs in self.data.trajectories.items():
            color = self.group_colors[group_name]

            for traj_idx, traj in enumerate(trajs):
                x = traj[:, d0]
                y = traj[:, d1]
                z = traj[:, d2]

                # Draw the full trajectory line
                self.ax.plot(x, y, z, color=color, alpha=0.4, linewidth=1.0)

                # Start marker (first layer)
                self.ax.scatter([x[0]], [y[0]], [z[0]], c=[color], s=20,
                              marker='o', alpha=0.4, edgecolors='none')

                # End marker (last layer) — bigger
                self.ax.scatter([x[-1]], [y[-1]], [z[-1]], c=[color], s=60,
                              marker='o', alpha=0.9, edgecolors='k', linewidths=0.5)

                # Direction arrows (quiver) at regular intervals
                for i in range(0, len(x) - 1, self.arrow_step):
                    dx_arr = x[i+1] - x[i]
                    dy_arr = y[i+1] - y[i]
                    dz_arr = z[i+1] - z[i]

                    # Scale arrow length for visibility
                    length = np.sqrt(dx_arr**2 + dy_arr**2 + dz_arr**2)
                    if length > 1e-8:
                        # Fade arrows: earlier = more transparent
                        alpha = 0.2 + 0.6 * (i / max(len(x) - 1, 1))
                        self.ax.quiver(
                            x[i], y[i], z[i],
                            dx_arr, dy_arr, dz_arr,
                            color=color, alpha=alpha,
                            arrow_length_ratio=0.3,
                            linewidth=0.8
                        )

            # Token labels at final position
            predictions = self.data.predictions.get(group_name, [])
            for traj_idx, traj in enumerate(trajs):
                if traj_idx < len(predictions) and predictions[traj_idx]:
                    self.ax.text(
                        traj[-1, d0], traj[-1, d1], traj[-1, d2],
                        f" {predictions[traj_idx]}",
                        fontsize=7, alpha=0.8, color=color,
                        fontweight='bold'
                    )

            # Centroid at final layer
            if self.show_centroids:
                final_points = np.stack([t[-1] for t in trajs])
                cx = final_points[:, d0].mean()
                cy = final_points[:, d1].mean()
                cz = final_points[:, d2].mean()
                self.ax.scatter([cx], [cy], [cz], c=[color], s=300, marker="*",
                              edgecolors="k", linewidths=1.5, zorder=10,
                              label=group_name)

        # Highlighted trajectory
        if self.highlighted_idx is not None and self.highlighted_idx < len(self.flat_trajs):
            gname, pidx, traj = self.flat_trajs[self.highlighted_idx]
            x = traj[:, d0]
            y = traj[:, d1]
            z = traj[:, d2]

            self.ax.plot(x, y, z, "r-", linewidth=3.0, alpha=0.95, zorder=20)

            # Arrows on highlighted
            for i in range(0, len(x) - 1, max(1, self.arrow_step // 2)):
                dx_arr = x[i+1] - x[i]
                dy_arr = y[i+1] - y[i]
                dz_arr = z[i+1] - z[i]
                length = np.sqrt(dx_arr**2 + dy_arr**2 + dz_arr**2)
                if length > 1e-8:
                    self.ax.quiver(
                        x[i], y[i], z[i],
                        dx_arr, dy_arr, dz_arr,
                        color='red', alpha=0.8,
                        arrow_length_ratio=0.35,
                        linewidth=1.5, zorder=21
                    )

            # Start = green triangle, End = red diamond
            self.ax.scatter([x[0]], [y[0]], [z[0]], c="green", s=100,
                          marker="^", edgecolors="k", linewidths=1.5, zorder=22)
            self.ax.scatter([x[-1]], [y[-1]], [z[-1]], c="red", s=150,
                          marker="D", edgecolors="k", linewidths=1.5, zorder=22)

        # Labels
        self.ax.set_xlabel(f"dim {d0}", fontsize=10)
        self.ax.set_ylabel(f"dim {d1}", fontsize=10)
        self.ax.set_zlabel(f"dim {d2}", fontsize=10)

        n_groups = len(self.data.trajectories)
        n_total = len(self.flat_trajs)
        title = (f"dims [{d0}, {d1}, {d2}]  |  "
                f"combo {self.combo_idx+1}/{len(self.dim_combos)}  |  "
                f"{n_groups} group(s), {n_total} trajectories\n"
                f"←/→ = change dims   ↑/↓ = rotate triplet   "
                f"F = convergence dims   1-9 = highlight   drag = rotate")

        self.ax.set_title(title, fontsize=9, fontfamily="monospace")

        if self.show_centroids:
            self.ax.legend(loc="upper left", fontsize=8)

        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def run(self):
        print("\n" + "=" * 65)
        print("  3D RESIDUAL STREAM TRAJECTORY VIEWER")
        print("=" * 65)
        print(f"  Layers: {self.n_layers}  |  Dimensions: {self.d_model}")
        print(f"  Dimension combos available: {len(self.dim_combos)}")
        print(f"  Current dims: {self.dims}")
        print()
        print("  CONTROLS:")
        print("    ← / →        Cycle dimension triplets")
        print("    ↑ / ↓        Rotate axes within current triplet")
        print("    F            Switch to convergence dims")
        print("    C            Toggle centroids")
        print("    1-9          Highlight trajectory")
        print("    0            Clear highlight")
        print("    P            Save PNG")
        print("    R            Reset")
        print("    Q / Esc      Quit")
        print("    Ctrl+C       Force quit (works even when unfocused)")
        print()
        print("  MOUSE:")
        print("    Left drag    Trackball rotation (natural)")
        print("    Scroll       Zoom in/out")
        print("=" * 65 + "\n")
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D Residual Stream Trajectory Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 residual_attractors.py attractor_data/berlin_multilingual
  python3 residual_attractors.py attractor_data/
  python3 residual_attractors.py attractor_data/ --groups berlin_multilingual paris_multilingual
  python3 residual_attractors.py attractor_data/berlin_multilingual --dims 47,203,512
        """
    )
    parser.add_argument("data_dir", type=str,
                       help="Path to group directory or parent directory containing groups")
    parser.add_argument("--groups", type=str, nargs="*", default=None,
                       help="Specific groups to load (when data_dir is parent)")
    parser.add_argument("--dims", type=str, default=None,
                       help="Initial dimension triplet, comma-separated (e.g. '47,203,512')")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Path not found: {data_dir}")
        sys.exit(1)

    initial_dims = None
    if args.dims:
        initial_dims = [int(d.strip()) for d in args.dims.split(",")]
        if len(initial_dims) < 3:
            print(f"ERROR: Need at least 3 dimensions, got {len(initial_dims)}")
            sys.exit(1)

    data = TrajectoryData(data_dir, groups=args.groups)
    viewer = Viewer3D(data, initial_dims=initial_dims)
    viewer.run()


if __name__ == "__main__":
    main()
