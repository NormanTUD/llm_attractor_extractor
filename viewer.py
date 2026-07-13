# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
Interactive Residual Stream Trajectory Explorer

Controls:
    Left/Right Arrow  — change dimension pair (cycle through dim combinations)
    Up/Down Arrow     — step through layers (time dimension)
    Space             — play/pause animation through layers
    T                 — toggle between 2D and 3D view
    G                 — toggle group coloring vs trajectory coloring
    P                 — toggle showing all points vs single trajectory
    +/-               — zoom in/out
    R                 — reset view
    S                 — save current frame as PNG
    Q                 — quit
    1-9               — select specific trajectory to highlight
    D                 — enter dimension selection mode (type dim indices)
    A                 — auto-detect interesting dimensions (high variance)
    C                 — show centroid trajectory
    H                 — show convex hull of final-layer points

Usage:
    uv run explore_attractors.py attractor_data/paris_multilingual
    uv run explore_attractors.py attractor_data/  --all-groups
    uv run explore_attractors.py attractor_data/ --dims 47,203,1891
"""

import sys
import os
import shutil
import subprocess
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

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Interactive backend!
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Data Loader
# =============================================================================

class TrajectoryData:
    """Loads and manages residual stream trajectory data."""

    def __init__(self, data_dir: Path, groups: list[str] = None):
        self.data_dir = data_dir
        self.trajectories = {}  # group_name -> list of (n_layers, d_model) arrays
        self.prompts = {}       # group_name -> list of prompt strings
        self.predictions = {}   # group_name -> list of predicted tokens
        self.n_layers = 0
        self.d_model = 0

        self._load(groups)

    def _load(self, groups: list[str] = None):
        """Load trajectory data from the output of residual_attractors.py"""
        data_dir = self.data_dir

        # Detect available groups
        if groups:
            group_dirs = [data_dir / g for g in groups if (data_dir / g).is_dir()]
        else:
            group_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name != "visualizations"]

        if not group_dirs:
            print(f"ERROR: No group directories found in {data_dir}")
            sys.exit(1)

        for group_dir in sorted(group_dirs):
            group_name = group_dir.name
            final_dir = group_dir / "final_token_streams"

            if not final_dir.exists():
                print(f"  Skipping {group_name}: no final_token_streams/ directory")
                continue

            # Load the combined CSV or reconstruct from per-layer CSVs
            all_csv = final_dir / "all_layers_all_prompts.csv"
            if all_csv.exists():
                self._load_from_combined_csv(group_name, all_csv)
            else:
                self._load_from_layer_csvs(group_name, final_dir)

            # Load metadata
            meta_csv = group_dir / "prompts_meta.csv"
            if meta_csv.exists():
                self._load_meta(group_name, meta_csv)

        if not self.trajectories:
            print("ERROR: No trajectory data loaded!")
            sys.exit(1)

        # Determine dimensions
        first_group = list(self.trajectories.keys())[0]
        first_traj = self.trajectories[first_group][0]
        self.n_layers, self.d_model = first_traj.shape

        total_trajs = sum(len(v) for v in self.trajectories.values())
        print(f"Loaded: {len(self.trajectories)} groups, {total_trajs} trajectories")
        print(f"  n_layers={self.n_layers}, d_model={self.d_model}")
        for name, trajs in self.trajectories.items():
            print(f"  {name}: {len(trajs)} prompts")

    def _load_from_combined_csv(self, group_name: str, csv_path: Path):
        """Load from the all_layers_all_prompts.csv format."""
        import csv

        trajectories = {}  # prompt_idx -> list of layer vectors

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = int(row["layer"])
                prompt_idx = int(row["prompt_idx"])

                # Extract dimension values
                dims = []
                for key in sorted(row.keys()):
                    if key.startswith("dim_"):
                        dims.append(float(row[key]))

                if prompt_idx not in trajectories:
                    trajectories[prompt_idx] = {}
                trajectories[prompt_idx][layer] = np.array(dims, dtype=np.float32)

        # Convert to (n_layers, d_model) arrays
        result = []
        for prompt_idx in sorted(trajectories.keys()):
            layers_dict = trajectories[prompt_idx]
            n_layers = max(layers_dict.keys()) + 1
            d_model = len(list(layers_dict.values())[0])
            arr = np.zeros((n_layers, d_model), dtype=np.float32)
            for l, vec in layers_dict.items():
                arr[l] = vec
            result.append(arr)

        self.trajectories[group_name] = result

    def _load_from_layer_csvs(self, group_name: str, final_dir: Path):
        """Load from individual layer_XXX.csv files."""
        import csv

        layer_files = sorted(final_dir.glob("layer_*.csv"))
        if not layer_files:
            return

        n_layers = len(layer_files)

        # Read first file to get n_prompts and d_model
        with open(layer_files[0], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        n_prompts = len(rows)
        dim_keys = sorted([k for k in rows[0].keys() if k.startswith("dim_")])
        d_model = len(dim_keys)

        # Allocate arrays
        all_trajs = [np.zeros((n_layers, d_model), dtype=np.float32) for _ in range(n_prompts)]

        for layer_idx, layer_file in enumerate(layer_files):
            with open(layer_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for prompt_idx, row in enumerate(reader):
                    vec = np.array([float(row[k]) for k in dim_keys], dtype=np.float32)
                    all_trajs[prompt_idx][layer_idx] = vec

        self.trajectories[group_name] = all_trajs

    def _load_meta(self, group_name: str, meta_path: Path):
        """Load prompt metadata."""
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
        """Return flat list of (group_name, prompt_idx, trajectory_array)."""
        result = []
        for name, trajs in self.trajectories.items():
            for i, traj in enumerate(trajs):
                result.append((name, i, traj))
        return result

    def find_interesting_dims(self, top_k: int = 20) -> list[int]:
        """Find dimensions with highest variance across all final-layer points."""
        all_final = []
        for trajs in self.trajectories.values():
            for traj in trajs:
                all_final.append(traj[-1])  # last layer

        X = np.stack(all_final)
        variances = X.var(axis=0)
        top_dims = np.argsort(variances)[::-1][:top_k]
        return top_dims.tolist()

    def find_attractor_dims(self, group_name: str, top_k: int = 10) -> list[int]:
        """Find dims where this group converges most (variance decreases most from first to last layer)."""
        trajs = self.trajectories[group_name]
        first_layer = np.stack([t[0] for t in trajs])
        last_layer = np.stack([t[-1] for t in trajs])

        var_first = first_layer.var(axis=0)
        var_last = last_layer.var(axis=0)

        # Convergence ratio: how much variance decreased
        convergence = var_first / (var_last + 1e-10)
        top_dims = np.argsort(convergence)[::-1][:top_k]
        return top_dims.tolist()


# =============================================================================
# Interactive Explorer
# =============================================================================

class TrajectoryExplorer:
    """Interactive matplotlib-based trajectory explorer."""

    def __init__(self, data: TrajectoryData, initial_dims: list[int] = None):
        self.data = data
        self.n_layers = data.n_layers
        self.d_model = data.d_model

        # State
        self.current_layer = data.n_layers - 1  # Start at final layer
        self.mode_3d = False
        self.playing = False
        self.play_speed = 1  # layers per frame
        self.show_all = True  # show all trajectories
        self.show_trails = True
        self.trail_length = 5
        self.highlighted_idx = None  # index into flat trajectory list
        self.show_centroids = True
        self.show_hull = False

        # Dimension selection
        if initial_dims:
            self.dims = initial_dims[:3]
        else:
            interesting = data.find_interesting_dims(top_k=50)
            self.dims = interesting[:3]  # Start with top-3 variance dims

        # Pre-compute dimension combinations for cycling
        interesting = data.find_interesting_dims(top_k=30)
        self.dim_combos = []
        for i in range(0, len(interesting) - 2, 2):
            self.dim_combos.append(interesting[i:i+3])
        if not self.dim_combos:
            self.dim_combos = [[0, 1, 2]]
        self.combo_idx = 0

        # Colors per group
        group_names = list(data.trajectories.keys())
        cmap = plt.cm.tab10
        self.group_colors = {name: cmap(i / max(len(group_names), 1))
                           for i, name in enumerate(group_names)}

        # Flat trajectory list for indexing
        self.flat_trajs = data.get_all_trajectories()

        # Setup figure
        self._setup_figure()

    def _setup_figure(self):
        """Create the matplotlib figure with controls."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title("Residual Stream Trajectory Explorer")

        # Main plot area
        if self.mode_3d:
            self.ax = self.fig.add_axes([0.05, 0.15, 0.7, 0.8], projection='3d')
        else:
            self.ax = self.fig.add_axes([0.05, 0.15, 0.7, 0.8])

        # Info panel on the right
        self.info_ax = self.fig.add_axes([0.78, 0.15, 0.2, 0.8])
        self.info_ax.axis("off")

        # Layer slider at bottom
        slider_ax = self.fig.add_axes([0.1, 0.03, 0.5, 0.03])
        self.layer_slider = Slider(slider_ax, 'Layer', 0, self.n_layers - 1,
                                   valinit=self.current_layer, valstep=1)
        self.layer_slider.on_changed(self._on_slider_change)

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Initial draw
        self._draw()

    def _on_slider_change(self, val):
        self.current_layer = int(val)
        self._draw()

    def _on_key(self, event):
        if event.key == 'right':
            # Next dimension combination
            self.combo_idx = (self.combo_idx + 1) % len(self.dim_combos)
            self.dims = self.dim_combos[self.combo_idx]
            self._draw()
        elif event.key == 'left':
            # Previous dimension combination
            self.combo_idx = (self.combo_idx - 1) % len(self.dim_combos)
            self.dims = self.dim_combos[self.combo_idx]
            self._draw()
        elif event.key == 'up':
            # Next layer
            self.current_layer = min(self.current_layer + 1, self.n_layers - 1)
            self.layer_slider.set_val(self.current_layer)
            self._draw()
        elif event.key == 'down':
            # Previous layer
            self.current_layer = max(self.current_layer - 1, 0)
            self.layer_slider.set_val(self.current_layer)
            self._draw()
        elif event.key == ' ':
            # Play/pause
            self.playing = not self.playing
            if self.playing:
                self._play_animation()
        elif event.key == 't':
            # Toggle 2D/3D
            self.mode_3d = not self.mode_3d
            self.fig.clear()
            self._setup_figure()
        elif event.key == 'c':
            self.show_centroids = not self.show_centroids
            self._draw()
        elif event.key == 'h':
            self.show_hull = not self.show_hull
            self._draw()
        elif event.key == 'r':
            # Reset
            self.current_layer = self.n_layers - 1
            self.combo_idx = 0
            self.dims = self.dim_combos[0]
            self.layer_slider.set_val(self.current_layer)
            self._draw()
        elif event.key == 's':
            # Save
            fname = f"frame_layer{self.current_layer:03d}_dims{'_'.join(map(str,self.dims))}.png"
            self.fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname}")
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key == 'a':
            # Auto-detect interesting dims for current view
            # Find dims where current group has most convergence
            group_names = list(self.data.trajectories.keys())
            if group_names:
                dims = self.data.find_attractor_dims(group_names[0], top_k=30)
                self.dim_combos = []
                for i in range(0, len(dims) - 2, 2):
                    self.dim_combos.append(dims[i:i+3])
                self.combo_idx = 0
                self.dims = self.dim_combos[0]
                self._draw()
        elif event.key in '123456789':
            # Highlight specific trajectory
            idx = int(event.key) - 1
            if idx < len(self.flat_trajs):
                self.highlighted_idx = idx
                self._draw()
        elif event.key == '0':
            self.highlighted_idx = None
            self._draw()
        elif event.key == 'pageup':
            # Jump 5 layers forward
            self.current_layer = min(self.current_layer + 5, self.n_layers - 1)
            self.layer_slider.set_val(self.current_layer)
            self._draw()
        elif event.key == 'pagedown':
            # Jump 5 layers back
            self.current_layer = max(self.current_layer - 5, 0)
            self.layer_slider.set_val(self.current_layer)
            self._draw()

    def _play_animation(self):
        """Animate through layers."""
        import time
        while self.playing and self.current_layer < self.n_layers - 1:
            self.current_layer += self.play_speed
            self.current_layer = min(self.current_layer, self.n_layers - 1)
            self.layer_slider.set_val(self.current_layer)
            self._draw()
            self.fig.canvas.flush_events()
            time.sleep(0.1)

            if self.current_layer >= self.n_layers - 1:
                self.playing = False

    def _draw(self):
        """Redraw the main plot."""
        self.ax.clear()
        self.info_ax.clear()
        self.info_ax.axis("off")

        layer = self.current_layer
        d0, d1 = self.dims[0], self.dims[1]
        d2 = self.dims[2] if len(self.dims) > 2 else None

        # Collect points at current layer
        for group_name, trajs in self.data.trajectories.items():
            color = self.group_colors[group_name]
            points = np.stack([t[layer] for t in trajs])

            x = points[:, d0]
            y = points[:, d1]

            if self.mode_3d and d2 is not None:
                z = points[:, d2]
                self.ax.scatter(x, y, z, c=[color], s=40, alpha=0.7,
                              edgecolors="k", linewidths=0.3, label=group_name)
            else:
                self.ax.scatter(x, y, c=[color], s=40, alpha=0.7,
                              edgecolors="k", linewidths=0.3, label=group_name)

            # Draw trails
            if self.show_trails and layer > 0:
                trail_start = max(0, layer - self.trail_length)
                for traj in trajs:
                    trail_x = traj[trail_start:layer+1, d0]
                    trail_y = traj[trail_start:layer+1, d1]
                    if self.mode_3d and d2 is not None:
                        trail_z = traj[trail_start:layer+1, d2]
                        self.ax.plot(trail_x, trail_y, trail_z,
                                   color=color, alpha=0.2, linewidth=0.7)
                    else:
                        self.ax.plot(trail_x, trail_y,
                                   color=color, alpha=0.2, linewidth=0.7)

            # Centroid
            if self.show_centroids:
                cx, cy = x.mean(), y.mean()
                if self.mode_3d and d2 is not None:
                    cz = z.mean()
                    self.ax.scatter([cx], [cy], [cz], c=[color], s=200, marker="*",
                                  edgecolors="k", linewidths=1, zorder=10)
                else:
                    self.ax.scatter([cx], [cy], c=[color], s=200, marker="*",
                                  edgecolors="k", linewidths=1, zorder=10)

        # Highlighted trajectory
        if self.highlighted_idx is not None and self.highlighted_idx < len(self.flat_trajs):
            gname, pidx, traj = self.flat_trajs[self.highlighted_idx]
            full_x = traj[:layer+1, d0]
            full_y = traj[:layer+1, d1]
            if self.mode_3d and d2 is not None:
                full_z = traj[:layer+1, d2]
                self.ax.plot(full_x, full_y, full_z, "r-", linewidth=2.5, alpha=0.9, zorder=20)
                self.ax.scatter([full_x[-1]], [full_y[-1]], [full_z[-1]],
                              c="red", s=120, marker="D", edgecolors="k", linewidths=1.5, zorder=21)
            else:
                self.ax.plot(full_x, full_y, "r-", linewidth=2.5, alpha=0.9, zorder=20)
                self.ax.scatter([full_x[-1]], [full_y[-1]],
                              c="red", s=120, marker="D", edgecolors="k", linewidths=1.5, zorder=21)

        # Labels
        if self.mode_3d and d2 is not None:
            self.ax.set_xlabel(f"dim_{d0:04d}")
            self.ax.set_ylabel(f"dim_{d1:04d}")
            self.ax.set_zlabel(f"dim_{d2:04d}")
        else:
            self.ax.set_xlabel(f"dim_{d0:04d}")
            self.ax.set_ylabel(f"dim_{d1:04d}")

        self.ax.set_title(f"Layer {layer}/{self.n_layers-1} | Dims: {d0}, {d1}" +
                         (f", {d2}" if d2 and self.mode_3d else ""))
        self.ax.legend(loc="upper left", fontsize=7)
        self.ax.grid(True, alpha=0.3)

        # Info panel
        info_lines = [
            "═══ CONTROLS ═══",
            "←/→  Change dims",
            "↑/↓  Step layers",
            "Space Play/Pause",
            "T    Toggle 2D/3D",
            "C    Centroids",
            "H    Convex hull",
            "A    Auto-dims",
            "S    Save frame",
            "1-9  Highlight traj",
            "0    Clear highlight",
            "PgUp/Dn  Jump ±5",
            "Q    Quit",
            "",
            "═══ STATE ═══",
            f"Layer: {layer}/{self.n_layers-1}",
            f"Dims: {self.dims}",
            f"Combo: {self.combo_idx+1}/{len(self.dim_combos)}",
            f"3D: {'ON' if self.mode_3d else 'OFF'}",
            f"Trails: {'ON' if self.show_trails else 'OFF'}",
            "",
            "═══ GROUPS ═══",
        ]

        for gname in self.data.trajectories:
            n = len(self.data.trajectories[gname])
            info_lines.append(f"  {gname}: {n}")

        if self.highlighted_idx is not None and self.highlighted_idx < len(self.flat_trajs):
            gname, pidx, _ = self.flat_trajs[self.highlighted_idx]
            info_lines.append("")
            info_lines.append("═══ HIGHLIGHTED ═══")
            info_lines.append(f"Group: {gname}")
            info_lines.append(f"Prompt #{pidx}")
            if gname in self.data.prompts and pidx < len(self.data.prompts[gname]):
                prompt = self.data.prompts[gname][pidx]
                # Wrap long prompts
                if len(prompt) > 25:
                    info_lines.append(f"'{prompt[:25]}...'")
                else:
                    info_lines.append(f"'{prompt}'")
            if gname in self.data.predictions and pidx < len(self.data.predictions[gname]):
                pred = self.data.predictions[gname][pidx]
                info_lines.append(f"→ '{pred}'")

        for i, line in enumerate(info_lines):
            self.info_ax.text(0, 1.0 - i * 0.028, line, fontsize=7,
                            fontfamily="monospace", va="top", transform=self.info_ax.transAxes)

        self.fig.canvas.draw_idle()

    def run(self):
        """Start the interactive explorer."""
        print("\n" + "=" * 50)
        print("INTERACTIVE TRAJECTORY EXPLORER")
        print("=" * 50)
        print("Use arrow keys to navigate, Q to quit")
        print("See right panel for all controls")
        print("=" * 50 + "\n")
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive Residual Stream Trajectory Explorer")
    parser.add_argument("data_dir", type=str,
                       help="Path to attractor_data/ directory (output of residual_attractors.py)")
    parser.add_argument("--groups", type=str, nargs="*", default=None,
                       help="Specific groups to load (default: all)")
    parser.add_argument("--dims", type=str, default=None,
                       help="Initial dimensions to show, comma-separated (e.g. '47,203,1891')")
    parser.add_argument("--all-groups", action="store_true",
                       help="Load all available groups")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    # Parse initial dims
    initial_dims = None
    if args.dims:
        initial_dims = [int(d.strip()) for d in args.dims.split(",")]

    # Load data
    print(f"Loading data from: {data_dir}")
    data = TrajectoryData(data_dir, groups=args.groups)

    # Launch explorer
    explorer = TrajectoryExplorer(data, initial_dims=initial_dims)
    explorer.run()


if __name__ == "__main__":
    main()
