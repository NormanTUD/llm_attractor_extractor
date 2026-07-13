# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
Interactive Residual Stream Trajectory Viewer (3D, WASD)

Shows how token representations move through embedding dimensions across layers.
Each point = one prompt's final-token residual stream at the current layer.
Layer = time dimension. You step through it with A/D.

Controls:
    A/D         — step backward/forward through layers (time)
    W/S         — cycle through dimension triplets
    Space       — play/pause animation through layers
    1-9         — highlight specific trajectory
    0           — clear highlight
    C           — toggle centroids
    F           — find best dims (auto-detect convergence dims)
    R           — reset to last layer + first dim combo
    P           — save current frame as PNG
    Q / Escape  — quit

Usage:
    python3 viewer.py attractor_data/berlin_multilingual
    python3 viewer.py attractor_data/
    python3 viewer.py attractor_data/ --groups berlin_multilingual paris_multilingual
    python3 viewer.py attractor_data/berlin_multilingual --dims 47,203,512
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
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =============================================================================
# Data Loader — handles both single-group and multi-group directories
# =============================================================================

class TrajectoryData:
    """Loads residual stream trajectory data from CSV output."""

    def __init__(self, data_dir: Path, groups: list[str] = None):
        self.data_dir = data_dir
        self.trajectories = {}   # group_name -> list of (n_layers, d_model) arrays
        self.prompts = {}        # group_name -> list of prompt strings
        self.predictions = {}    # group_name -> list of predicted next tokens
        self.n_layers = 0
        self.d_model = 0

        self._load(groups)

    def _load(self, groups: list[str] = None):
        """Smart loader: detects whether data_dir is a group dir or parent dir."""
        data_dir = self.data_dir

        # Case 1: data_dir itself contains final_token_streams/ (it's a single group)
        if (data_dir / "final_token_streams").is_dir():
            group_name = data_dir.name
            self._load_group(group_name, data_dir)
        else:
            # Case 2: data_dir contains subdirectories that are groups
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
            print(f"Expected structure:")
            print(f"  {data_dir}/final_token_streams/layer_000.csv  (single group)")
            print(f"  OR")
            print(f"  {data_dir}/<group_name>/final_token_streams/layer_000.csv  (multi group)")
            sys.exit(1)

        # Set dimensions
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
        """Load a single group from its directory."""
        final_dir = group_dir / "final_token_streams"

        # Try combined CSV first (fastest)
        all_csv = final_dir / "all_layers_all_prompts.csv"
        if all_csv.exists():
            self._load_combined_csv(group_name, all_csv)
        else:
            # Fall back to per-layer CSVs
            self._load_layer_csvs(group_name, final_dir)

        # Load metadata
        meta_csv = group_dir / "prompts_meta.csv"
        if meta_csv.exists():
            self._load_meta(group_name, meta_csv)

    def _load_combined_csv(self, group_name: str, csv_path: Path):
        """Load from all_layers_all_prompts.csv."""
        import csv

        print(f"  Loading {group_name} from combined CSV...")
        trajectories = {}  # prompt_idx -> {layer: vector}

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

        # Convert to arrays
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
        """Load from individual layer_XXX.csv files."""
        import csv

        layer_files = sorted(final_dir.glob("layer_*.csv"))
        if not layer_files:
            print(f"  WARNING: No layer CSVs in {final_dir}")
            return

        print(f"  Loading {group_name} from {len(layer_files)} layer CSVs...")

        # Read first to get structure
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
        """Flat list of (group_name, prompt_idx, trajectory)."""
        result = []
        for name, trajs in self.trajectories.items():
            for i, traj in enumerate(trajs):
                result.append((name, i, traj))
        return result

    def find_interesting_dims(self, top_k: int = 30) -> list[int]:
        """Dims with highest variance at final layer (most spread = most visible)."""
        all_final = []
        for trajs in self.trajectories.values():
            for traj in trajs:
                all_final.append(traj[-1])
        X = np.stack(all_final)
        variances = X.var(axis=0)
        return np.argsort(variances)[::-1][:top_k].tolist()

    def find_convergence_dims(self, top_k: int = 30) -> list[int]:
        """Dims where variance decreases most from first to last layer (convergence)."""
        all_first = []
        all_last = []
        for trajs in self.trajectories.values():
            for traj in trajs:
                all_first.append(traj[0])
                all_last.append(traj[-1])

        var_first = np.stack(all_first).var(axis=0)
        var_last = np.stack(all_last).var(axis=0)

        # Ratio: high = converged a lot
        ratio = var_first / (var_last + 1e-10)
        return np.argsort(ratio)[::-1][:top_k].tolist()


# =============================================================================
# 3D Interactive Viewer
# =============================================================================

class Viewer3D:
    """Always-3D interactive viewer with WASD controls."""

    def __init__(self, data: TrajectoryData, initial_dims: list[int] = None):
        self.data = data
        self.n_layers = data.n_layers
        self.d_model = data.d_model

        # State
        self.current_layer = data.n_layers - 1
        self.playing = False
        self.show_centroids = True
        self.show_trails = True
        self.trail_length = 8
        self.highlighted_idx = None

        # Dimension combos
        interesting = data.find_interesting_dims(top_k=30)
        self.dim_combos = []
        for i in range(0, len(interesting) - 2, 3):
            self.dim_combos.append(interesting[i:i+3])
        if not self.dim_combos:
            self.dim_combos = [[0, 1, 2]]

        if initial_dims and len(initial_dims) >= 3:
            self.dims = initial_dims[:3]
            # Insert at front
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
        """Create figure."""
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title(
            "Residual Stream 3D Viewer — WASD to navigate"
        )

        # 3D axes
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._draw()

    def _on_key(self, event):
        if event.key == 'd':
            # Forward in time (next layer)
            self.current_layer = min(self.current_layer + 1, self.n_layers - 1)
            self._draw()
        elif event.key == 'a':
            # Backward in time (previous layer)
            self.current_layer = max(self.current_layer - 1, 0)
            self._draw()
        elif event.key == 'w':
            # Next dim combo
            self.combo_idx = (self.combo_idx + 1) % len(self.dim_combos)
            self.dims = self.dim_combos[self.combo_idx]
            self._draw()
        elif event.key == 's':
            # Previous dim combo
            self.combo_idx = (self.combo_idx - 1) % len(self.dim_combos)
            self.dims = self.dim_combos[self.combo_idx]
            self._draw()
        elif event.key == ' ':
            self.playing = not self.playing
            if self.playing:
                self._play()
        elif event.key == 'c':
            self.show_centroids = not self.show_centroids
            self._draw()
        elif event.key == 'f':
            # Find convergence dims
            conv_dims = self.data.find_convergence_dims(top_k=30)
            self.dim_combos = []
            for i in range(0, len(conv_dims) - 2, 3):
                self.dim_combos.append(conv_dims[i:i+3])
            self.combo_idx = 0
            self.dims = self.dim_combos[0]
            print(f"  Switched to convergence dims: {self.dims}")
            self._draw()
        elif event.key == 'r':
            self.current_layer = self.n_layers - 1
            self.combo_idx = 0
            self.dims = self.dim_combos[0]
            self.highlighted_idx = None
            self._draw()
        elif event.key == 'p':
            fname = f"frame_L{self.current_layer:03d}_d{self.dims[0]}_{self.dims[1]}_{self.dims[2]}.png"
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
        elif event.key == 'right':
            # Jump 5 layers forward
            self.current_layer = min(self.current_layer + 5, self.n_layers - 1)
            self._draw()
        elif event.key == 'left':
            # Jump 5 layers back
            self.current_layer = max(self.current_layer - 5, 0)
            self._draw()

    def _play(self):
        """Animate through layers."""
        import time
        while self.playing and self.current_layer < self.n_layers - 1:
            self.current_layer += 1
            self._draw()
            self.fig.canvas.flush_events()
            time.sleep(0.08)
            if self.current_layer >= self.n_layers - 1:
                self.playing = False

    def _draw(self):
        """Redraw the 3D plot."""
        # Store current view angle
        elev = self.ax.elev
        azim = self.ax.azim

        self.ax.clear()

        layer = self.current_layer
        d0, d1, d2 = self.dims[0], self.dims[1], self.dims[2]

        # Plot each group
        for group_name, trajs in self.data.trajectories.items():
            color = self.group_colors[group_name]
            points = np.stack([t[layer] for t in trajs])

            x = points[:, d0]
            y = points[:, d1]
            z = points[:, d2]

            self.ax.scatter(x, y, z, c=[color], s=50, alpha=0.8,
                          edgecolors="k", linewidths=0.3, label=group_name)

            # Token labels
            predictions = self.data.predictions.get(group_name, [])
            for i in range(len(x)):
                if i < len(predictions) and predictions[i]:
                    self.ax.text(x[i], y[i], z[i], f" {predictions[i]}",
                               fontsize=6, alpha=0.7, color=color)

            # Trails
            if self.show_trails and layer > 0:
                trail_start = max(0, layer - self.trail_length)
                for traj in trajs:
                    tx = traj[trail_start:layer+1, d0]
                    ty = traj[trail_start:layer+1, d1]
                    tz = traj[trail_start:layer+1, d2]
                    self.ax.plot(tx, ty, tz, color=color, alpha=0.15, linewidth=0.8)

            # Centroid
            if self.show_centroids:
                cx, cy, cz = x.mean(), y.mean(), z.mean()
                self.ax.scatter([cx], [cy], [cz], c=[color], s=250, marker="*",
                              edgecolors="k", linewidths=1.2, zorder=10)

        # Highlighted trajectory (full path up to current layer)
        if self.highlighted_idx is not None and self.highlighted_idx < len(self.flat_trajs):
            gname, pidx, traj = self.flat_trajs[self.highlighted_idx]
            path_x = traj[:layer+1, d0]
            path_y = traj[:layer+1, d1]
            path_z = traj[:layer+1, d2]

            self.ax.plot(path_x, path_y, path_z, "r-", linewidth=2.5, alpha=0.9, zorder=20)
            self.ax.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]],
                          c="red", s=150, marker="D", edgecolors="k",
                          linewidths=1.5, zorder=21)

            # Start point
            if len(path_x) > 1:
                self.ax.scatter([path_x[0]], [path_y[0]], [path_z[0]],
                              c="green", s=80, marker="^", edgecolors="k",
                              linewidths=1, zorder=21)

        # Labels and title
        self.ax.set_xlabel(f"dim {d0}", fontsize=9)
        self.ax.set_ylabel(f"dim {d1}", fontsize=9)
        self.ax.set_zlabel(f"dim {d2}", fontsize=9)

        # Build title with info
        n_groups = len(self.data.trajectories)
        n_total = len(self.flat_trajs)
        title = (f"Layer {layer}/{self.n_layers-1}  |  "
                f"dims [{d0}, {d1}, {d2}]  |  "
                f"combo {self.combo_idx+1}/{len(self.dim_combos)}\n"
                f"{n_groups} group(s), {n_total} trajectories  |  "
                f"A/D=layer  W/S=dims  Space=play  F=find  1-9=highlight")

        self.ax.set_title(title, fontsize=9, fontfamily="monospace")
        self.ax.legend(loc="upper left", fontsize=7)

        # Restore view angle
        self.ax.view_init(elev=elev, azim=azim)

        self.fig.canvas.draw_idle()

    def run(self):
        """Launch the viewer."""
        print("\n" + "=" * 60)
        print("  3D RESIDUAL STREAM TRAJECTORY VIEWER")
        print("=" * 60)
        print(f"  Layers: {self.n_layers}  |  Dimensions: {self.d_model}")
        print(f"  Current dims: {self.dims}")
        print()
        print("  CONTROLS:")
        print("    A / D        Layer back / forward (time)")
        print("    W / S        Cycle dimension triplets")
        print("    ← / →        Jump ±5 layers")
        print("    Space        Play/pause animation")
        print("    F            Find convergence dims")
        print("    C            Toggle centroids")
        print("    1-9          Highlight trajectory")
        print("    0            Clear highlight")
        print("    P            Save PNG")
        print("    R            Reset")
        print("    Q / Esc      Quit")
        print("=" * 60 + "\n")
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
  python3 viewer.py attractor_data/berlin_multilingual
  python3 viewer.py attractor_data/
  python3 viewer.py attractor_data/ --groups berlin_multilingual paris_multilingual
  python3 viewer.py attractor_data/berlin_multilingual --dims 47,203,512
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

    # Parse dims
    initial_dims = None
    if args.dims:
        initial_dims = [int(d.strip()) for d in args.dims.split(",")]
        if len(initial_dims) < 3:
            print(f"ERROR: Need at least 3 dimensions, got {len(initial_dims)}")
            sys.exit(1)

    # Load
    data = TrajectoryData(data_dir, groups=args.groups)

    # Launch
    viewer = Viewer3D(data, initial_dims=initial_dims)
    viewer.run()


if __name__ == "__main__":
    main()
