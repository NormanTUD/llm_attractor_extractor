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
Each axis dimension is independently controllable.

Controls:
    Left/Right    — change X-axis dimension (sorted by index)
    Up/Down       — change Y-axis dimension (sorted by index)
    Page Up/Down  — change Z-axis dimension (sorted by index)
    Shift+←/→    — jump X dim by 10
    Shift+↑/↓    — jump Y dim by 10

    F             — switch to convergence-ranked dims
    V             — switch to variance-ranked dims
    N             — switch to sequential dims (0,1,2,3,...)
    
    1-9           — highlight specific trajectory
    0             — clear highlight
    C             — toggle centroids
    T             — toggle token labels
    A             — toggle arrows
    R             — reset
    P             — save current frame as PNG
    Q / Escape / Ctrl+C — quit

    Mouse drag    — trackball rotation (natural)
    Scroll        — zoom

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
# Force Ctrl+C to kill immediately
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

    def get_variance_ranked_dims(self, top_k: int = None) -> list[int]:
        """All dims sorted by final-layer variance (descending)."""
        all_final = []
        for trajs in self.trajectories.values():
            for traj in trajs:
                all_final.append(traj[-1])
        X = np.stack(all_final)
        variances = X.var(axis=0)
        ranked = np.argsort(variances)[::-1]
        if top_k:
            return ranked[:top_k].tolist()
        return ranked.tolist()

    def get_convergence_ranked_dims(self, top_k: int = None) -> list[int]:
        """All dims sorted by convergence ratio (descending)."""
        all_first = []
        all_last = []
        for trajs in self.trajectories.values():
            for traj in trajs:
                all_first.append(traj[0])
                all_last.append(traj[-1])
        var_first = np.stack(all_first).var(axis=0)
        var_last = np.stack(all_last).var(axis=0)
        ratio = var_first / (var_last + 1e-10)
        ranked = np.argsort(ratio)[::-1]
        if top_k:
            return ranked[:top_k].tolist()
        return ranked.tolist()


# =============================================================================
# Trackball rotation
# =============================================================================

class Trackball:
    """Natural trackball-style 3D rotation replacing matplotlib's default."""

    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self._dragging = False
        self._last_x = 0
        self._last_y = 0

        # Disable matplotlib's built-in rotation
        self.ax.disable_mouse_rotation()

        self._cid_press = fig.canvas.mpl_connect('button_press_event', self._on_press)
        self._cid_release = fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._cid_motion = fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self._cid_scroll = fig.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self._dragging = True
            self._last_x = event.x
            self._last_y = event.y

    def _on_release(self, event):
        if event.button == 1:
            self._dragging = False

    def _on_motion(self, event):
        if not self._dragging or event.x is None or event.y is None:
            return

        dx = event.x - self._last_x
        dy = event.y - self._last_y
        self._last_x = event.x
        self._last_y = event.y

        sensitivity = 0.4
        azim = self.ax.azim - dx * sensitivity
        elev = self.ax.elev + dy * sensitivity
        elev = max(-90, min(90, elev))

        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        factor = 0.85 if event.button == 'up' else 1.15

        for getter, setter in [(self.ax.get_xlim, self.ax.set_xlim),
                               (self.ax.get_ylim, self.ax.set_ylim),
                               (self.ax.get_zlim, self.ax.set_zlim)]:
            lo, hi = getter()
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * factor
            setter(mid - half, mid + half)

        self.fig.canvas.draw_idle()


# =============================================================================
# 3D Viewer
# =============================================================================

class Viewer3D:
    """
    3D viewer with independently controllable axis dimensions.
    
    Each axis (X, Y, Z) has its own dimension index that you step through
    sequentially. The dim list can be sorted by:
      - index (sequential: 0, 1, 2, 3, ...)
      - variance (most informative first)
      - convergence (most attractor-like first)
    """

    def __init__(self, data: TrajectoryData, initial_dims: list[int] = None):
        self.data = data
        self.n_layers = data.n_layers
        self.d_model = data.d_model

        # Dimension ordering modes
        self.dim_orders = {
            'sequential': list(range(data.d_model)),
            'variance': data.get_variance_ranked_dims(),
            'convergence': data.get_convergence_ranked_dims(),
        }
        self.current_order_name = 'variance'
        self.dim_order = self.dim_orders[self.current_order_name]

        # Current position in the dim_order list for each axis
        if initial_dims and len(initial_dims) >= 3:
            # Find positions in current order
            self.axis_pos = [0, 1, 2]
            for i, d in enumerate(initial_dims[:3]):
                if d in self.dim_order:
                    self.axis_pos[i] = self.dim_order.index(d)
                else:
                    self.axis_pos[i] = i
        else:
            self.axis_pos = [0, 1, 2]  # positions into dim_order

        # State
        self.show_centroids = True
        self.show_arrows = True
        self.show_labels = True
        self.highlighted_idx = None
        self.arrow_step = max(1, data.n_layers // 8)

        # Colors
        group_names = list(data.trajectories.keys())
        cmap = plt.cm.tab10
        self.group_colors = {}
        for i, name in enumerate(group_names):
            self.group_colors[name] = cmap(i % 10)

        self.flat_trajs = data.get_all_trajectories()

        self._build()

    @property
    def dims(self) -> list[int]:
        """Current actual dimension indices for X, Y, Z."""
        return [self.dim_order[p % len(self.dim_order)] for p in self.axis_pos]

    def _build(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title(
            "Residual Stream 3D — ←→ X dim, ↑↓ Y dim, PgUp/Dn Z dim"
        )

        self.ax = self.fig.add_subplot(111, projection='3d')
        self.trackball = Trackball(self.ax, self.fig)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('close_event', lambda _: os._exit(0))

        self._draw()

    def _step_axis(self, axis_idx: int, delta: int):
        """Step one axis dimension by delta positions in the current ordering."""
        self.axis_pos[axis_idx] = (self.axis_pos[axis_idx] + delta) % len(self.dim_order)
        # Avoid duplicates: if we landed on a dim already used by another axis, skip
        used = set()
        for i in range(3):
            if i != axis_idx:
                used.add(self.axis_pos[i] % len(self.dim_order))
        while self.axis_pos[axis_idx] % len(self.dim_order) in used:
            self.axis_pos[axis_idx] = (self.axis_pos[axis_idx] + (1 if delta > 0 else -1)) % len(self.dim_order)

    def _on_key(self, event):
        redraw = True

        if event.key == 'right':
            self._step_axis(0, 1)  # X dim +1
        elif event.key == 'left':
            self._step_axis(0, -1)  # X dim -1
        elif event.key == 'up':
            self._step_axis(1, 1)  # Y dim +1
        elif event.key == 'down':
            self._step_axis(1, -1)  # Y dim -1
        elif event.key == 'pageup':
            self._step_axis(2, 1)  # Z dim +1
        elif event.key == 'pagedown':
            self._step_axis(2, -1)  # Z dim -1
        # Shift variants: jump by 10
        elif event.key == 'shift+right':
            self._step_axis(0, 10)
        elif event.key == 'shift+left':
            self._step_axis(0, -10)
        elif event.key == 'shift+up':
            self._step_axis(1, 10)
        elif event.key == 'shift+down':
            self._step_axis(1, -10)
        elif event.key == 'shift+pageup' or event.key == 'ctrl+pageup':
            self._step_axis(2, 10)
        elif event.key == 'shift+pagedown' or event.key == 'ctrl+pagedown':
            self._step_axis(2, -10)
        # Ordering modes
        elif event.key == 'n':
            self._switch_order('sequential')
        elif event.key == 'v':
            self._switch_order('variance')
        elif event.key == 'f':
            self._switch_order('convergence')
        # Toggles
        elif event.key == 'c':
            self.show_centroids = not self.show_centroids
        elif event.key == 'a':
            self.show_arrows = not self.show_arrows
        elif event.key == 't':
            self.show_labels = not self.show_labels
        elif event.key == 'r':
            self.axis_pos = [0, 1, 2]
            self.highlighted_idx = None
        elif event.key == 'p':
            d = self.dims
            fname = f"frame_d{d[0]}_{d[1]}_{d[2]}.png"
            self.fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved: {fname}")
            redraw = False
        elif event.key in '123456789':
            idx = int(event.key) - 1
            if idx < len(self.flat_trajs):
                self.highlighted_idx = idx
                gname, pidx, _ = self.flat_trajs[idx]
                prompt = ""
                if gname in self.data.prompts and pidx < len(self.data.prompts[gname]):
                    prompt = self.data.prompts[gname][pidx]
                pred = ""
                if gname in self.data.predictions and pidx < len(self.data.predictions[gname]):
                    pred = self.data.predictions[gname][pidx]
                print(f"  Highlighted #{idx+1}: [{gname}] '{prompt}' → '{pred}'")
        elif event.key == '0':
            self.highlighted_idx = None
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)
            os._exit(0)
        else:
            redraw = False

        if redraw:
            self._draw()

    def _switch_order(self, order_name: str):
        """Switch dimension ordering, preserving current actual dims."""
        old_dims = self.dims
        self.current_order_name = order_name
        self.dim_order = self.dim_orders[order_name]
        # Re-find positions for current dims in new order
        for i, d in enumerate(old_dims):
            if d in self.dim_order:
                self.axis_pos[i] = self.dim_order.index(d)
            else:
                self.axis_pos[i] = i
        print(f"  Switched to '{order_name}' ordering. Dims: {self.dims}")

    def _draw(self):
        elev = self.ax.elev
        azim = self.ax.azim

        self.ax.clear()

        d0, d1, d2 = self.dims

        # Plot each group
        for group_name, trajs in self.data.trajectories.items():
            color = self.group_colors[group_name]

            for traj_idx, traj in enumerate(trajs):
                x = traj[:, d0]
                y = traj[:, d1]
                z = traj[:, d2]

                # Full trajectory line
                self.ax.plot(x, y, z, color=color, alpha=0.35, linewidth=0.9)

                # Start marker
                self.ax.scatter([x[0]], [y[0]], [z[0]], c=[color], s=15,
                              marker='o', alpha=0.3, edgecolors='none')

                # End marker
                self.ax.scatter([x[-1]], [y[-1]], [z[-1]], c=[color], s=55,
                              marker='o', alpha=0.9, edgecolors='k', linewidths=0.4)

                # Direction arrows
                if self.show_arrows:
                    for i in range(0, len(x) - 1, self.arrow_step):
                        dx_a = x[i+1] - x[i]
                        dy_a = y[i+1] - y[i]
                        dz_a = z[i+1] - z[i]
                        length = np.sqrt(dx_a**2 + dy_a**2 + dz_a**2)
                        if length > 1e-8:
                            alpha = 0.15 + 0.5 * (i / max(len(x) - 1, 1))
                            self.ax.quiver(
                                x[i], y[i], z[i],
                                dx_a, dy_a, dz_a,
                                color=color, alpha=alpha,
                                arrow_length_ratio=0.3,
                                linewidth=0.7
                            )

            # Token labels at final position
            if self.show_labels:
                predictions = self.data.predictions.get(group_name, [])
                for traj_idx, traj in enumerate(trajs):
                    if traj_idx < len(predictions) and predictions[traj_idx]:
                        self.ax.text(
                            traj[-1, d0], traj[-1, d1], traj[-1, d2],
                            f" {predictions[traj_idx]}",
                            fontsize=7, alpha=0.8, color=color, fontweight='bold'
                        )

            # Centroid at final layer
            if self.show_centroids:
                final_pts = np.stack([t[-1] for t in trajs])
                cx = final_pts[:, d0].mean()
                cy = final_pts[:, d1].mean()
                cz = final_pts[:, d2].mean()
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

            # Denser arrows on highlighted
            step = max(1, self.arrow_step // 2)
            for i in range(0, len(x) - 1, step):
                dx_a = x[i+1] - x[i]
                dy_a = y[i+1] - y[i]
                dz_a = z[i+1] - z[i]
                length = np.sqrt(dx_a**2 + dy_a**2 + dz_a**2)
                if length > 1e-8:
                    self.ax.quiver(
                        x[i], y[i], z[i],
                        dx_a, dy_a, dz_a,
                        color='red', alpha=0.7,
                        arrow_length_ratio=0.35,
                        linewidth=1.4, zorder=21
                    )

            self.ax.scatter([x[0]], [y[0]], [z[0]], c="limegreen", s=100,
                          marker="^", edgecolors="k", linewidths=1.5, zorder=22)
            self.ax.scatter([x[-1]], [y[-1]], [z[-1]], c="red", s=150,
                          marker="D", edgecolors="k", linewidths=1.5, zorder=22)

        # Axis labels
        self.ax.set_xlabel(f"dim {d0}", fontsize=10, fontweight='bold')
        self.ax.set_ylabel(f"dim {d1}", fontsize=10, fontweight='bold')
        self.ax.set_zlabel(f"dim {d2}", fontsize=10, fontweight='bold')

        # Title
        n_groups = len(self.data.trajectories)
        n_total = len(self.flat_trajs)
        order_char = self.current_order_name[0].upper()
        title = (
            f"X=dim {d0}  Y=dim {d1}  Z=dim {d2}  |  "
            f"order: {self.current_order_name}  |  "
            f"{n_groups} group(s), {n_total} traj\n"
            f"←→=X  ↑↓=Y  PgUp/Dn=Z  (shift=×10)  |  "
            f"N=seq  V=var  F=conv  |  A=arrows  T=labels  C=centroids"
        )
        self.ax.set_title(title, fontsize=8.5, fontfamily="monospace")

        if self.show_centroids:
            self.ax.legend(loc="upper left", fontsize=8)

        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def run(self):
        d = self.dims
        print("\n" + "=" * 70)
        print("  3D RESIDUAL STREAM TRAJECTORY VIEWER")
        print("=" * 70)
        print(f"  Layers: {self.n_layers}  |  Total dims: {self.d_model}")
        print(f"  Starting dims: X={d[0]}, Y={d[1]}, Z={d[2]}")
        print(f"  Ordering: {self.current_order_name}")
        print()
        print("  DIMENSION NAVIGATION (each axis independent):")
        print("    ← / →          X-axis dim ±1")
        print("    ↑ / ↓          Y-axis dim ±1")
        print("    PgUp / PgDn    Z-axis dim ±1")
        print("    Shift+key      Jump ±10")
        print()
        print("  ORDERING MODES:")
        print("    N              Sequential (0, 1, 2, 3, ...)")
        print("    V              By variance (most spread first)")
        print("    F              By convergence (most attractor-like first)")
        print()
        print("  DISPLAY:")
        print("    A              Toggle arrows")
        print("    T              Toggle token labels")
        print("    C              Toggle centroids")
        print("    1-9            Highlight trajectory")
        print("    0              Clear highlight")
        print("    P              Save PNG")
        print("    R              Reset")
        print("    Q / Esc        Quit")
        print("    Ctrl+C         Force quit (works unfocused)")
        print()
        print("  MOUSE:")
        print("    Left drag      Natural trackball rotation")
        print("    Scroll         Zoom")
        print("=" * 70 + "\n")
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D Residual Stream Trajectory Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("data_dir", type=str,
                       help="Path to group directory or parent directory")
    parser.add_argument("--groups", type=str, nargs="*", default=None,
                       help="Specific groups to load")
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
