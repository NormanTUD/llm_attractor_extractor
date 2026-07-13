# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
Token-Level Residual Stream Trajectory Viewer (3D) — COMPLETE REWRITE

Every single token of every prompt/language = its own trajectory through layers.
Groups are color-coded. All trajectories visible in parallel.

LIVE ANIMATION: Press SPACE to watch points paint their trajectories
layer by layer (0.1s per layer). See attractors form in real-time.

On mouseover:
  - Token text
  - Full sentence
  - Position in sentence (POS index)
  - Group/language

Controls:
    Left/Right    — change X-axis dimension
    Up/Down       — change Y-axis dimension
    Page Up/Down  — change Z-axis dimension
    Shift+key     — jump by 10

    N             — sequential dim ordering
    V             — variance-ranked dims
    K             — convergence-ranked dims (was F, now K)

    G             — toggle group coloring vs token-position coloring
    A             — toggle arrows
    T             — toggle token text labels
    C             — toggle group centroids
    R             — reset view
    P             — save PNG
    Q / Escape    — quit

    SPACE         — start/stop LIVE animation (points paint trajectories)
    B             — reset animation to beginning
    +/-           — speed up / slow down animation

    Mouse drag    — trackball rotation
    Scroll        — zoom
    Hover         — show token info tooltip

Usage:
    python3 viewer.py attractor_data/berlin_multilingual
    python3 viewer.py attractor_data/
    python3 viewer.py attractor_data/ --groups berlin_multilingual paris_multilingual
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

signal.signal(signal.SIGINT, lambda *_: os._exit(0))

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.font_manager as fm

# CJK Font Support
_cjk_fonts = [
    'Noto Sans CJK JP', 'Noto Sans JP', 'IPAGothic', 'IPAPGothic',
    'WenQuanYi Micro Hei', 'Arial Unicode MS', 'MS Gothic',
    'Hiragino Sans', 'Yu Gothic',
]
_available = {f.name for f in fm.fontManager.ttflist}
_fallback = [f for f in _cjk_fonts if f in _available]
if _fallback:
    plt.rcParams['font.family'] = ['DejaVu Sans'] + _fallback + ['sans-serif']
    print(f"  Font: using {_fallback[0]} for CJK support")
else:
    print("  WARNING: No CJK font found. Install fonts-noto-cjk for Japanese/Chinese.")


# =============================================================================
# Data Loader
# =============================================================================

class TokenTrajectoryData:
    """
    Loads residual stream data at the TOKEN level.
    
    Supports:
      - all_token_streams/all_layers_all_tokens.csv (full per-token data)
      - all_token_streams/layer_*.csv (per-layer token files)
      - final_token_streams/ (fallback: one trajectory per prompt)
    """

    def __init__(self, data_dir: Path, groups: list[str] = None):
        self.data_dir = data_dir
        self.tokens: list[dict] = []
        self.groups: list[str] = []
        self.n_layers = 0
        self.d_model = 0
        self._load(groups)

    def _load(self, groups: list[str] = None):
        data_dir = self.data_dir

        if (data_dir / "all_token_streams").is_dir():
            self._load_group_tokens(data_dir.name, data_dir)
        elif (data_dir / "final_token_streams").is_dir():
            self._load_group_final_only(data_dir.name, data_dir)
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
                if (group_dir / "all_token_streams").is_dir():
                    self._load_group_tokens(group_dir.name, group_dir)
                elif (group_dir / "final_token_streams").is_dir():
                    self._load_group_final_only(group_dir.name, group_dir)
                else:
                    print(f"  Skipping {group_dir.name}: no token stream data")

        if not self.tokens:
            print(f"\nERROR: No token trajectory data found in {data_dir}")
            sys.exit(1)

        self.groups = sorted(set(t['group'] for t in self.tokens))
        first = self.tokens[0]['trajectory']
        self.n_layers, self.d_model = first.shape

        print(f"\nLoaded: {len(self.tokens)} token trajectories across {len(self.groups)} group(s)")
        print(f"  Layers: {self.n_layers}, Dimensions: {self.d_model}")
        for g in self.groups:
            count = sum(1 for t in self.tokens if t['group'] == g)
            print(f"  {g}: {count} tokens")

    def _load_group_tokens(self, group_name: str, group_dir: Path):
        import csv
        token_dir = group_dir / "all_token_streams"
        all_csv = token_dir / "all_layers_all_tokens.csv"

        if not all_csv.exists():
            self._load_group_token_layers(group_name, token_dir, group_dir)
            return

        print(f"  Loading {group_name} token trajectories from combined CSV...")
        prompts, predictions = self._read_meta(group_dir)

        token_data = {}
        with open(all_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            dim_keys = None
            for row in reader:
                if dim_keys is None:
                    dim_keys = sorted([k for k in row.keys() if k.startswith("dim_")])
                layer = int(row["layer"])
                prompt_idx = int(row["prompt_idx"])
                token_pos = int(row.get("token_pos", 0))
                token_text = row.get("token_text", f"tok_{token_pos}")

                key = (prompt_idx, token_pos)
                if key not in token_data:
                    token_data[key] = {'layers': {}, 'text': token_text}
                token_data[key]['layers'][layer] = np.array(
                    [float(row[k]) for k in dim_keys], dtype=np.float32
                )

        for (prompt_idx, token_pos), info in sorted(token_data.items()):
            layers_dict = info['layers']
            n_layers = max(layers_dict.keys()) + 1
            d_model = len(dim_keys)
            arr = np.zeros((n_layers, d_model), dtype=np.float32)
            for l, vec in layers_dict.items():
                arr[l] = vec

            sentence = prompts[prompt_idx] if prompt_idx < len(prompts) else ""
            prediction = predictions[prompt_idx] if prompt_idx < len(predictions) else ""

            self.tokens.append({
                'group': group_name,
                'prompt_idx': prompt_idx,
                'token_pos': token_pos,
                'token_text': info['text'],
                'sentence': sentence,
                'prediction': prediction,
                'trajectory': arr,
            })

    def _load_group_token_layers(self, group_name: str, token_dir: Path, group_dir: Path):
        import csv
        layer_files = sorted(token_dir.glob("layer_*.csv"))
        if not layer_files:
            print(f"  WARNING: No layer CSVs in {token_dir}")
            return

        print(f"  Loading {group_name} tokens from {len(layer_files)} layer CSVs...")
        prompts, predictions = self._read_meta(group_dir)

        token_data = {}
        dim_keys = None

        for layer_idx, layer_file in enumerate(layer_files):
            with open(layer_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if dim_keys is None:
                        dim_keys = sorted([k for k in row.keys() if k.startswith("dim_")])
                    prompt_idx = int(row.get("prompt_idx", 0))
                    token_pos = int(row.get("token_pos", 0))
                    token_text = row.get("token_text", f"tok_{token_pos}")

                    key = (prompt_idx, token_pos)
                    if key not in token_data:
                        token_data[key] = {'layers': {}, 'text': token_text}
                    vec = np.array([float(row[k]) for k in dim_keys], dtype=np.float32)
                    token_data[key]['layers'][layer_idx] = vec

        if not dim_keys:
            return

        n_layers = len(layer_files)
        d_model = len(dim_keys)

        for (prompt_idx, token_pos), info in sorted(token_data.items()):
            arr = np.zeros((n_layers, d_model), dtype=np.float32)
            for l, vec in info['layers'].items():
                arr[l] = vec

            sentence = prompts[prompt_idx] if prompt_idx < len(prompts) else ""
            prediction = predictions[prompt_idx] if prompt_idx < len(predictions) else ""

            self.tokens.append({
                'group': group_name,
                'prompt_idx': prompt_idx,
                'token_pos': token_pos,
                'token_text': info['text'],
                'sentence': sentence,
                'prediction': prediction,
                'trajectory': arr,
            })

    def _load_group_final_only(self, group_name: str, group_dir: Path):
        import csv
        final_dir = group_dir / "final_token_streams"
        prompts, predictions = self._read_meta(group_dir)

        all_csv = final_dir / "all_layers_all_prompts.csv"
        if all_csv.exists():
            print(f"  Loading {group_name} (final-token only)...")
            trajectories = {}

            with open(all_csv, "r", encoding="utf-8") as f:
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

            for pidx in sorted(trajectories.keys()):
                layers_dict = trajectories[pidx]
                n_layers = max(layers_dict.keys()) + 1
                d_model = len(dim_keys)
                arr = np.zeros((n_layers, d_model), dtype=np.float32)
                for l, vec in layers_dict.items():
                    arr[l] = vec

                sentence = prompts[pidx] if pidx < len(prompts) else ""
                prediction = predictions[pidx] if pidx < len(predictions) else ""
                # Use prompt index as token_pos for coloring (since we don't have real pos)
                token_text = prediction if prediction else (sentence.split()[-1] if sentence else f"prompt_{pidx}")

                self.tokens.append({
                    'group': group_name,
                    'prompt_idx': pidx,
                    'token_pos': pidx,  # Use prompt index, NOT -1
                    'token_text': token_text,
                    'sentence': sentence,
                    'prediction': prediction,
                    'trajectory': arr,
                })
        else:
            layer_files = sorted(final_dir.glob("layer_*.csv"))
            if not layer_files:
                return

            print(f"  Loading {group_name} (final-token only) from {len(layer_files)} layer CSVs...")

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

            for pidx, arr in enumerate(all_trajs):
                sentence = prompts[pidx] if pidx < len(prompts) else ""
                prediction = predictions[pidx] if pidx < len(predictions) else ""
                token_text = prediction if prediction else (sentence.split()[-1] if sentence else f"prompt_{pidx}")

                self.tokens.append({
                    'group': group_name,
                    'prompt_idx': pidx,
                    'token_pos': pidx,  # Use prompt index, NOT -1
                    'token_text': token_text,
                    'sentence': sentence,
                    'prediction': prediction,
                    'trajectory': arr,
                })

    def _read_meta(self, group_dir: Path) -> tuple[list[str], list[str]]:
        import csv
        meta_csv = group_dir / "prompts_meta.csv"
        prompts = []
        predictions = []
        if meta_csv.exists():
            with open(meta_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompts.append(row.get("prompt", ""))
                    predictions.append(row.get("predicted_next_token", ""))
        return prompts, predictions

    def get_variance_ranked_dims(self) -> list[int]:
        all_final = np.stack([t['trajectory'][-1] for t in self.tokens])
        variances = all_final.var(axis=0)
        return np.argsort(variances)[::-1].tolist()

    def get_convergence_ranked_dims(self) -> list[int]:
        all_first = np.stack([t['trajectory'][0] for t in self.tokens])
        all_last = np.stack([t['trajectory'][-1] for t in self.tokens])
        var_first = all_first.var(axis=0)
        var_last = all_last.var(axis=0)
        ratio = var_first / (var_last + 1e-10)
        return np.argsort(ratio)[::-1].tolist()


# =============================================================================
# Trackball Rotation
# =============================================================================

class Trackball:
    def __init__(self, ax, fig):
        self.ax = ax
        self.fig = fig
        self._dragging = False
        self._last_x = 0
        self._last_y = 0
        self.ax.disable_mouse_rotation()
        fig.canvas.mpl_connect('button_press_event', self._on_press)
        fig.canvas.mpl_connect('button_release_event', self._on_release)
        fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        fig.canvas.mpl_connect('scroll_event', self._on_scroll)

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
# Main Viewer — Token-Level 3D with LIVE Animation
# =============================================================================

class TokenViewer3D:
    """
    Every token = its own trajectory line through layers.
    Groups color-coded. Live animation shows points painting their paths.
    Hover shows full context.
    """

    def __init__(self, data: TokenTrajectoryData, initial_dims: list[int] = None):
        self.data = data
        self.n_layers = data.n_layers
        self.d_model = data.d_model

        # Dimension orderings
        self.dim_orders = {
            'sequential': list(range(data.d_model)),
            'variance': data.get_variance_ranked_dims(),
            'convergence': data.get_convergence_ranked_dims(),
        }
        self.current_order_name = 'variance'
        self.dim_order = self.dim_orders[self.current_order_name]

        if initial_dims and len(initial_dims) >= 3:
            self.axis_pos = [0, 1, 2]
            for i, d in enumerate(initial_dims[:3]):
                if d in self.dim_order:
                    self.axis_pos[i] = self.dim_order.index(d)
                else:
                    self.axis_pos[i] = i
        else:
            self.axis_pos = [0, 1, 2]

        # Display state
        self.show_centroids = True
        self.show_arrows = False  # off by default for cleaner view
        self.show_labels = False
        self.color_by_group = True
        self.highlighted_idx = None

        # Animation state
        self.animating = False
        self.anim_layer = 0  # current layer shown in animation
        self.anim_speed = 0.1  # seconds per layer
        self.anim_timer = None
        self.show_all_layers = True  # False = only show up to anim_layer

        # Group colors (distinct, saturated)
        n_groups = len(data.groups)
        if n_groups <= 10:
            cmap = plt.cm.tab10
        else:
            cmap = plt.cm.tab20
        self.group_colors = {}
        for i, g in enumerate(data.groups):
            self.group_colors[g] = cmap(i % cmap.N)

        # Tooltip
        self._tooltip_annotation = None
        self._hover_idx = None
        self._endpoint_positions = []

        self._build()

    @property
    def dims(self) -> list[int]:
        return [self.dim_order[p % len(self.dim_order)] for p in self.axis_pos]

    def _build(self):
        self.fig = plt.figure(figsize=(16, 11))
        self.fig.canvas.manager.set_window_title(
            "Token Trajectory Viewer — SPACE for live animation"
        )
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.trackball = Trackball(self.ax, self.fig)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        self.fig.canvas.mpl_connect('close_event', lambda _: os._exit(0))

        self._draw()

    def _step_axis(self, axis_idx: int, delta: int):
        self.axis_pos[axis_idx] = (self.axis_pos[axis_idx] + delta) % len(self.dim_order)
        used = set()
        for i in range(3):
            if i != axis_idx:
                used.add(self.axis_pos[i] % len(self.dim_order))
        while self.axis_pos[axis_idx] % len(self.dim_order) in used:
            self.axis_pos[axis_idx] = (self.axis_pos[axis_idx] + (1 if delta > 0 else -1)) % len(self.dim_order)

    def _on_key(self, event):
        redraw = True

        if event.key == 'right':
            self._step_axis(0, 1)
        elif event.key == 'left':
            self._step_axis(0, -1)
        elif event.key == 'up':
            self._step_axis(1, 1)
        elif event.key == 'down':
            self._step_axis(1, -1)
        elif event.key == 'pageup':
            self._step_axis(2, 1)
        elif event.key == 'pagedown':
            self._step_axis(2, -1)
        elif event.key == 'shift+right':
            self._step_axis(0, 10)
        elif event.key == 'shift+left':
            self._step_axis(0, -10)
        elif event.key == 'shift+up':
            self._step_axis(1, 10)
        elif event.key == 'shift+down':
            self._step_axis(1, -10)

        # Ordering
        elif event.key == 'n':
            self._switch_order('sequential')
        elif event.key == 'v':
            self._switch_order('variance')
        elif event.key == 'k':
            self._switch_order('convergence')

        # Display toggles
        elif event.key == 'c':
            self.show_centroids = not self.show_centroids
        elif event.key == 'a':
            self.show_arrows = not self.show_arrows
        elif event.key == 't':
            self.show_labels = not self.show_labels
        elif event.key == 'g':
            self.color_by_group = not self.color_by_group
            mode = "group" if self.color_by_group else "token position"
            print(f"  Color mode: {mode}")

        # Animation
        elif event.key == ' ':
            self._toggle_animation()
            redraw = False
        elif event.key == 'b':
            self.anim_layer = 0
            self.show_all_layers = False
            print(f"  Animation reset to layer 0")
        elif event.key == '+' or event.key == '=':
            self.anim_speed = max(0.02, self.anim_speed - 0.02)
            print(f"  Animation speed: {self.anim_speed:.2f}s per layer")
            redraw = False
        elif event.key == '-':
            self.anim_speed = min(1.0, self.anim_speed + 0.02)
            print(f"  Animation speed: {self.anim_speed:.2f}s per layer")
            redraw = False

        # Misc
        elif event.key == 'r':
            self.axis_pos = [0, 1, 2]
            self.highlighted_idx = None
            self.show_all_layers = True
            self.anim_layer = self.n_layers - 1
        elif event.key == 'p':
            d = self.dims
            fname = f"token_frame_d{d[0]}_{d[1]}_{d[2]}.png"
            self.fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved: {fname}")
            redraw = False
        elif event.key in '123456789':
            idx = int(event.key) - 1
            if idx < len(self.data.tokens):
                self.highlighted_idx = idx
                tok = self.data.tokens[idx]
                print(f"  Highlighted #{idx+1}: '{tok['token_text']}' "
                      f"pos={tok['token_pos']} group={tok['group']}")
                print(f"    Sentence: {tok['sentence'][:100]}")
        elif event.key == '0':
            self.highlighted_idx = None
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)
            os._exit(0)
        else:
            redraw = False

        if redraw:
            self._draw()

    def _toggle_animation(self):
        """Start or stop the live trajectory painting animation."""
        if self.animating:
            self.animating = False
            if self.anim_timer:
                self.anim_timer.stop()
                self.anim_timer = None
            print("  Animation STOPPED")
        else:
            self.animating = True
            self.show_all_layers = False
            if self.anim_layer >= self.n_layers - 1:
                self.anim_layer = 0
            print(f"  Animation STARTED from layer {self.anim_layer} "
                  f"({self.anim_speed:.2f}s/layer)")
            self._animate_step()

    def _animate_step(self):
        """One step of the animation."""
        if not self.animating:
            return

        self._draw()
        self.fig.canvas.flush_events()

        self.anim_layer += 1
        if self.anim_layer >= self.n_layers:
            self.anim_layer = self.n_layers - 1
            self.animating = False
            print("  Animation COMPLETE — all layers shown")
            return

        # Schedule next step
        interval_ms = int(self.anim_speed * 1000)
        self.anim_timer = self.fig.canvas.new_timer(interval=interval_ms)
        self.anim_timer.add_callback(self._animate_step)
        self.anim_timer.single_shot = True
        self.anim_timer.start()

    def _on_hover(self, event):
        """Show tooltip on hover near a token endpoint."""
        if event.inaxes != self.ax or not self._endpoint_positions:
            if self._tooltip_annotation:
                self._tooltip_annotation.set_visible(False)
                self.fig.canvas.draw_idle()
            return

        if event.x is None or event.y is None:
            return

        min_dist = float('inf')
        nearest_idx = None

        for (x3d, y3d, z3d, tok_idx) in self._endpoint_positions:
            try:
                x2d, y2d, _ = proj3d.proj_transform(x3d, y3d, z3d, self.ax.get_proj())
                coords = self.ax.transData.transform((x2d, y2d))
                sx, sy = coords[0], coords[1]
                dist = ((event.x - sx)**2 + (event.y - sy)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = tok_idx
            except Exception:
                continue

        # Threshold: 20 pixels
        if min_dist < 20 and nearest_idx is not None:
            if nearest_idx != self._hover_idx:
                self._hover_idx = nearest_idx
                tok = self.data.tokens[nearest_idx]

                tooltip_lines = [
                    f"Token: \"{tok['token_text']}\"",
                    f"Position: {tok['token_pos']} in sentence",
                    f"Group: {tok['group']}",
                    f"Prompt #{tok['prompt_idx']}",
                    f"Sentence: {tok['sentence'][:80]}{'...' if len(tok['sentence']) > 80 else ''}",
                ]
                if tok['prediction']:
                    tooltip_lines.append(f"Next prediction: {tok['prediction']}")

                tooltip = "\n".join(tooltip_lines)

                if self._tooltip_annotation:
                    self._tooltip_annotation.remove()

                self._tooltip_annotation = self.fig.text(
                    0.02, 0.02, tooltip,
                    transform=self.fig.transFigure,
                    fontsize=9, fontfamily='monospace',
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                             edgecolor='gray', alpha=0.95),
                    zorder=100
                )
                self.fig.canvas.draw_idle()
        else:
            if self._tooltip_annotation and self._hover_idx is not None:
                self._tooltip_annotation.set_visible(False)
                self._hover_idx = None
                self.fig.canvas.draw_idle()

    def _switch_order(self, order_name: str):
        old_dims = self.dims
        self.current_order_name = order_name
        self.dim_order = self.dim_orders[order_name]
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
        self._endpoint_positions = []

        d0, d1, d2 = self.dims

        n_tokens = len(self.data.tokens)

        # Determine how many layers to show (for animation)
        if self.show_all_layers:
            max_layer = self.n_layers
        else:
            max_layer = self.anim_layer + 1  # show up to current animation layer

        # Position-based colormap
        pos_cmap = plt.cm.viridis

        # Precompute max_pos for position coloring (avoid the crash)
        max_pos = 0
        if not self.color_by_group:
            positions = [t['token_pos'] for t in self.data.tokens if t['token_pos'] >= 0]
            max_pos = max(positions) if positions else 1

        # Draw every token trajectory
        for tok_idx, tok in enumerate(self.data.tokens):
            traj = tok['trajectory'][:max_layer]  # only up to current anim layer
            group = tok['group']

            if self.color_by_group:
                color = self.group_colors[group]
            else:
                pos_val = tok['token_pos'] if tok['token_pos'] >= 0 else tok['prompt_idx']
                pos_norm = pos_val / max(max_pos, 1)
                color = pos_cmap(min(pos_norm, 1.0))

            x = traj[:, d0]
            y = traj[:, d1]
            z = traj[:, d2]

            # Alpha/linewidth based on highlight
            base_alpha = 0.5
            line_width = 1.0
            if self.highlighted_idx is not None:
                if tok_idx == self.highlighted_idx:
                    base_alpha = 1.0
                    line_width = 3.0
                else:
                    base_alpha = 0.08
                    line_width = 0.4

            # Trajectory line
            self.ax.plot(x, y, z, color=color, alpha=base_alpha, linewidth=line_width)

            # Current endpoint marker (last visible layer)
            marker_size = 30 if self.highlighted_idx is None else (80 if tok_idx == self.highlighted_idx else 15)
            self.ax.scatter([x[-1]], [y[-1]], [z[-1]], c=[color], s=marker_size,
                          marker='o', alpha=base_alpha, edgecolors='k', linewidths=0.3)

            # Store endpoint for hover
            self._endpoint_positions.append((x[-1], y[-1], z[-1], tok_idx))

            # Direction arrows
            if self.show_arrows and base_alpha > 0.1 and len(x) > 1:
                arrow_step = max(1, len(x) // 5)
                for i in range(0, len(x) - 1, arrow_step):
                    dx_a = x[i+1] - x[i]
                    dy_a = y[i+1] - y[i]
                    dz_a = z[i+1] - z[i]
                    length = np.sqrt(dx_a**2 + dy_a**2 + dz_a**2)
                    if length > 1e-8:
                        a = base_alpha * (0.3 + 0.5 * (i / max(len(x) - 1, 1)))
                        self.ax.quiver(
                            x[i], y[i], z[i],
                            dx_a, dy_a, dz_a,
                            color=color, alpha=a,
                            arrow_length_ratio=0.3,
                            linewidth=0.5
                        )

            # Token text labels
            if self.show_labels and base_alpha > 0.1:
                self.ax.text(
                    x[-1], y[-1], z[-1],
                    f" {tok['token_text']}",
                    fontsize=5.5, alpha=base_alpha * 0.9, color=color
                )

        # Highlighted trajectory emphasis
        if self.highlighted_idx is not None and self.highlighted_idx < n_tokens:
            tok = self.data.tokens[self.highlighted_idx]
            traj = tok['trajectory'][:max_layer]
            x = traj[:, d0]
            y = traj[:, d1]
            z = traj[:, d2]

            self.ax.plot(x, y, z, color='red', linewidth=3.5, alpha=0.95, zorder=20)
            self.ax.scatter([x[0]], [y[0]], [z[0]], c="limegreen", s=80,
                          marker="^", edgecolors="k", linewidths=1.2, zorder=22)
            self.ax.scatter([x[-1]], [y[-1]], [z[-1]], c="red", s=120,
                          marker="D", edgecolors="k", linewidths=1.2, zorder=22)
            self.ax.text(
                x[-1], y[-1], z[-1],
                f"  \"{tok['token_text']}\" (pos {tok['token_pos']})\n"
                f"  {tok['group']}",
                fontsize=7, color='red', zorder=23
            )

        # Group centroids
        if self.show_centroids:
            for group_name in self.data.groups:
                color = self.group_colors[group_name]
                group_tokens = [t for t in self.data.tokens if t['group'] == group_name]
                if not group_tokens:
                    continue
                final_pts = np.stack([t['trajectory'][min(max_layer-1, self.n_layers-1)] for t in group_tokens])
                cx = final_pts[:, d0].mean()
                cy = final_pts[:, d1].mean()
                cz = final_pts[:, d2].mean()
                self.ax.scatter([cx], [cy], [cz], c=[color], s=300, marker="*",
                              edgecolors="k", linewidths=1.5, zorder=10,
                              label=group_name)

        # Axis labels
        self.ax.set_xlabel(f"dim {d0}", fontsize=10)
        self.ax.set_ylabel(f"dim {d1}", fontsize=10)
        self.ax.set_zlabel(f"dim {d2}", fontsize=10)

        # Title
        n_groups = len(self.data.groups)
        color_mode = "group" if self.color_by_group else "position"
        layer_info = f"layers 0-{max_layer-1}/{self.n_layers-1}" if not self.show_all_layers else f"all {self.n_layers} layers"
        anim_info = " [ANIMATING]" if self.animating else ""
        title = (
            f"X=dim {d0}  Y=dim {d1}  Z=dim {d2}  |  "
            f"order: {self.current_order_name}  |  "
            f"{n_groups} group(s), {n_tokens} tokens\n"
            f"{layer_info}{anim_info}  |  "
            f"SPACE=animate  G=color({color_mode})  A=arrows  T=labels  C=centroids"
        )
        self.ax.set_title(title, fontsize=8.5, fontfamily="monospace")

        if self.show_centroids:
            self.ax.legend(loc="upper left", fontsize=8)

        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def run(self):
        d = self.dims
        print("\n" + "=" * 70)
        print("  TOKEN-LEVEL 3D TRAJECTORY VIEWER")
        print("  Every token = its own trajectory through layers")
        print("  Watch attractors form in real-time with SPACE")
        print("=" * 70)
        print(f"  Layers: {self.n_layers}  |  Total dims: {self.d_model}")
        print(f"  Tokens: {len(self.data.tokens)}  |  Groups: {len(self.data.groups)}")
        print(f"  Starting dims: X={d[0]}, Y={d[1]}, Z={d[2]}")
        print(f"  Ordering: {self.current_order_name}")
        print()
        print("  DIMENSION NAVIGATION:")
        print("    ← / →          X-axis dim ±1")
        print("    ↑ / ↓          Y-axis dim ±1")
        print("    PgUp / PgDn    Z-axis dim ±1")
        print("    Shift+key      Jump ±10")
        print()
        print("  ORDERING MODES:")
        print("    N              Sequential (0, 1, 2, 3, ...)")
        print("    V              By variance (most spread first)")
        print("    K              By convergence (most attractor-like first)")
        print()
        print("  ANIMATION:")
        print("    SPACE          Start/stop live trajectory painting")
        print("    B              Reset animation to layer 0")
        print("    +/-            Speed up / slow down (current: 0.1s/layer)")
        print()
        print("  DISPLAY:")
        print("    G              Toggle group/position coloring")
        print("    A              Toggle arrows")
        print("    T              Toggle token labels")
        print("    C              Toggle centroids")
        print("    1-9            Highlight token trajectory")
        print("    0              Clear highlight")
        print("    P              Save PNG")
        print("    R              Reset view")
        print("    Q / Esc        Quit")
        print()
        print("  MOUSE:")
        print("    Left drag      Trackball rotation")
        print("    Scroll         Zoom")
        print("    Hover          Show token info (sentence, POS, group)")
        print("=" * 70 + "\n")
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Token-Level 3D Residual Stream Trajectory Viewer with Live Animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 viewer.py attractor_data/paris_multilingual
  python3 viewer.py attractor_data/
  python3 viewer.py attractor_data/ --groups berlin_multilingual paris_multilingual
  python3 viewer.py attractor_data/berlin_multilingual --dims 47,203,512
        """
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

    # Load token-level data
    data = TokenTrajectoryData(data_dir, groups=args.groups)

    # Launch viewer
    viewer = TokenViewer3D(data, initial_dims=initial_dims)
    viewer.run()


if __name__ == "__main__":
    main()
