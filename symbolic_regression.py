# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas>=2.0",
#     "numpy>=1.24",
#     "scikit-learn>=1.3",
#     "pysr>=1.0",
#     "matplotlib>=3.7",
# ]
# ///
"""
Symbolic Regression: Approximate ENTIRE layer transformation at once.

Models ALL output dimensions simultaneously. Each output dim gets its own
symbolic equation, but they all share the same input feature set and are
fitted together.

The plot shows the FULL activation matrix: all dims × all samples.
Single matplotlib window.
"""

import argparse
import json
import sys
import os
import shutil
from pathlib import Path

# =============================================================================
# Auto-restart under `uv run`
# =============================================================================

def _ensure_uv_run():
    if os.environ.get("_UV_RUN_ACTIVE") == "1":
        return
    uv_path = shutil.which("uv")
    if uv_path is None:
        import subprocess
        subprocess.run(
            ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
            check=True
        )
        for p in [os.path.expanduser("~/.local/bin"), os.path.expanduser("~/.cargo/bin")]:
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        uv_path = shutil.which("uv")
        if uv_path is None:
            sys.exit(1)

    script_path = os.path.abspath(__file__)
    import subprocess
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
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent / "results_single_word_larger_deepseek"

LAYER_GROUPS = {
    "early": list(range(1, 22)),
    "mid": list(range(22, 43)),
    "late": list(range(43, 64)),
    "all": list(range(1, 64)),
}


# =============================================================================
# Data Loading
# =============================================================================

def discover_experiments(results_dir: Path) -> list[str]:
    return sorted(
        d.name for d in results_dir.iterdir()
        if d.is_dir() and (d / "all_token_streams").exists()
    )


def discover_layers(experiment_dir: Path) -> list[int]:
    streams_dir = experiment_dir / "all_token_streams"
    layers = []
    for f in streams_dir.glob("layer_*.csv"):
        layers.append(int(f.stem.split("_")[1]))
    return sorted(layers)


def load_layer_data(experiment_dir: Path, layer_idx: int) -> pd.DataFrame:
    path = experiment_dir / "all_token_streams" / f"layer_{layer_idx:03d}.csv"
    return pd.read_csv(path)


def get_dim_columns(df: pd.DataFrame) -> list[str]:
    meta_cols = {"prompt_idx", "token_pos", "token_text"}
    return [c for c in df.columns if c not in meta_cols]


# =============================================================================
# Build full layer I/O
# =============================================================================

def build_layer_io_raw(experiment_dir: Path, layer_idx: int) -> dict:
    input_layer = layer_idx - 1
    output_layer = layer_idx

    df_in = load_layer_data(experiment_dir, input_layer)
    df_out = load_layer_data(experiment_dir, output_layer)

    merge_keys = ["prompt_idx", "token_pos"]
    df_in = df_in.sort_values(merge_keys).reset_index(drop=True)
    df_out = df_out.sort_values(merge_keys).reset_index(drop=True)

    assert len(df_in) == len(df_out), f"Size mismatch: {len(df_in)} vs {len(df_out)}"
    assert (df_in["prompt_idx"].values == df_out["prompt_idx"].values).all()
    assert (df_in["token_pos"].values == df_out["token_pos"].values).all()

    dim_cols_in = get_dim_columns(df_in)
    dim_cols_out = get_dim_columns(df_out)

    X_raw = df_in[dim_cols_in].values.astype(np.float32)
    Y_raw = df_out[dim_cols_out].values.astype(np.float32)

    meta = df_in[["prompt_idx", "token_pos"]].copy()
    if "token_text" in df_in.columns:
        meta["token_text"] = df_in["token_text"]

    return {
        "X_raw": X_raw,
        "Y_raw": Y_raw,
        "input_dim_names": dim_cols_in,
        "output_dim_names": dim_cols_out,
        "meta": meta,
        "input_layer": input_layer,
        "output_layer": output_layer,
        "n_samples": len(df_in),
        "n_dims": len(dim_cols_in),
    }


# =============================================================================
# Feature selection for the whole layer (shared features)
# =============================================================================

def select_global_features(
    X_raw: np.ndarray,
    Y_raw: np.ndarray,
    dim_names: list[str],
    n_features: int,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Select input dimensions that are most informative across ALL output dims.
    Uses mean absolute correlation across all outputs.
    """
    n_samples, n_out = Y_raw.shape
    X_centered = X_raw - X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0) + 1e-12

    # Mean abs correlation with all output dims
    Y_centered = Y_raw - Y_raw.mean(axis=0)
    Y_std = Y_raw.std(axis=0) + 1e-12

    # (n_in,) — average importance across all outputs
    importance = np.zeros(X_raw.shape[1])
    for j in range(n_out):
        corr = np.abs((X_centered * Y_centered[:, j:j+1]).mean(axis=0)) / (X_std * Y_std[j])
        importance += corr

    importance /= n_out
    top_idxs = np.argsort(importance)[::-1][:n_features].tolist()

    return X_raw[:, top_idxs], [dim_names[i] for i in top_idxs], top_idxs


# =============================================================================
# Single-window visualization — shows ALL dims at once
# =============================================================================

class SingleWindowPlotter:
    def __init__(self):
        self.fig = None

    def ensure_figure(self):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig = plt.figure(figsize=(22, 14))
        return self.fig

    def plot_full_layer(
        self,
        Y_true: np.ndarray,
        Y_pred: np.ndarray,
        layer_label: str,
        r2_per_dim: np.ndarray,
        equations: list[str],
        feature_names: list[str],
    ):
        """
        Show the ENTIRE layer approximation:
        - Heatmap of real activations (all samples × all dims)
        - Heatmap of predicted activations
        - Heatmap of error
        - R² distribution across all dims
        - Signal overlay for ALL dims superimposed
        """
        fig = self.ensure_figure()
        fig.clf()

        n_samples, n_dims = Y_true.shape
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

        # Shared color limits
        vmin = min(Y_true.min(), Y_pred.min())
        vmax = max(Y_true.max(), Y_pred.max())

        # --- Panel 1: Real activation heatmap (samples × dims) ---
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(Y_true.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_xlabel(f"Sample (0..{n_samples-1})", fontsize=9)
        ax1.set_ylabel(f"Dimension (0..{n_dims-1})", fontsize=9)
        ax1.set_title(f"REAL — all {n_dims} dims × {n_samples} samples", fontsize=10, fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # --- Panel 2: Predicted activation heatmap ---
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(Y_pred.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_xlabel(f"Sample (0..{n_samples-1})", fontsize=9)
        ax2.set_ylabel(f"Dimension (0..{n_dims-1})", fontsize=9)
        ax2.set_title(f"PREDICTED — all {n_dims} dims × {n_samples} samples", fontsize=10, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # --- Panel 3: Error heatmap ---
        ax3 = fig.add_subplot(gs[0, 2])
        error = np.abs(Y_true - Y_pred)
        im3 = ax3.imshow(error.T, aspect='auto', cmap='hot')
        ax3.set_xlabel(f"Sample (0..{n_samples-1})", fontsize=9)
        ax3.set_ylabel(f"Dimension (0..{n_dims-1})", fontsize=9)
        ax3.set_title(f"|Error| — MAE={error.mean():.4f}", fontsize=10, fontweight='bold')
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # --- Panel 4: R² distribution across all dims ---
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(r2_per_dim, bins=80, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.3)
        ax4.axvline(np.median(r2_per_dim), color='red', linewidth=2, linestyle='--',
                    label=f'Median R²={np.median(r2_per_dim):.4f}')
        ax4.axvline(np.mean(r2_per_dim), color='orange', linewidth=2, linestyle=':',
                    label=f'Mean R²={np.mean(r2_per_dim):.4f}')
        ax4.set_xlabel("R² per dimension", fontsize=10)
        ax4.set_ylabel("Count", fontsize=10)
        ax4.set_title(f"R² across all {n_dims} dims", fontsize=10, fontweight='bold')
        ax4.legend(fontsize=9)

        # --- Panel 5: Signal overlay — ALL dims superimposed ---
        ax5 = fig.add_subplot(gs[1, 1:3])
        # Plot every dim's real and predicted signal (thin lines, alpha low)
        sample_idx = np.arange(n_samples)
        # Subsample dims for visibility if too many
        step = max(1, n_dims // 200)
        for d in range(0, n_dims, step):
            ax5.plot(sample_idx, Y_true[:, d], color='steelblue', alpha=0.05, linewidth=0.5)
            ax5.plot(sample_idx, Y_pred[:, d], color='orangered', alpha=0.05, linewidth=0.5)
        # Plot mean signal prominently
        ax5.plot(sample_idx, Y_true.mean(axis=1), color='blue', linewidth=2, label='Real (mean over dims)')
        ax5.plot(sample_idx, Y_pred.mean(axis=1), color='red', linewidth=2, label='Predicted (mean over dims)', linestyle='--')
        ax5.set_xlabel(f"Sample index (all {n_samples})", fontsize=10)
        ax5.set_ylabel("Activation", fontsize=10)
        ax5.set_title(f"ALL dims overlaid — every {step}th dim shown", fontsize=10, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.2)

        # --- Panel 6: R² sorted (which dims are well/poorly approximated) ---
        ax6 = fig.add_subplot(gs[2, 0])
        sorted_r2 = np.sort(r2_per_dim)[::-1]
        ax6.plot(sorted_r2, color='darkgreen', linewidth=1.5)
        ax6.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='R²=0.9')
        ax6.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='R²=0.5')
        ax6.set_xlabel("Dimension rank", fontsize=10)
        ax6.set_ylabel("R²", fontsize=10)
        ax6.set_title(f"R² sorted — {(r2_per_dim > 0.9).sum()}/{n_dims} dims > 0.9", fontsize=10, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.2)

        # --- Panel 7: Per-sample total error ---
        ax7 = fig.add_subplot(gs[2, 1])
        sample_mse = np.mean((Y_true - Y_pred) ** 2, axis=1)
        ax7.bar(sample_idx, sample_mse, color='coral', alpha=0.8)
        ax7.set_xlabel("Sample index", fontsize=10)
        ax7.set_ylabel("MSE (across all dims)", fontsize=10)
        ax7.set_title(f"Per-sample MSE — {n_samples} samples", fontsize=10, fontweight='bold')
        ax7.grid(True, alpha=0.2, axis='y')

        # --- Panel 8: Feature usage info ---
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        info_text = f"Layer: {layer_label}\n"
        info_text += f"Dims: {n_dims} | Samples: {n_samples}\n"
        info_text += f"Features: {feature_names[:6]}\n"
        if len(feature_names) > 6:
            info_text += f"  + {len(feature_names)-6} more\n"
        info_text += f"\nR² stats:\n"
        info_text += f"  Mean:   {np.mean(r2_per_dim):.4f}\n"
        info_text += f"  Median: {np.median(r2_per_dim):.4f}\n"
        info_text += f"  >0.99:  {(r2_per_dim > 0.99).sum()}/{n_dims}\n"
        info_text += f"  >0.9:   {(r2_per_dim > 0.9).sum()}/{n_dims}\n"
        info_text += f"  >0.5:   {(r2_per_dim > 0.5).sum()}/{n_dims}\n"
        info_text += f"  <0:     {(r2_per_dim < 0).sum()}/{n_dims}\n"
        # Show a few example equations
        info_text += f"\nExample equations:\n"
        for i, eq in enumerate(equations[:5]):
            eq_short = eq if len(eq) < 60 else eq[:57] + "..."
            info_text += f"  d{i}: {eq_short}\n"
        if len(equations) > 5:
            info_text += f"  ... +{len(equations)-5} more"

        ax8.text(0.05, 0.95, info_text, transform=ax8.transAxes, fontsize=8,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle(layer_label, fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.5)


# =============================================================================
# Core: Fit ALL dims at once using PySR multi-output
# =============================================================================

def fit_all_dims(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: list[str],
    output_dim_names: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fit symbolic equations for ALL output dimensions.

    PySR supports multi-output natively when you pass a 2D y.
    Returns: (Y_pred, r2_per_dim, equations_list)
    """
    from pysr import PySRRegressor

    n_samples, n_dims = Y.shape
    print(f"    [SR] Fitting ALL {n_dims} output dims simultaneously")
    print(f"    [SR] X: {X.shape}, Y: {Y.shape}")
    print(f"    [SR] Features: {feature_names}")

    # Normalize
    x_scaler = StandardScaler()
    X_norm = x_scaler.fit_transform(X)

    y_scaler = StandardScaler()
    Y_norm = y_scaler.fit_transform(Y)

    model = PySRRegressor(
        niterations=args.iterations,
        population_size=args.population,
        parsimony=args.parsimony,
        maxsize=args.maxsize,
        binary_operators=args.binary_ops.split(","),
        unary_operators=args.unary_ops.split(","),
        progress=True,
        batching=True,
        batch_size=min(256, n_samples),
        random_state=42,
        update=False,
    )

    # PySR multi-output: pass Y as 2D array
    model.fit(X_norm, Y_norm, variable_names=feature_names)

    # Predict all dims
    Y_pred_norm = model.predict(X_norm)
    if Y_pred_norm.ndim == 1:
        # Single output fallback
        Y_pred_norm = Y_pred_norm.reshape(-1, 1)

    # If PySR returned fewer outputs than expected (it fits one equation for all),
    # we need to handle this
    if Y_pred_norm.shape[1] != n_dims:
        # PySR multi-output: it returns one prediction per output column
        # This means it found one shared equation. Broadcast.
        Y_pred_norm = np.broadcast_to(Y_pred_norm, (n_samples, n_dims))

    Y_pred = y_scaler.inverse_transform(Y_pred_norm)

    # R² per dim
    r2_per_dim = np.zeros(n_dims)
    for d in range(n_dims):
        ss_res = np.sum((Y[:, d] - Y_pred[:, d]) ** 2)
        ss_tot = np.sum((Y[:, d] - Y[:, d].mean()) ** 2)
        r2_per_dim[d] = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # Get equations
    equations = []
    try:
        # Multi-output: model.equations_ is a list of DataFrames
        if isinstance(model.equations_, list):
            for eq_df in model.equations_:
                best_idx = eq_df["loss"].idxmin()
                equations.append(str(eq_df.loc[best_idx, "equation"]))
        else:
            best_idx = model.equations_["loss"].idxmin()
            eq_str = str(model.equations_.loc[best_idx, "equation"])
            equations = [eq_str] * n_dims
    except Exception as e:
        equations = [f"(error: {e})"] * n_dims

    return Y_pred, r2_per_dim, equations


def fit_all_dims_individually(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: list[str],
    output_dim_names: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fallback: fit each dim individually but in a tight loop.
    More equations but guaranteed to work with any PySR version.
    """
    from pysr import PySRRegressor

    n_samples, n_dims = Y.shape
    print(f"    [SR] Fitting {n_dims} dims individually (parallel loop)")
    print(f"    [SR] X: {X.shape}, Y: {Y.shape}")

    x_scaler = StandardScaler()
    X_norm = x_scaler.fit_transform(X)

    Y_pred = np.zeros_like(Y)
    r2_per_dim = np.zeros(n_dims)
    equations = []

    for d in range(n_dims):
        y = Y[:, d]
        y_std = y.std()
        if y_std < 1e-10:
            Y_pred[:, d] = y.mean()
            r2_per_dim[d] = 0.0
            equations.append("0")
            continue

        y_norm = (y - y.mean()) / y_std

        model = PySRRegressor(
            niterations=args.iterations,
            population_size=args.population,
            parsimony=args.parsimony,
            maxsize=args.maxsize,
            binary_operators=args.binary_ops.split(","),
            unary_operators=args.unary_ops.split(","),
            progress=False,  # quiet for individual dims
            batching=True,
            batch_size=min(256, n_samples),
            random_state=42,
            update=False,
        )

        model.fit(X_norm, y_norm, variable_names=feature_names)

        y_pred_norm = model.predict(X_norm)
        Y_pred[:, d] = y_pred_norm * y_std + y.mean()

        ss_res = np.sum((y - Y_pred[:, d]) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_per_dim[d] = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        best_idx = model.equations_["loss"].idxmin()
        eq_str = str(model.equations_.loc[best_idx, "equation"])
        equations.append(eq_str)

        if (d + 1) % 100 == 0 or d == n_dims - 1:
            print(f"      [{d+1}/{n_dims}] median R²={np.median(r2_per_dim[:d+1]):.4f}")

    return Y_pred, r2_per_dim, equations


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_layer_spec(spec: str, available_layers: list[int]) -> list[int]:
    spec = spec.strip().lower()
    if spec in LAYER_GROUPS:
        candidates = LAYER_GROUPS[spec]
    else:
        candidates = set()
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                candidates.update(range(int(lo), int(hi) + 1))
            else:
                candidates.add(int(part))
        candidates = sorted(candidates)

    valid = []
    for l in candidates:
        if l in available_layers and (l - 1) in available_layers:
            valid.append(l)
    return valid


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Symbolic regression: approximate ENTIRE layer (all dims at once)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run symbolic_regression.py -l 20
  uv run symbolic_regression.py -l 20 --n-features 12
  uv run symbolic_regression.py -l 20 --residual --mode individual
  uv run symbolic_regression.py --list-experiments
""",
    )

    parser.add_argument("--experiment", "-e", type=str, default=None)
    parser.add_argument("--results-dir", "-r", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--layers", "-l", type=str, default="20")

    parser.add_argument("--residual", action="store_true",
                        help="Model residual (output - input) per dimension")
    parser.add_argument("--n-features", type=int, default=10,
                        help="Number of shared input features (selected globally). Default: 10")
    parser.add_argument("--mode", type=str, default="multi",
                        choices=["multi", "individual"],
                        help="'multi': PySR multi-output (one call, all dims). "
                             "'individual': fit each dim separately. Default: multi")

    # SR parameters
    parser.add_argument("--iterations", "-i", type=int, default=40)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--parsimony", type=float, default=0.0032)
    parser.add_argument("--maxsize", type=int, default=30)
    parser.add_argument("--binary-ops", type=str, default="+,*,/,^,-")
    parser.add_argument("--unary-ops", type=str, default="sin,cos,exp,log,sqrt,tanh,abs")

    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--list-experiments", action="store_true")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if args.list_experiments:
        print("Available experiments:")
        for exp in discover_experiments(results_dir):
            exp_dir = results_dir / exp
            layers = discover_layers(exp_dir)
            print(f"  {exp}  (layers: {min(layers)}-{max(layers)}, count={len(layers)})")
        return

    if args.experiment:
        exp_name = args.experiment
    else:
        experiments = discover_experiments(results_dir)
        if not experiments:
            print(f"ERROR: No experiments in {results_dir}", file=sys.stderr)
            sys.exit(1)
        exp_name = experiments[0]

    exp_dir = results_dir / exp_name
    if not exp_dir.exists():
        print(f"ERROR: {exp_dir} not found", file=sys.stderr)
        sys.exit(1)

    available_layers = discover_layers(exp_dir)
    layers = parse_layer_spec(args.layers, available_layers)

    if not layers:
        print(f"ERROR: No valid layers. Available: {available_layers}", file=sys.stderr)
        sys.exit(1)

    print("=" * 90)
    print("SYMBOLIC REGRESSION — ALL DIMENSIONS AT ONCE")
    print("=" * 90)
    print(f"  Experiment:     {exp_name}")
    print(f"  Layer(s):       {layers}")
    print(f"  Mode:           {args.mode} | {'RESIDUAL' if args.residual else 'FULL OUTPUT'}")
    print(f"  Shared features:{args.n_features} (globally selected input dims)")
    print(f"  SR iterations:  {args.iterations}")
    print(f"  SR maxsize:     {args.maxsize}")
    print("=" * 90)

    plotter = SingleWindowPlotter() if not args.no_plots else None
    if not args.no_plots:
        plt.ion()

    all_results = []

    for layer_idx in layers:
        print(f"\n{'#' * 90}")
        print(f"### LAYER {layer_idx - 1} → {layer_idx} (ALL {5120} DIMS)")
        print(f"{'#' * 90}")

        try:
            io_data = build_layer_io_raw(exp_dir, layer_idx)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        X_raw = io_data["X_raw"]
        Y_raw = io_data["Y_raw"]
        input_dim_names = io_data["input_dim_names"]
        output_dim_names = io_data["output_dim_names"]
        n_samples = io_data["n_samples"]
        n_dims = io_data["n_dims"]

        if args.residual:
            Y_target = Y_raw - X_raw
            mode_label = "residual"
        else:
            Y_target = Y_raw
            mode_label = "full"

        print(f"  Samples: {n_samples}, Dims: {n_dims}")

        # Select shared features (globally informative input dims)
        X_sel, feat_names, feat_idxs = select_global_features(
            X_raw, Y_target, input_dim_names, args.n_features)
        print(f"  Samples: {n_samples}, Dims: {n_dims}")
        print(f"  Shared input features: {feat_names}")

        # Normalize inputs (shared across all targets)
        x_scaler = StandardScaler()
        X_norm = x_scaler.fit_transform(X_sel)

        # Fit ALL dims at once
        if args.mode == "multi":
            Y_pred, r2_per_dim, equations = fit_all_dims(
                X_sel, Y_target, feat_names, output_dim_names, args
            )
        else:
            Y_pred, r2_per_dim, equations = fit_all_dims_individually(
                X_sel, Y_target, feat_names, output_dim_names, args
            )

        print(f"\n  RESULTS:")
        print(f"    Mean R²:   {np.mean(r2_per_dim):.4f}")
        print(f"    Median R²: {np.median(r2_per_dim):.4f}")
        print(f"    Dims R²>0.9: {(r2_per_dim > 0.9).sum()}/{n_dims}")
        print(f"    Dims R²>0.5: {(r2_per_dim > 0.5).sum()}/{n_dims}")

        # Plot full layer in single window
        if plotter is not None:
            plotter.plot_full_layer(
                Y_true=Y_target,
                Y_pred=Y_pred,
                layer_label=f"Layer {layer_idx-1}→{layer_idx} ({mode_label}) | {exp_name}",
                r2_per_dim=r2_per_dim,
                equations=equations,
                feature_names=feat_names,
            )

        layer_result = {
            "input_layer": layer_idx - 1,
            "output_layer": layer_idx,
            "mode": mode_label,
            "n_dims": n_dims,
            "n_samples": n_samples,
            "shared_features": feat_names,
            "mean_r2": float(np.mean(r2_per_dim)),
            "median_r2": float(np.median(r2_per_dim)),
            "dims_above_0.9": int((r2_per_dim > 0.9).sum()),
            "dims_above_0.5": int((r2_per_dim > 0.5).sum()),
            "r2_per_dim": r2_per_dim.tolist(),
            "equations": equations,
        }
        all_results.append(layer_result)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY: FULL LAYER APPROXIMATIONS")
    print("=" * 90)
    for res in all_results:
        print(f"\n  Layer {res['input_layer']} → {res['output_layer']} ({res['mode']}):")
        print(f"    {res['n_dims']} dims, {res['n_samples']} samples")
        print(f"    Shared features: {res['shared_features']}")
        print(f"    Mean R²={res['mean_r2']:.4f}  Median R²={res['median_r2']:.4f}")
        print(f"    Dims R²>0.9: {res['dims_above_0.9']}/{res['n_dims']}")
        print(f"    Dims R²>0.5: {res['dims_above_0.5']}/{res['n_dims']}")
        print(f"    Example equations:")
        for i, eq in enumerate(res["equations"][:10]):
            r2_val = res["r2_per_dim"][i] if i < len(res["r2_per_dim"]) else 0
            print(f"      dim_{i:04d} = {eq}  (R²={r2_val:.4f})")
        if len(res["equations"]) > 10:
            print(f"      ... +{len(res['equations'])-10} more")
    print("=" * 90)

    # Save
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"experiment": exp_name, "results": all_results}, f, indent=2)
        print(f"\nSaved to {output_path}")

    # Keep window open at end
    if not args.no_plots:
        plt.ioff()
        plt.show(block=True)


if __name__ == "__main__":
    main()
