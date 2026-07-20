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
Symbolic Regression: Approximate real LLM layer transformations.

NO PCA. Works directly on the real activation dimensions.
For each output dimension, finds a symbolic equation using the most
correlated input dimensions as features.

Single matplotlib window that updates in-place.
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
# Build real input→output data (NO PCA)
# =============================================================================

def select_features_for_target(
    X_raw: np.ndarray,
    y: np.ndarray,
    dim_names: list[str],
    n_features: int,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Select the top-N most correlated input dimensions for a given output target.
    Returns the selected X subset, their real dimension names, and indices.
    """
    # Correlation of each input dim with the target
    y_centered = y - y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        # Zero variance target, just pick first N
        idxs = list(range(n_features))
    else:
        correlations = np.abs(
            (X_raw - X_raw.mean(axis=0)) * y_centered[:, None]
        ).mean(axis=0) / (X_raw.std(axis=0) * y_std + 1e-12)
        idxs = np.argsort(correlations)[::-1][:n_features].tolist()

    return X_raw[:, idxs], [dim_names[i] for i in idxs], idxs


def build_layer_io_raw(
    experiment_dir: Path,
    layer_idx: int,
) -> dict:
    """
    Load layer L-1 (input) and layer L (output). Return raw arrays.
    No PCA, no compression.
    """
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
# Single-window visualization (updates in place)
# =============================================================================

class SingleWindowPlotter:
    """Manages a single matplotlib figure that updates in-place."""

    def __init__(self):
        self.fig = None
        self.initialized = False

    def ensure_figure(self):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig = plt.figure(figsize=(20, 12))
            self.initialized = False
        return self.fig

    def plot_sr_result(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: list[str],
        target_name: str,
        equation_str: str,
        layer_label: str,
        pareto_complexities: np.ndarray = None,
        pareto_losses: np.ndarray = None,
        best_complexity: int = None,
        best_loss: float = None,
    ):
        """Update the single window with results for one target dimension."""
        fig = self.ensure_figure()
        fig.clf()

        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        n_samples = len(y_true)
        sample_indices = np.arange(n_samples)

        # --- Panel 1: Real vs Predicted scatter ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_true, y_pred, alpha=0.4, s=12, c='steelblue', edgecolors='none')
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect')
        ax1.set_xlabel("Real activation", fontsize=10)
        ax1.set_ylabel("Predicted", fontsize=10)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax1.set_title(f"R² = {r2:.4f}", fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2)

        # --- Panel 2: FULL signal overlay (ALL samples, not just first N) ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(sample_indices, y_true, label='Real', color='steelblue', linewidth=0.6, alpha=0.8)
        ax2.plot(sample_indices, y_pred, label='Predicted', color='orangered', linewidth=0.6, alpha=0.7)
        ax2.set_xlabel("Sample index (all data)", fontsize=10)
        ax2.set_ylabel("Activation value", fontsize=10)
        ax2.set_title(f"Full signal overlay — {n_samples} samples", fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2)

        # --- Panel 3: Residual distribution ---
        ax3 = fig.add_subplot(gs[0, 2])
        residuals = y_true - y_pred
        ax3.hist(residuals, bins=60, color='coral', alpha=0.7, edgecolor='black', linewidth=0.3)
        ax3.axvline(0, color='black', linewidth=1, linestyle='--')
        mae = np.mean(np.abs(residuals))
        ax3.set_xlabel("Residual", fontsize=10)
        ax3.set_ylabel("Count", fontsize=10)
        ax3.set_title(f"Residuals — MAE={mae:.4f}", fontsize=11, fontweight='bold')

        # --- Panel 4: Scatter vs most important input feature ---
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(X[:, 0], y_true, alpha=0.3, s=10, c='steelblue', edgecolors='none', label='Real')
        ax4.scatter(X[:, 0], y_pred, alpha=0.2, s=10, c='orangered', edgecolors='none', label='Predicted')
        ax4.set_xlabel(f"{feature_names[0]} (real dim)", fontsize=10)
        ax4.set_ylabel(f"{target_name}", fontsize=10)
        ax4.set_title(f"vs top feature: {feature_names[0]}", fontsize=10, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.2)

        # --- Panel 5: Scatter vs second feature ---
        ax5 = fig.add_subplot(gs[1, 1])
        if X.shape[1] > 1:
            ax5.scatter(X[:, 1], y_true, alpha=0.3, s=10, c='steelblue', edgecolors='none', label='Real')
            ax5.scatter(X[:, 1], y_pred, alpha=0.2, s=10, c='orangered', edgecolors='none', label='Predicted')
            ax5.set_xlabel(f"{feature_names[1]} (real dim)", fontsize=10)
            ax5.set_ylabel(f"{target_name}", fontsize=10)
            ax5.set_title(f"vs feature: {feature_names[1]}", fontsize=10, fontweight='bold')
            ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.2)

        # --- Panel 6: Pareto front ---
        ax6 = fig.add_subplot(gs[1, 2])
        if pareto_complexities is not None and pareto_losses is not None:
            ax6.semilogy(pareto_complexities, pareto_losses, 'o-', color='darkgreen', markersize=6)
            if best_complexity is not None and best_loss is not None:
                ax6.semilogy(best_complexity, best_loss, '*', color='red', markersize=15, label='Best')
            ax6.set_xlabel("Complexity", fontsize=10)
            ax6.set_ylabel("Loss (log)", fontsize=10)
            ax6.set_title("Pareto Front", fontsize=11, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        # Equation text at bottom
        eq_display = equation_str if len(equation_str) < 120 else equation_str[:117] + "..."
        fig.text(0.5, 0.01, f"{target_name} = {eq_display}",
                 ha='center', fontsize=9, style='italic',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle(
            f"{layer_label} | Target: {target_name} | Features: {feature_names[:5]}{'...' if len(feature_names) > 5 else ''}",
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)


# =============================================================================
# Core SR — no PCA, real dimensions
# =============================================================================

def run_symbolic_regression_raw(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    label: str,
    args: argparse.Namespace,
):
    """Run PySR on real dimension data. No intermediate windows."""
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=args.iterations,
        population_size=args.population,
        parsimony=args.parsimony,
        maxsize=args.maxsize,
        binary_operators=args.binary_ops.split(","),
        unary_operators=args.unary_ops.split(","),
        progress=True,
        batching=True,
        batch_size=min(256, len(X)),
        random_state=42,
        # CRITICAL: prevent PySR from opening its own plot windows
        update=False,
    )

    print(f"    [SR] Fitting: {label}")
    print(f"    [SR] X shape: {X.shape}, y shape: {y.shape}")
    print(f"    [SR] Real features: {feature_names}")
    model.fit(X, y, variable_names=feature_names)
    return model


def select_target_dims(
    Y_raw: np.ndarray,
    output_dim_names: list[str],
    n_targets: int,
    method: str = "variance",
) -> list[tuple[int, str]]:
    """
    Select which output dimensions to approximate.
    Methods:
      - "variance": pick dims with highest variance (most signal)
      - "change": pick dims where input→output change is largest
    Returns list of (index, name) tuples.
    """
    if method == "variance":
        variances = np.var(Y_raw, axis=0)
        top_idxs = np.argsort(variances)[::-1][:n_targets]
    else:
        top_idxs = list(range(min(n_targets, Y_raw.shape[1])))

    return [(int(i), output_dim_names[i]) for i in top_idxs]


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
        description="Symbolic regression on REAL layer dimensions (no PCA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run symbolic_regression.py -l 20
  uv run symbolic_regression.py -l 20 --n-features 8 --n-targets 10
  uv run symbolic_regression.py -l 10-15 --residual
  uv run symbolic_regression.py --list-experiments
""",
    )

    parser.add_argument("--experiment", "-e", type=str, default=None)
    parser.add_argument("--results-dir", "-r", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--layers", "-l", type=str, default="20",
                        help="Layer spec: number, range, or group name")

    # What to model
    parser.add_argument("--residual", action="store_true",
                        help="Model residual (output - input) per dimension")
    parser.add_argument("--n-features", type=int, default=8,
                        help="Number of input dimensions to use as SR features (selected by correlation). Default: 8")
    parser.add_argument("--n-targets", type=int, default=5,
                        help="Number of output dimensions to approximate. Default: 5")
    parser.add_argument("--target-method", type=str, default="variance",
                        choices=["variance"],
                        help="How to select target dims. Default: variance (highest variance)")

    # SR parameters
    parser.add_argument("--iterations", "-i", type=int, default=40)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--parsimony", type=float, default=0.0032)
    parser.add_argument("--maxsize", type=int, default=30)
    parser.add_argument("--binary-ops", type=str, default="+,*,/,^,-")
    parser.add_argument("--unary-ops", type=str, default="sin,cos,exp,log,sqrt,tanh,abs")

    # Output
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

    # Resolve experiment
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

    # Print config
    print("=" * 90)
    print("SYMBOLIC REGRESSION — REAL DIMENSIONS (NO PCA)")
    print("=" * 90)
    print(f"  Experiment:     {exp_name}")
    print(f"  Layer(s):       {layers}")
    print(f"  Mode:           {'RESIDUAL' if args.residual else 'FULL OUTPUT'}")
    print(f"  Input features: {args.n_features} (top correlated real dims per target)")
    print(f"  Output targets: {args.n_targets} (highest variance dims)")
    print(f"  SR iterations:  {args.iterations}")
    print(f"  SR maxsize:     {args.maxsize}")
    print("=" * 90)

    # Single window plotter
    plotter = SingleWindowPlotter() if not args.no_plots else None

    # Enable interactive mode — single window, no blocking
    if not args.no_plots:
        plt.ion()

    all_results = []

    for layer_idx in layers:
        print(f"\n{'#' * 90}")
        print(f"### LAYER {layer_idx - 1} → {layer_idx}")
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

        # If residual mode, target is (output - input) per dimension
        if args.residual:
            Y_target = Y_raw - X_raw
            mode_label = "residual"
        else:
            Y_target = Y_raw
            mode_label = "full"

        print(f"  Samples: {io_data['n_samples']}, Dims: {io_data['n_dims']}")

        # Select target dimensions
        targets = select_target_dims(Y_target, output_dim_names, args.n_targets, args.target_method)
        print(f"  Target dims (by {args.target_method}): {[name for _, name in targets]}")

        layer_result = {
            "input_layer": layer_idx - 1,
            "output_layer": layer_idx,
            "mode": mode_label,
            "equations": [],
        }

        for target_idx, target_name in targets:
            y = Y_target[:, target_idx]

            if np.std(y) < 1e-10:
                print(f"  {target_name}: SKIPPED (zero variance)")
                continue

            # Select best input features FOR THIS specific target
            X_sel, feat_names, feat_idxs = select_features_for_target(
                X_raw, y, input_dim_names, args.n_features
            )

            # Normalize for SR stability (but names stay real)
            x_scaler = StandardScaler()
            X_norm = x_scaler.fit_transform(X_sel)
            y_scaler = StandardScaler()
            y_norm = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

            full_label = f"L{layer_idx-1}→L{layer_idx}_{target_name}"
            print(f"\n  --- {full_label} ---")
            print(f"      Features: {feat_names}")
            print(f"      y range: [{y.min():.3f}, {y.max():.3f}], std={y.std():.4f}")

            # Run SR
            model = run_symbolic_regression_raw(X_norm, y_norm, feat_names, full_label, args)

            # Get predictions (in original scale)
            y_pred_norm = model.predict(X_norm)
            y_pred = y_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).ravel()

            # Metrics
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            best_eq = str(model.sympy())
            eqs_df = model.equations_
            best_row = eqs_df.iloc[eqs_df["loss"].idxmin()]

            # Print
            print(f"  ✓ {target_name} = {best_eq}")
            print(f"    R²={r2:.4f}  loss={best_row['loss']:.4e}  complexity={int(best_row['complexity'])}")

            # Plot in single window
            if plotter is not None:
                plotter.plot_sr_result(
                    X=X_sel,
                    y_true=y,
                    y_pred=y_pred,
                    feature_names=feat_names,
                    target_name=target_name,
                    equation_str=best_eq,
                    layer_label=f"Layer {layer_idx-1}→{layer_idx} ({mode_label})",
                    pareto_complexities=eqs_df["complexity"].values,
                    pareto_losses=eqs_df["loss"].values,
                    best_complexity=int(best_row["complexity"]),
                    best_loss=float(best_row["loss"]),
                )

            eq_info = {
                "target": target_name,
                "equation": best_eq,
                "features": feat_names,
                "loss": float(best_row["loss"]),
                "r2": float(r2),
                "complexity": int(best_row["complexity"]),
            }
            layer_result["equations"].append(eq_info)

        all_results.append(layer_result)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY: SYMBOLIC EQUATIONS (REAL DIMENSIONS)")
    print("=" * 90)
    for res in all_results:
        print(f"\n  Layer {res['input_layer']} → {res['output_layer']} ({res['mode']}):")
        for eq_info in res["equations"]:
            print(f"    {eq_info['target']} = {eq_info['equation']}")
            print(f"      features={eq_info['features']}")
            print(f"      R²={eq_info['r2']:.4f}  loss={eq_info['loss']:.4e}  complexity={eq_info['complexity']}")
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
