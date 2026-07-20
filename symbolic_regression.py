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
Symbolic regression on LLM layer activations using PySR.
With live visualization of real vs predicted data, loss curves, and full equations.

Bootstraps itself via `uv run symbolic_regression.py`.
"""

import argparse
import json
import sys
import os
import shutil
from pathlib import Path

# =============================================================================
# Auto-restart under `uv run` (self-bootstrapping)
# =============================================================================

def _ensure_uv_run():
    """If not already running under uv, re-exec ourselves with uv run."""
    if os.environ.get("_UV_RUN_ACTIVE") == "1":
        return
    uv_path = shutil.which("uv")
    if uv_path is None:
        print("uv not found. Installing...")
        import subprocess
        subprocess.run(
            ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
            check=True
        )
        for p in [
            os.path.expanduser("~/.local/bin"),
            os.path.expanduser("~/.cargo/bin"),
        ]:
            if p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
        uv_path = shutil.which("uv")
        if uv_path is None:
            print("ERROR: uv installation failed")
            sys.exit(1)

    script_path = os.path.abspath(__file__)
    import subprocess
    cmd = [uv_path, "run", script_path] + sys.argv[1:]
    env = os.environ.copy()
    env["_UV_RUN_ACTIVE"] = "1"
    print(f"Bootstrapping: {' '.join(cmd)}")
    if sys.platform == "win32":
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    else:
        os.execvpe(uv_path, cmd, env)

_ensure_uv_run()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent / "results_single_word_larger_deepseek"

LAYER_GROUPS = {
    "early": list(range(0, 22)),
    "mid": list(range(22, 43)),
    "late": list(range(43, 64)),
    "first-half": list(range(0, 32)),
    "second-half": list(range(32, 64)),
    "all": list(range(0, 64)),
}


def discover_experiments(results_dir: Path) -> list[str]:
    return sorted(
        d.name for d in results_dir.iterdir() if d.is_dir() and (d / "all_token_streams").exists()
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


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    meta_cols = ["prompt_idx", "token_pos", "token_text"]
    X = df[["prompt_idx", "token_pos"]].values.astype(float)
    y = df.drop(columns=meta_cols).values.astype(float)
    token_texts = df["token_text"].tolist()
    return X, y, token_texts


def build_cross_layer_matrix(
    experiment_dir: Path, layer_indices: list[int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    all_X, all_y, all_labels = [], [], []
    for layer_idx in layer_indices:
        df = load_layer_data(experiment_dir, layer_idx)
        meta_cols = ["prompt_idx", "token_pos", "token_text"]
        X = df[["prompt_idx", "token_pos"]].values.astype(float)
        layer_col = np.full((len(df), 1), layer_idx)
        X_full = np.hstack([layer_col, X])
        y = df.drop(columns=meta_cols).values.astype(float)
        labels = [f"L{layer_idx}|p{r.prompt_idx}|{r.token_text}" for _, r in df.iterrows()]
        all_X.append(X_full)
        all_y.append(y)
        all_labels.extend(labels)
    return np.vstack(all_X), np.vstack(all_y), all_labels


def apply_pca(y: np.ndarray, n_components: int, scaler: StandardScaler | None = None) -> tuple[PCA, np.ndarray, StandardScaler]:
    if scaler is None:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y)
    else:
        y_scaled = scaler.transform(y)
    pca = PCA(n_components=min(n_components, y.shape[1], y.shape[0]))
    y_pca = pca.fit_transform(y_scaled)
    return pca, y_pca, scaler


# =============================================================================
# Visualization Functions
# =============================================================================

def print_full_equations_table(model):
    """Print the full Pareto front of equations without truncation."""
    eqs = model.equations_
    print("\n" + "=" * 100)
    print("FULL PARETO FRONT OF EQUATIONS (no truncation)")
    print("=" * 100)
    print(f"{'Complexity':<12} {'Loss':<14} {'Score':<12} Equation")
    print("-" * 100)
    for _, row in eqs.iterrows():
        complexity = int(row["complexity"])
        loss = row["loss"]
        score = row.get("score", 0.0)
        equation = row["equation"]
        print(f"{complexity:<12} {loss:<14.4e} {score:<12.4e} y = {equation}")
    print("=" * 100 + "\n")


def plot_real_vs_predicted(X, y_true, model, feature_names, label, save_path=None):
    """Plot real data vs predicted values from the best symbolic equation."""
    y_pred = model.predict(X)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: Real vs Predicted scatter ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    ax1.set_xlabel("Real Data", fontsize=11)
    ax1.set_ylabel("Predicted (Best Equation)", fontsize=11)
    ax1.set_title(f"Real vs Predicted — {label}", fontsize=12, fontweight='bold')
    ax1.legend()
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    ax1.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Panel 2: Residuals ---
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20, c='coral', edgecolors='none')
    ax2.axhline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_xlabel("Predicted", fontsize=11)
    ax2.set_ylabel("Residual (Real - Predicted)", fontsize=11)
    ax2.set_title("Residuals", fontsize=12, fontweight='bold')
    ax2.text(0.05, 0.92, f"MAE = {np.mean(np.abs(residuals)):.4f}",
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # --- Panel 3: Real & Predicted vs sample index ---
    ax3 = fig.add_subplot(gs[0, 2])
    sort_idx = np.argsort(y_true)
    ax3.plot(y_true[sort_idx], label='Real (sorted)', color='steelblue', linewidth=1.2)
    ax3.plot(y_pred[sort_idx], label='Predicted', color='orangered', linewidth=1.2, alpha=0.8)
    ax3.set_xlabel("Sample (sorted by real value)", fontsize=11)
    ax3.set_ylabel("Value", fontsize=11)
    ax3.set_title("Sorted Overlay", fontsize=12, fontweight='bold')
    ax3.legend()

    # --- Panel 4: Real vs feature[0] (prompt_idx) colored by feature[1] ---
    ax4 = fig.add_subplot(gs[1, 0])
    feat0 = X[:, 0]
    feat1 = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
    sc = ax4.scatter(feat0, y_true, c=feat1, cmap='viridis', alpha=0.6, s=20, edgecolors='none')
    ax4.set_xlabel(feature_names[0], fontsize=11)
    ax4.set_ylabel("Real Value", fontsize=11)
    ax4.set_title(f"Real vs {feature_names[0]}", fontsize=12, fontweight='bold')
    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label(feature_names[1] if len(feature_names) > 1 else "")

    # --- Panel 5: Predicted vs feature[0] colored by feature[1] ---
    ax5 = fig.add_subplot(gs[1, 1])
    sc2 = ax5.scatter(feat0, y_pred, c=feat1, cmap='viridis', alpha=0.6, s=20, edgecolors='none')
    ax5.set_xlabel(feature_names[0], fontsize=11)
    ax5.set_ylabel("Predicted Value", fontsize=11)
    ax5.set_title(f"Predicted vs {feature_names[0]}", fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(sc2, ax=ax5)
    cbar2.set_label(feature_names[1] if len(feature_names) > 1 else "")

    # --- Panel 6: Loss vs Complexity (Pareto front) ---
    ax6 = fig.add_subplot(gs[1, 2])
    eqs = model.equations_
    ax6.semilogy(eqs["complexity"], eqs["loss"], 'o-', color='darkgreen', markersize=6)
    best_idx = eqs["loss"].idxmin()
    ax6.semilogy(eqs.loc[best_idx, "complexity"], eqs.loc[best_idx, "loss"],
                 '*', color='red', markersize=15, label='Best (lowest loss)')
    ax6.set_xlabel("Complexity", fontsize=11)
    ax6.set_ylabel("Loss (log scale)", fontsize=11)
    ax6.set_title("Pareto Front: Loss vs Complexity", fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add the best equation as text at the bottom
    best_eq_str = str(model.sympy())
    fig.text(0.5, 0.01, f"Best equation: y = {best_eq_str}",
             ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             wrap=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [VIZ] Saved plot to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_pareto_scores(model, label, save_path=None):
    """Plot the score (improvement rate) for each equation on the Pareto front."""
    eqs = model.equations_
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Score vs complexity
    ax1.bar(eqs["complexity"], eqs["score"], color='teal', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Complexity", fontsize=11)
    ax1.set_ylabel("Score (improvement rate)", fontsize=11)
    ax1.set_title(f"Score vs Complexity — {label}", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Loss vs complexity with annotations
    ax2.semilogy(eqs["complexity"], eqs["loss"], 's-', color='darkred', markersize=7)
    for _, row in eqs.iterrows():
        if row["score"] > 0.05:  # Annotate high-score equations
            ax2.annotate(f"s={row['score']:.3f}",
                         (row["complexity"], row["loss"]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax2.set_xlabel("Complexity", fontsize=11)
    ax2.set_ylabel("Loss (log)", fontsize=11)
    ax2.set_title(f"Loss vs Complexity — {label}", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [VIZ] Saved score plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_live_update(X, y_true, model, feature_names, label, fig, axes):
    """Update a live plot with the current best prediction. Call during/after fitting."""
    y_pred = model.predict(X)

    for ax in axes:
        ax.clear()

    # Real vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=15, c='steelblue', edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, 'r--', linewidth=1)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    axes[0].set_title(f"{label} | R²={r2:.4f}", fontsize=10)
    axes[0].set_xlabel("Real")
    axes[0].set_ylabel("Predicted")

    # Pareto front
    eqs = model.equations_
    axes[1].semilogy(eqs["complexity"], eqs["loss"], 'o-', color='darkgreen', markersize=5)
    axes[1].set_title("Pareto Front", fontsize=10)
    axes[1].set_xlabel("Complexity")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Live: {label} — Best: y = {str(model.sympy())[:80]}...", fontsize=9)
    plt.tight_layout()
    plt.pause(0.1)


# =============================================================================
# Core SR function
# =============================================================================

def run_symbolic_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    label: str,
    args: argparse.Namespace,
):
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
        batch_size=min(64, len(X)),
    )

    model.fit(X, y, variable_names=feature_names)
    return model


def parse_layer_spec(spec: str, max_layer: int = 63) -> list[int]:
    spec = spec.strip().lower()
    if spec in LAYER_GROUPS:
        return [l for l in LAYER_GROUPS[spec] if l <= max_layer]
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.update(range(int(lo), min(int(hi) + 1, max_layer + 1)))
        else:
            layers.add(int(part))
    return sorted(layers)


def main():
    parser = argparse.ArgumentParser(
        description="Symbolic regression on LLM layer activations (with visualization)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with visualization
  uv run symbolic_regression.py --visualize

  # Save plots to directory
  uv run symbolic_regression.py --visualize --plot-dir ./plots

  # Live plotting mode
  uv run symbolic_regression.py --live-plot

  # Specific experiment, specific layers
  uv run symbolic_regression.py --experiment capital_varied_english --layers early --visualize
""",
    )

    parser.add_argument("--experiment", "-e", type=str, default=None)
    parser.add_argument("--results-dir", "-r", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--layers", "-l", type=str, default="all")
    parser.add_argument("--mode", "-m", choices=["per-layer", "cross-layer"], default="per-layer")
    parser.add_argument("--pca", "-p", type=int, default=10)
    parser.add_argument("--iterations", "-i", type=int, default=30)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--parsimony", type=float, default=0.0032)
    parser.add_argument("--maxsize", type=int, default=25)
    parser.add_argument("--binary-ops", type=str, default="+,*,/,^,-")
    parser.add_argument("--unary-ops", type=str, default="sin,cos,exp,log,sqrt,tanh,abs")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--max-dims", type=int, default=None)
    parser.add_argument("--list-experiments", action="store_true")

    # Visualization arguments
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Enable visualization of real vs predicted data, residuals, and Pareto front",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: show interactively)",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Enable live updating plot during fitting (requires interactive backend)",
    )
    parser.add_argument(
        "--full-equations",
        action="store_true",
        help="Print the full Pareto front of equations without truncation",
    )

    args = parser.parse_args()

    # If --visualize is set, also enable full equations by default
    if args.visualize:
        args.full_equations = True

    results_dir = Path(args.results_dir)

    if args.list_experiments:
        for exp in discover_experiments(results_dir):
            exp_dir = results_dir / exp
            layers = discover_layers(exp_dir)
            info_path = exp_dir / "model_info.json"
            model_name = ""
            if info_path.exists():
                info = json.loads(info_path.read_text())
                model_name = info.get("model_name", "")
            print(f"  {exp}  (layers: {len(layers)}, model: {model_name})")
        return

    if args.experiment:
        exp_name = args.experiment
    else:
        experiments = discover_experiments(results_dir)
        if not experiments:
            print(f"No experiments found in {results_dir}", file=sys.stderr)
            sys.exit(1)
        exp_name = experiments[0]

    exp_dir = results_dir / exp_name
    if not exp_dir.exists():
        print(f"Experiment not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    available_layers = discover_layers(exp_dir)
    max_layer = max(available_layers)
    layers = parse_layer_spec(args.layers, max_layer)

    model_info = {}
    info_path = exp_dir / "model_info.json"
    if info_path.exists():
        model_info = json.loads(info_path.read_text())

    print(f"Experiment:  {exp_name}")
    print(f"Model:       {model_info.get('model_name', 'unknown')}")
    print(f"Layers:      {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Mode:        {args.mode}")
    print(f"PCA:         {args.pca} components" if args.pca > 0 else "PCA:         disabled")
    print(f"Iterations:  {args.iterations}")
    print(f"Operators:   binary={args.binary_ops}  unary={args.unary_ops}")
    print(f"Visualize:   {args.visualize}")
    print(f"Live plot:   {args.live_plot}")
    print()

    # Setup plot directory
    plot_dir = None
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Setup live plot
    if args.live_plot:
        plt.ion()
        live_fig, live_axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.show(block=False)
    else:
        live_fig, live_axes = None, None

    feature_names = ["prompt_idx", "token_pos"]
    if args.mode == "cross-layer":
        feature_names = ["layer_idx", "prompt_idx", "token_pos"]

    all_results = []

    if args.mode == "per-layer":
        for layer_idx in layers:
            df = load_layer_data(exp_dir, layer_idx)
            X_raw, y_raw, token_texts = build_feature_matrix(df)
            print(f"--- Layer {layer_idx:3d}  (samples={len(y_raw)}, dims={y_raw.shape[1]}) ---")

            if args.pca > 0:
                pca, y_reduced, scaler = apply_pca(y_raw, args.pca)
                var_explained = pca.explained_variance_ratio_
                print(f"  PCA explained variance: {var_explained.sum():.3f} total, "
                      f"top-3: {var_explained[:3].round(3).tolist()}")
            else:
                y_reduced = y_raw
                var_explained = None

            n_dims = y_reduced.shape[1]
            if args.max_dims:
                n_dims = min(n_dims, args.max_dims)

            layer_results = {"layer": layer_idx, "equations": []}

            for dim_idx in range(n_dims):
                label = f"pc{dim_idx}" if args.pca > 0 else f"dim{dim_idx}"
                full_label = f"L{layer_idx}_{label}"
                y_target = y_reduced[:, dim_idx]

                if np.std(y_target) < 1e-10:
                    continue

                model = run_symbolic_regression(
                    X_raw, y_target, feature_names, full_label, args
                )

                # Print full equations table
                if args.full_equations:
                    print_full_equations_table(model)

                # Live plot update
                if args.live_plot and live_fig is not None:
                    plot_live_update(X_raw, y_target, model, feature_names, full_label, live_fig, live_axes)

                # Full visualization
                if args.visualize:
                    save_path = None
                    if plot_dir:
                        save_path = plot_dir / f"{exp_name}_layer{layer_idx:03d}_{label}_real_vs_pred.png"
                    plot_real_vs_predicted(X_raw, y_target, model, feature_names, full_label, save_path)

                    save_path_scores = None
                    if plot_dir:
                        save_path_scores = plot_dir / f"{exp_name}_layer{layer_idx:03d}_{label}_scores.png"
                    plot_pareto_scores(model, full_label, save_path_scores)

                best_eq = model.sympy()
                best_row = model.equations_
                best_row = best_row.iloc[best_row["loss"].idxmin()]

                result_entry = {
                    "label": label,
                    "equation": str(best_eq),
                    "loss": float(best_row["loss"]),
                    "score": float(best_row["score"]) if "score" in best_row.index else 0.0,
                    "complexity": int(best_row["complexity"]) if "complexity" in best_row.index else 0,
                }
                layer_results["equations"].append(result_entry)

                print(f"  {label}: y = {best_eq}")
                print(f"         loss={result_entry['loss']:.4e}  complexity={result_entry['complexity']}")

            all_results.append(layer_results)
            print()

    elif args.mode == "cross-layer":
        X_raw, y_raw, labels = build_cross_layer_matrix(exp_dir, layers)
        print(f"Cross-layer matrix: {X_raw.shape[0]} samples × {y_raw.shape[1]} dims")

        if args.pca > 0:
            pca, y_reduced, scaler = apply_pca(y_raw, args.pca)
            var_explained = pca.explained_variance_ratio_
            print(f"PCA explained variance: {var_explained.sum():.3f} total")
            for i, v in enumerate(var_explained[:min(10, len(var_explained))]):
                print(f"  PC{i}: {v:.4f} (cumulative: {var_explained[:i+1].sum():.4f})")
        else:
            y_reduced = y_raw

        n_dims = y_reduced.shape[1]
        if args.max_dims:
            n_dims = min(n_dims, args.max_dims)

        cross_results = {"mode": "cross-layer", "layers": layers, "equations": []}

        for dim_idx in range(n_dims):
            label = f"pc{dim_idx}" if args.pca > 0 else f"dim{dim_idx}"
            full_label = f"cross_{label}"
            y_target = y_reduced[:, dim_idx]

            if np.std(y_target) < 1e-10:
                continue

            model = run_symbolic_regression(
                X_raw, y_target, feature_names, full_label, args
            )

            # Print full equations table
            if args.full_equations:
                print_full_equations_table(model)

            # Live plot update
            if args.live_plot and live_fig is not None:
                plot_live_update(X_raw, y_target, model, feature_names, full_label, live_fig, live_axes)

            # Full visualization
            if args.visualize:
                save_path = None
                if plot_dir:
                    save_path = plot_dir / f"{exp_name}_cross_{label}_real_vs_pred.png"
                plot_real_vs_predicted(X_raw, y_target, model, feature_names, full_label, save_path)

                save_path_scores = None
                if plot_dir:
                    save_path_scores = plot_dir / f"{exp_name}_cross_{label}_scores.png"
                plot_pareto_scores(model, full_label, save_path_scores)

            best_eq = model.sympy()
            best_row = model.equations_
            best_row = best_row.iloc[best_row["loss"].idxmin()]

            result_entry = {
                "label": label,
                "equation": str(best_eq),
                "loss": float(best_row["loss"]),
                "score": float(best_row["score"]) if "score" in best_row.index else 0.0,
                "complexity": int(best_row["complexity"]) if "complexity" in best_row.index else 0,
            }
            cross_results["equations"].append(result_entry)

            print(f"  {label}: y = {best_eq}")
            print(f"         loss={result_entry['loss']:.4e}  complexity={result_entry['complexity']}")

        all_results.append(cross_results)
        print()

    # Close live plot
    if args.live_plot:
        plt.ioff()
        plt.show()

    # ==========================================================================
    # Summary visualization: compare all fitted dimensions
    # ==========================================================================
    if args.visualize and all_results:
        # Summary plot: best loss per layer/dimension
        fig_summary, ax_summary = plt.subplots(figsize=(14, 6))

        if args.mode == "per-layer":
            for layer_res in all_results:
                layer_idx = layer_res["layer"]
                for eq_info in layer_res["equations"]:
                    ax_summary.scatter(
                        layer_idx, eq_info["loss"],
                        c='steelblue', s=30, alpha=0.7, edgecolors='none'
                    )
            ax_summary.set_xlabel("Layer Index", fontsize=12)
            ax_summary.set_ylabel("Best Loss (per dimension)", fontsize=12)
            ax_summary.set_title(f"Best SR Loss per Layer — {exp_name}", fontsize=13, fontweight='bold')
            ax_summary.set_yscale('log')
            ax_summary.grid(True, alpha=0.3)
        elif args.mode == "cross-layer":
            for cross_res in all_results:
                labels_plot = [eq["label"] for eq in cross_res["equations"]]
                losses_plot = [eq["loss"] for eq in cross_res["equations"]]
                ax_summary.bar(labels_plot, losses_plot, color='teal', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax_summary.set_xlabel("PCA Component", fontsize=12)
            ax_summary.set_ylabel("Best Loss", fontsize=12)
            ax_summary.set_title(f"Best SR Loss per Component (Cross-Layer) — {exp_name}", fontsize=13, fontweight='bold')
            ax_summary.set_yscale('log')
            ax_summary.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if plot_dir:
            summary_path = plot_dir / f"{exp_name}_summary_losses.png"
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            print(f"  [VIZ] Saved summary plot to {summary_path}")
        else:
            plt.show()
        plt.close(fig_summary)

    # ==========================================================================
    # Save results JSON
    # ==========================================================================
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "experiment": exp_name,
                    "model": model_info,
                    "args": vars(args),
                    "results": all_results,
                },
                f,
                indent=2,
            )
        print(f"Results saved to {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
