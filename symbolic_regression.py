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
Uses RAW activation dimensions as features (dynamically selected by correlation).
Shows matplotlib visualizations of signals, real vs predicted, Pareto fronts.

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
import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend
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


# =============================================================================
# Data Loading
# =============================================================================

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


# =============================================================================
# Feature Building — uses RAW dims, dynamically selected
# =============================================================================

def get_dim_columns(df: pd.DataFrame) -> list[str]:
    """Get all dimension column names from the dataframe."""
    meta_cols = {"prompt_idx", "token_pos", "token_text"}
    return [c for c in df.columns if c not in meta_cols]


def select_features_for_target(
    df: pd.DataFrame,
    target_col: str,
    n_features: int = 8,
    always_include: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Dynamically select the top-correlated raw dimensions as input features
    for predicting target_col. Always includes prompt_idx and token_pos.
    
    Returns:
        X: (n_samples, n_features) array
        y: (n_samples,) target array
        feature_names: list of selected feature names
    """
    if always_include is None:
        always_include = ["prompt_idx", "token_pos"]
    
    dim_cols = get_dim_columns(df)
    y = df[target_col].values.astype(float)
    
    # Compute correlation of every other dim with the target
    correlations = {}
    for col in dim_cols:
        if col == target_col:
            continue
        col_data = df[col].values.astype(float)
        # Handle constant columns
        if np.std(col_data) < 1e-12:
            correlations[col] = 0.0
            continue
        corr = np.corrcoef(y, col_data)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        correlations[col] = abs(corr)
    
    # Sort by absolute correlation, pick top N
    n_dim_features = max(0, n_features - len(always_include))
    sorted_dims = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_dims = [name for name, _ in sorted_dims[:n_dim_features]]
    
    # Build X matrix
    feature_names = always_include + top_dims
    X = df[feature_names].values.astype(float)
    
    return X, y, feature_names


def build_feature_matrix_raw(
    df: pd.DataFrame,
    target_idx: int,
    n_features: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    """
    Build features for predicting a specific raw dimension by index.
    Uses prompt_idx, token_pos, and top-correlated other raw dims.
    
    Returns: X, y, feature_names, target_name
    """
    dim_cols = get_dim_columns(df)
    target_col = dim_cols[target_idx]
    X, y, feature_names = select_features_for_target(
        df, target_col, n_features=n_features
    )
    return X, y, feature_names, target_col


def build_feature_matrix_pca_target(
    df: pd.DataFrame,
    pca_component_idx: int,
    n_pca_components: int = 10,
    n_features: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[str], str, PCA, StandardScaler]:
    """
    Target is a PCA component of all dims. Features are prompt_idx, token_pos,
    and top-correlated RAW dims (not PCA of features!).
    
    Returns: X, y, feature_names, target_name, pca, scaler
    """
    dim_cols = get_dim_columns(df)
    all_dims = df[dim_cols].values.astype(float)
    
    # PCA on targets
    scaler = StandardScaler()
    dims_scaled = scaler.fit_transform(all_dims)
    pca = PCA(n_components=min(n_pca_components, all_dims.shape[1], all_dims.shape[0] - 1))
    pca_values = pca.fit_transform(dims_scaled)
    
    y = pca_values[:, pca_component_idx]
    target_name = f"pc{pca_component_idx}"
    
    # Find which raw dims correlate most with this PC target
    correlations = {}
    for i, col in enumerate(dim_cols):
        col_data = all_dims[:, i]
        if np.std(col_data) < 1e-12:
            correlations[col] = 0.0
            continue
        corr = np.corrcoef(y, col_data)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        correlations[col] = abs(corr)
    
    # Top correlated raw dims as features
    n_dim_features = max(0, n_features - 2)  # reserve 2 for prompt_idx, token_pos
    sorted_dims = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_dims = [name for name, _ in sorted_dims[:n_dim_features]]
    
    feature_names = ["prompt_idx", "token_pos"] + top_dims
    X = df[feature_names].values.astype(float)
    
    return X, y, feature_names, target_name, pca, scaler


# =============================================================================
# Visualization Functions
# =============================================================================

def print_full_equations_table(model):
    """Print the full Pareto front of equations without truncation."""
    eqs = model.equations_
    print("\n" + "=" * 140)
    print("FULL PARETO FRONT OF EQUATIONS")
    print("=" * 140)
    print(f"{'Complexity':<12} {'Loss':<14} {'Score':<12} Equation")
    print("-" * 140)
    for _, row in eqs.iterrows():
        complexity = int(row["complexity"])
        loss = row["loss"]
        score = row.get("score", 0.0)
        equation = row["equation"]
        print(f"{complexity:<12} {loss:<14.4e} {score:<12.4e} y = {equation}")
    print("=" * 140 + "\n")


def plot_signal_overview(df: pd.DataFrame, layer_idx: int, n_dims_show: int = 8):
    """
    Show raw activation signals: top-variance dims plotted over token_pos,
    colored by prompt_idx. This is the 'look at the actual data' plot.
    """
    dim_cols = get_dim_columns(df)
    
    # Pick dims with highest variance
    variances = df[dim_cols].var().sort_values(ascending=False)
    top_dims = variances.index[:n_dims_show].tolist()
    
    n_prompts = df["prompt_idx"].nunique()
    prompt_ids = sorted(df["prompt_idx"].unique())
    
    fig, axes = plt.subplots(n_dims_show, 1, figsize=(14, 2.5 * n_dims_show), sharex=True)
    if n_dims_show == 1:
        axes = [axes]
    
    cmap = plt.cm.tab10
    
    for ax_idx, dim_name in enumerate(top_dims):
        ax = axes[ax_idx]
        for i, pid in enumerate(prompt_ids):
            subset = df[df["prompt_idx"] == pid].sort_values("token_pos")
            color = cmap(i % 10)
            ax.plot(subset["token_pos"], subset[dim_name],
                    color=color, alpha=0.7, linewidth=1.2, label=f"p{int(pid)}")
        ax.set_ylabel(dim_name, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{dim_name} (var={variances[dim_name]:.4f})", fontsize=9, loc='left')
    
    axes[-1].set_xlabel("token_pos", fontsize=11)
    if n_prompts <= 12:
        axes[0].legend(fontsize=7, ncol=min(6, n_prompts), loc='upper right')
    
    fig.suptitle(f"Layer {layer_idx} — Top {n_dims_show} Highest-Variance Raw Dimensions",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


def plot_signal_heatmap(df: pd.DataFrame, layer_idx: int):
    """Heatmap of all raw activations: rows=samples, cols=dims."""
    dim_cols = get_dim_columns(df)
    data = df[dim_cols].values
    
    fig, ax = plt.subplots(figsize=(16, 8))
    vmin, vmax = np.percentile(data, 2), np.percentile(data, 98)
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    ax.set_xlabel(f"Dimension (0..{data.shape[1]-1})", fontsize=11)
    ax.set_ylabel("Sample (prompt × token_pos)", fontsize=11)
    ax.set_title(f"Layer {layer_idx} — Raw Activation Heatmap "
                 f"({data.shape[0]} samples × {data.shape[1]} dims)",
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Activation value")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


def plot_real_vs_predicted(X, y_true, model, feature_names, label):
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
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ax1.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Panel 2: Signal overlay (real vs predicted by sample index) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(y_true, label='Real', color='steelblue', linewidth=0.8, alpha=0.9)
    ax2.plot(y_pred, label='Predicted', color='orangered', linewidth=0.8, alpha=0.8)
    ax2.set_xlabel("Sample index", fontsize=11)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title("Signal: Real vs Predicted", fontsize=12, fontweight='bold')
    ax2.legend()

    # --- Panel 3: Residuals ---
    ax3 = fig.add_subplot(gs[0, 2])
    residuals = y_true - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.5, s=20, c='coral', edgecolors='none')
    ax3.axhline(0, color='black', linewidth=1, linestyle='--')
    ax3.set_xlabel("Predicted", fontsize=11)
    ax3.set_ylabel("Residual", fontsize=11)
    ax3.set_title(f"Residuals — MAE={np.mean(np.abs(residuals)):.4f}", fontsize=12, fontweight='bold')

    # --- Panel 4: Real vs first feature colored by second ---
    ax4 = fig.add_subplot(gs[1, 0])
    feat0 = X[:, 0]
    feat1 = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
    sc = ax4.scatter(feat0, y_true, c=feat1, cmap='viridis', alpha=0.6, s=20, edgecolors='none')
    ax4.set_xlabel(feature_names[0], fontsize=11)
    ax4.set_ylabel("Real Value", fontsize=11)
    ax4.set_title(f"Real vs {feature_names[0]}", fontsize=12, fontweight='bold')
    plt.colorbar(sc, ax=ax4, label=feature_names[1] if len(feature_names) > 1 else "")

    # --- Panel 5: Real vs top correlated dim feature ---
    ax5 = fig.add_subplot(gs[1, 1])
    if X.shape[1] > 2:
        # Third feature is the top-correlated raw dim
        feat2 = X[:, 2]
        ax5.scatter(feat2, y_true, alpha=0.5, s=20, c='steelblue', edgecolors='none', label='Real')
        ax5.scatter(feat2, y_pred, alpha=0.3, s=15, c='orangered', edgecolors='none', label='Predicted')
        ax5.set_xlabel(feature_names[2], fontsize=11)
        ax5.set_ylabel("Value", fontsize=11)
        ax5.set_title(f"vs Top Correlated Dim: {feature_names[2]}", fontsize=12, fontweight='bold')
        ax5.legend()
    else:
        sort_idx = np.argsort(y_true)
        ax5.plot(y_true[sort_idx], label='Real (sorted)', color='steelblue', linewidth=1.2)
        ax5.plot(y_pred[sort_idx], label='Predicted', color='orangered', linewidth=1.2, alpha=0.8)
        ax5.set_xlabel("Sample (sorted)", fontsize=11)
        ax5.set_ylabel("Value", fontsize=11)
        ax5.set_title("Sorted Overlay", fontsize=12, fontweight='bold')
        ax5.legend()

    # --- Panel 6: Pareto front ---
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

    # Best equation as text
    best_eq_str = str(model.sympy())
    fig.text(0.5, 0.01, f"Best equation: y = {best_eq_str}",
             ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             wrap=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show(block=False)
    plt.pause(0.5)


def plot_pareto_scores(model, label):
    """Plot score and loss vs complexity."""
    eqs = model.equations_
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(eqs["complexity"], eqs["score"], color='teal', alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Complexity", fontsize=11)
    ax1.set_ylabel("Score (improvement rate)", fontsize=11)
    ax1.set_title(f"Score vs Complexity — {label}", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.semilogy(eqs["complexity"], eqs["loss"], 's-', color='darkred', markersize=7)
    for _, row in eqs.iterrows():
        if row["score"] > 0.05:
            ax2.annotate(f"s={row['score']:.3f}",
                         (row["complexity"], row["loss"]),
                         textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax2.set_xlabel("Complexity", fontsize=11)
    ax2.set_ylabel("Loss (log)", fontsize=11)
    ax2.set_title(f"Loss vs Complexity — {label}", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


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

    print(f"    [SR] Fitting {label} with features: {feature_names}")
    print(f"    [SR] X shape: {X.shape}, y shape: {y.shape}")
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


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Symbolic regression on LLM layer activations — uses raw dims as features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: layer 20, first experiment, shows plots
  uv run symbolic_regression.py

  # Specific layer, specific experiment
  uv run symbolic_regression.py -e capital_paris_multilingual -l 32

  # Multiple layers
  uv run symbolic_regression.py -l 10-15

  # Use raw dims as targets (no PCA)
  uv run symbolic_regression.py --no-pca --max-dims 3

  # More features from raw dims
  uv run symbolic_regression.py --n-features 12

  # Save plots instead of showing
  uv run symbolic_regression.py --plot-dir ./plots
""",
    )

    parser.add_argument("--experiment", "-e", type=str, default=None,
                        help="Experiment name (default: first found)")
    parser.add_argument("--results-dir", "-r", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--layers", "-l", type=str, default="20",
                        help="Layer spec: number, range (10-15), or group (early/mid/late/all). Default: 20")
    parser.add_argument("--mode", "-m", choices=["per-layer", "cross-layer"], default="per-layer")

    # Target configuration
    parser.add_argument("--no-pca", action="store_true",
                        help="Use raw dims as targets instead of PCA components")
    parser.add_argument("--pca", "-p", type=int, default=5,
                        help="Number of PCA components for targets (default: 5)")
    parser.add_argument("--max-dims", type=int, default=3,
                        help="Max number of target dims to fit (default: 3)")

    # Feature configuration
    parser.add_argument("--n-features", type=int, default=8,
                        help="Total number of input features for SR (includes prompt_idx, token_pos, "
                             "plus top-correlated raw dims). Default: 8")

    # SR parameters
    parser.add_argument("--iterations", "-i", type=int, default=40)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--parsimony", type=float, default=0.0032)
    parser.add_argument("--maxsize", type=int, default=30)
    parser.add_argument("--binary-ops", type=str, default="+,*,/,^,-")
    parser.add_argument("--unary-ops", type=str, default="sin,cos,exp,log,sqrt,tanh,abs")

    # Output
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Save plots to this directory instead of showing interactively")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable all matplotlib plots")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.list_experiments:
        print("Available experiments:")
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

    # Resolve experiment
    if args.experiment:
        exp_name = args.experiment
    else:
        experiments = discover_experiments(results_dir)
        if not experiments:
            print(f"ERROR: No experiments found in {results_dir}", file=sys.stderr)
            sys.exit(1)
        exp_name = experiments[0]

    exp_dir = results_dir / exp_name
    if not exp_dir.exists():
        print(f"ERROR: Experiment not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    available_layers = discover_layers(exp_dir)
    max_layer = max(available_layers)
    layers = parse_layer_spec(args.layers, max_layer)
    # Filter to only available layers
    layers = [l for l in layers if l in available_layers]
    if not layers:
        print(f"ERROR: No matching layers found. Available: {available_layers}", file=sys.stderr)
        sys.exit(1)

    model_info = {}
    info_path = exp_dir / "model_info.json"
    if info_path.exists():
        model_info = json.loads(info_path.read_text())

    # Print configuration clearly
    print("=" * 80)
    print("SYMBOLIC REGRESSION ON LLM ACTIVATIONS")
    print("=" * 80)
    print(f"  Experiment:    {exp_name}")
    print(f"  Model:         {model_info.get('model_name', 'unknown')}")
    print(f"  Layer(s):      {layers}")
    print(f"  Mode:          {args.mode}")
    print(f"  Target:        {'Raw dims' if args.no_pca else f'PCA (top {args.pca} components)'}")
    print(f"  Max targets:   {args.max_dims}")
    print(f"  N features:    {args.n_features} (prompt_idx + token_pos + {args.n_features - 2} top-correlated raw dims)")
    print(f"  SR iterations: {args.iterations}")
    print(f"  SR maxsize:    {args.maxsize}")
    print(f"  Operators:     binary={args.binary_ops}  unary={args.unary_ops}")
    print(f"  Plots:         {'disabled' if args.no_plots else 'enabled (interactive)'}")
    print("=" * 80)
    print()

    # Setup plot directory
    plot_dir = None
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for layer_idx in layers:
        df = load_layer_data(exp_dir, layer_idx)
        dim_cols = get_dim_columns(df)
        n_total_dims = len(dim_cols)

        print(f"\n{'#' * 80}")
        print(f"### LAYER {layer_idx}  —  {len(df)} samples × {n_total_dims} raw dimensions")
        print(f"{'#' * 80}")

        # =====================================================================
        # Show raw signal plots FIRST so you see the data
        # =====================================================================
        # Show raw signal plots FIRST so you see the data
        # =====================================================================
        if not args.no_plots:
            plot_signal_overview(df, layer_idx, n_dims_show=8)
            plot_signal_heatmap(df, layer_idx)

        # =====================================================================
        # Build targets (what we predict)
        # =====================================================================
        dim_cols = get_dim_columns(df)

        if args.no_pca:
            # Use raw dims directly as targets
            # Pick highest-variance dims as targets
            variances = df[dim_cols].var().sort_values(ascending=False)
            target_dims = variances.index[:args.max_dims].tolist()
            print(f"  Targets (raw, top-variance): {target_dims}")
        else:
            # PCA on all dims to get target components
            all_dims = df[dim_cols].values.astype(float)
            pca, y_reduced, scaler = apply_pca(all_dims, args.pca)
            var_explained = pca.explained_variance_ratio_
            print(f"  Target PCA explained variance: {var_explained.sum():.3f} total, "
                  f"top-3: {var_explained[:3].round(3).tolist()}")
            target_dims = None  # will use PCA components

        n_targets = args.max_dims
        layer_results = {"layer": layer_idx, "equations": []}

        for dim_idx in range(n_targets):
            # ==================================================================
            # Build X (features) and y (target) dynamically
            # ==================================================================
            if args.no_pca:
                # Target = raw dim, features = prompt_idx + token_pos + top-correlated other raw dims
                target_col = target_dims[dim_idx]
                target_col_idx = dim_cols.index(target_col)
                X, y_target, feat_names, target_name = build_feature_matrix_raw(
                    df, target_col_idx, n_features=args.n_features
                )
                label = target_name
            else:
                # Target = PCA component, features = prompt_idx + token_pos + top-correlated raw dims
                X, y_target, feat_names, target_name, _, _ = build_feature_matrix_pca_target(
                    df, dim_idx, n_pca_components=args.pca, n_features=args.n_features
                )
                label = target_name

            full_label = f"L{layer_idx}_{label}"

            if np.std(y_target) < 1e-10:
                print(f"  {label}: SKIPPED (zero variance)")
                continue

            print(f"\n  --- Fitting {full_label} ---")
            print(f"      Features ({len(feat_names)}): {feat_names}")
            print(f"      y std: {np.std(y_target):.4f}, range: [{y_target.min():.3f}, {y_target.max():.3f}]")

            # ==================================================================
            # Run symbolic regression
            # ==================================================================
            model = run_symbolic_regression(X, y_target, feat_names, full_label, args)

            # Always print full equations
            print_full_equations_table(model)

            # Always show visualization (unless --no-plots)
            if not args.no_plots:
                plot_real_vs_predicted(X, y_target, model, feat_names, full_label)
                plot_pareto_scores(model, full_label)

            # Store result
            best_eq = model.sympy()
            best_row = model.equations_
            best_row = best_row.iloc[best_row["loss"].idxmin()]

            result_entry = {
                "label": label,
                "equation": str(best_eq),
                "loss": float(best_row["loss"]),
                "score": float(best_row["score"]) if "score" in best_row.index else 0.0,
                "complexity": int(best_row["complexity"]) if "complexity" in best_row.index else 0,
                "features_used": feat_names,
            }
            layer_results["equations"].append(result_entry)

            print(f"  ✓ {label}: y = {best_eq}")
            print(f"    loss={result_entry['loss']:.4e}  complexity={result_entry['complexity']}")
            print(f"    features: {feat_names}")

        all_results.append(layer_results)

    # ==========================================================================
    # Summary visualization
    # ==========================================================================
    if not args.no_plots and all_results:
        fig_summary, ax_summary = plt.subplots(figsize=(14, 6))
        for layer_res in all_results:
            lid = layer_res["layer"]
            for eq_info in layer_res["equations"]:
                ax_summary.scatter(lid, eq_info["loss"],
                                   c='steelblue', s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax_summary.set_xlabel("Layer Index", fontsize=12)
        ax_summary.set_ylabel("Best Loss", fontsize=12)
        ax_summary.set_title(f"SR Loss per Layer — {exp_name}", fontsize=13, fontweight='bold')
        ax_summary.set_yscale('log')
        ax_summary.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

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

    print("\nDone. All plots shown.")
    # Keep plots open until user closes them
    if not args.no_plots:
        plt.show(block=True)


if __name__ == "__main__":
    main()

