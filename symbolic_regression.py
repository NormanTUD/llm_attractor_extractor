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
Symbolic Regression: Approximate real LLM layers as input→output transformations.

For each layer L, the INPUT is the activation from layer L-1 and the OUTPUT is
the activation from layer L. We find symbolic equations f such that:

    output_pc_k ≈ f(input_pc_0, input_pc_1, ..., input_pc_N)

This approximates what the layer actually *does* as a closed-form expression.

Strategy:
  1. Load layer L-1 (input) and layer L (output) for matched samples.
  2. PCA-compress both sides to manageable dimensionality.
  3. For each output principal component, run symbolic regression using
     input principal components as features.
  4. Visualize everything in consolidated multi-panel figures.

Bootstraps itself via `uv run`.
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
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent / "results_single_word_larger_deepseek"

LAYER_GROUPS = {
    "early": list(range(1, 22)),
    "mid": list(range(22, 43)),
    "late": list(range(43, 64)),
    "first-half": list(range(1, 32)),
    "second-half": list(range(32, 64)),
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
    """Get all dimension column names from the dataframe."""
    meta_cols = {"prompt_idx", "token_pos", "token_text"}
    return [c for c in df.columns if c not in meta_cols]


# =============================================================================
# Core: Build input→output pairs for a layer transition
# =============================================================================

def build_layer_io(
    experiment_dir: Path,
    layer_idx: int,
    n_input_pcs: int = 10,
    n_output_pcs: int = 5,
) -> dict:
    """
    Load layer (layer_idx - 1) as INPUT and layer_idx as OUTPUT.
    PCA-compress both. Return everything needed for SR.

    Returns dict with:
        X_pca: (n_samples, n_input_pcs) — input PCs
        Y_pca: (n_samples, n_output_pcs) — output PCs
        input_pca, output_pca: PCA models
        input_scaler, output_scaler: StandardScaler models
        input_var_explained, output_var_explained: variance ratios
        meta: DataFrame with prompt_idx, token_pos, token_text
        input_feature_names: list of str
    """
    input_layer = layer_idx - 1
    output_layer = layer_idx

    df_in = load_layer_data(experiment_dir, input_layer)
    df_out = load_layer_data(experiment_dir, output_layer)

    # Align samples by prompt_idx + token_pos
    merge_keys = ["prompt_idx", "token_pos"]
    # Ensure same ordering
    df_in = df_in.sort_values(merge_keys).reset_index(drop=True)
    df_out = df_out.sort_values(merge_keys).reset_index(drop=True)

    # Verify alignment
    assert len(df_in) == len(df_out), (
        f"Layer {input_layer} has {len(df_in)} samples but layer {output_layer} has {len(df_out)}"
    )
    assert (df_in["prompt_idx"].values == df_out["prompt_idx"].values).all(), "prompt_idx mismatch"
    assert (df_in["token_pos"].values == df_out["token_pos"].values).all(), "token_pos mismatch"

    dim_cols_in = get_dim_columns(df_in)
    dim_cols_out = get_dim_columns(df_out)

    X_raw = df_in[dim_cols_in].values.astype(np.float32)
    Y_raw = df_out[dim_cols_out].values.astype(np.float32)

    # PCA compress inputs
    input_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(X_raw)
    n_in_pcs = min(n_input_pcs, X_raw.shape[1], X_raw.shape[0] - 1)
    input_pca = PCA(n_components=n_in_pcs)
    X_pca = input_pca.fit_transform(X_scaled)

    # PCA compress outputs
    output_scaler = StandardScaler()
    Y_scaled = output_scaler.fit_transform(Y_raw)
    n_out_pcs = min(n_output_pcs, Y_raw.shape[1], Y_raw.shape[0] - 1)
    output_pca = PCA(n_components=n_out_pcs)
    Y_pca = output_pca.fit_transform(Y_scaled)

    input_feature_names = [f"in_pc{i}" for i in range(n_in_pcs)]

    meta = df_in[["prompt_idx", "token_pos"]].copy()
    if "token_text" in df_in.columns:
        meta["token_text"] = df_in["token_text"]

    return {
        "X_pca": X_pca,
        "Y_pca": Y_pca,
        "X_raw": X_raw,
        "Y_raw": Y_raw,
        "input_pca": input_pca,
        "output_pca": output_pca,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "input_var_explained": input_pca.explained_variance_ratio_,
        "output_var_explained": output_pca.explained_variance_ratio_,
        "meta": meta,
        "input_feature_names": input_feature_names,
        "input_layer": input_layer,
        "output_layer": output_layer,
        "n_samples": len(df_in),
        "n_raw_dims": len(dim_cols_in),
    }


def build_residual_io(
    experiment_dir: Path,
    layer_idx: int,
    n_input_pcs: int = 10,
    n_output_pcs: int = 5,
) -> dict:
    """
    Like build_layer_io, but targets the RESIDUAL (output - input) instead.
    This models what the layer *adds* rather than the full output.
    Useful because transformer layers are residual: output = input + layer(input).
    """
    input_layer = layer_idx - 1
    output_layer = layer_idx

    df_in = load_layer_data(experiment_dir, input_layer)
    df_out = load_layer_data(experiment_dir, output_layer)

    merge_keys = ["prompt_idx", "token_pos"]
    df_in = df_in.sort_values(merge_keys).reset_index(drop=True)
    df_out = df_out.sort_values(merge_keys).reset_index(drop=True)

    assert len(df_in) == len(df_out)
    assert (df_in["prompt_idx"].values == df_out["prompt_idx"].values).all()
    assert (df_in["token_pos"].values == df_out["token_pos"].values).all()

    dim_cols_in = get_dim_columns(df_in)
    dim_cols_out = get_dim_columns(df_out)

    X_raw = df_in[dim_cols_in].values.astype(np.float32)
    Y_raw = df_out[dim_cols_out].values.astype(np.float32)

    # Residual = what the layer adds
    R_raw = Y_raw - X_raw

    # PCA compress inputs
    input_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(X_raw)
    n_in_pcs = min(n_input_pcs, X_raw.shape[1], X_raw.shape[0] - 1)
    input_pca = PCA(n_components=n_in_pcs)
    X_pca = input_pca.fit_transform(X_scaled)

    # PCA compress residual
    output_scaler = StandardScaler()
    R_scaled = output_scaler.fit_transform(R_raw)
    n_out_pcs = min(n_output_pcs, R_raw.shape[1], R_raw.shape[0] - 1)
    output_pca = PCA(n_components=n_out_pcs)
    Y_pca = output_pca.fit_transform(R_scaled)

    input_feature_names = [f"in_pc{i}" for i in range(n_in_pcs)]

    meta = df_in[["prompt_idx", "token_pos"]].copy()
    if "token_text" in df_in.columns:
        meta["token_text"] = df_in["token_text"]

    return {
        "X_pca": X_pca,
        "Y_pca": Y_pca,
        "X_raw": X_raw,
        "Y_raw": Y_raw,
        "R_raw": R_raw,
        "input_pca": input_pca,
        "output_pca": output_pca,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "input_var_explained": input_pca.explained_variance_ratio_,
        "output_var_explained": output_pca.explained_variance_ratio_,
        "meta": meta,
        "input_feature_names": input_feature_names,
        "input_layer": layer_idx - 1,
        "output_layer": layer_idx,
        "n_samples": len(df_in),
        "n_raw_dims": len(dim_cols_in),
        "mode": "residual",
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_layer_overview(io_data: dict, title_prefix: str = ""):
    """
    Overview plot showing input vs output signal structure.
    """
    X_pca = io_data["X_pca"]
    Y_pca = io_data["Y_pca"]
    meta = io_data["meta"]
    in_var = io_data["input_var_explained"]
    out_var = io_data["output_var_explained"]

    n_in = X_pca.shape[1]
    n_out = Y_pca.shape[1]

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Row 1: Input PCs (first 4)
    for i in range(min(4, n_in)):
        ax = fig.add_subplot(gs[0, i])
        sc = ax.scatter(meta["token_pos"], X_pca[:, i],
                       c=meta["prompt_idx"], cmap='tab10', s=15, alpha=0.7, edgecolors='none')
        ax.set_xlabel("token_pos", fontsize=9)
        ax.set_ylabel(f"in_pc{i}", fontsize=9)
        ax.set_title(f"Input PC{i} ({in_var[i]*100:.1f}%)", fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)

    # Row 2: Output PCs (first 4)
    for i in range(min(4, n_out)):
        ax = fig.add_subplot(gs[1, i])
        sc = ax.scatter(meta["token_pos"], Y_pca[:, i],
                       c=meta["prompt_idx"], cmap='tab10', s=15, alpha=0.7, edgecolors='none')
        ax.set_xlabel("token_pos", fontsize=9)
        ax.set_ylabel(f"out_pc{i}", fontsize=9)
        ax.set_title(f"Output PC{i} ({out_var[i]*100:.1f}%)", fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.2)

    # Row 3: Cross-correlation structure
    ax_corr = fig.add_subplot(gs[2, 0:2])
    corr_matrix = np.corrcoef(X_pca.T, Y_pca.T)[:n_in, n_in:]
    im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax_corr.set_xlabel("Output PCs", fontsize=10)
    ax_corr.set_ylabel("Input PCs", fontsize=10)
    ax_corr.set_title("Input↔Output PC Correlation", fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax_corr)

    # Variance explained bar chart
    ax_var = fig.add_subplot(gs[2, 2])
    n_bars = min(10, len(in_var), len(out_var))
    x_pos = np.arange(n_bars)
    ax_var.bar(x_pos - 0.15, in_var[:n_bars] * 100, width=0.3,
               color='steelblue', label='Input', alpha=0.8)
    ax_var.bar(x_pos + 0.15, out_var[:n_bars] * 100, width=0.3,
               color='coral', label='Output', alpha=0.8)
    ax_var.set_xlabel("PC index", fontsize=9)
    ax_var.set_ylabel("Var explained (%)", fontsize=9)
    ax_var.set_title("PCA Variance", fontsize=10, fontweight='bold')
    ax_var.legend(fontsize=8)
    ax_var.grid(True, alpha=0.2, axis='y')

    # Input vs output scatter for top PCs
    ax_io = fig.add_subplot(gs[2, 3])
    ax_io.scatter(X_pca[:, 0], Y_pca[:, 0], s=15, alpha=0.5, c='steelblue', edgecolors='none')
    ax_io.set_xlabel("Input PC0", fontsize=9)
    ax_io.set_ylabel("Output PC0", fontsize=9)
    ax_io.set_title("in_pc0 → out_pc0", fontsize=10, fontweight='bold')
    ax_io.grid(True, alpha=0.2)

    mode_str = io_data.get("mode", "full output")
    fig.suptitle(
        f"{title_prefix}Layer {io_data['input_layer']}→{io_data['output_layer']} "
        f"({mode_str}) | {io_data['n_samples']} samples × {io_data['n_raw_dims']} dims",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.5)


def plot_sr_results(X, y_true, model, feature_names, label):
    """Consolidated 6-panel plot for one SR fit."""
    y_pred = model.predict(X)

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Real vs Predicted scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    ax1.set_xlabel("Real", fontsize=11)
    ax1.set_ylabel("Predicted", fontsize=11)
    ax1.set_title(f"Real vs Predicted — {label}", fontsize=11, fontweight='bold')
    ax1.legend()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ax1.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 2: Signal overlay
    ax2 = fig.add_subplot(gs[0, 1])
    n_show = min(200, len(y_true))
    ax2.plot(y_true[:n_show], label='Real', color='steelblue', linewidth=1.0)
    ax2.plot(y_pred[:n_show], label='Predicted', color='orangered', linewidth=1.0, alpha=0.8)
    ax2.set_xlabel("Sample index", fontsize=11)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title(f"Signal overlay (first {n_show})", fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    # Panel 3: Residuals
    ax3 = fig.add_subplot(gs[0, 2])
    residuals = y_true - y_pred
    ax3.hist(residuals, bins=50, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axvline(0, color='black', linewidth=1, linestyle='--')
    ax3.set_xlabel("Residual", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title(f"Residuals — MAE={np.mean(np.abs(residuals)):.4f}", fontsize=11, fontweight='bold')

    # Panel 4: Feature importance — scatter vs top 2 input PCs
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(X[:, 0], y_true, alpha=0.4, s=15, c='steelblue', edgecolors='none', label='Real')
    ax4.scatter(X[:, 0], y_pred, alpha=0.3, s=15, c='orangered', edgecolors='none', label='Predicted')
    ax4.set_xlabel(feature_names[0], fontsize=11)
    ax4.set_ylabel("Output value", fontsize=11)
    ax4.set_title(f"vs {feature_names[0]}", fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.2)

    # Panel 5: vs second feature
    ax5 = fig.add_subplot(gs[1, 1])
    if X.shape[1] > 1:
        ax5.scatter(X[:, 1], y_true, alpha=0.4, s=15, c='steelblue', edgecolors='none', label='Real')
        ax5.scatter(X[:, 1], y_pred, alpha=0.3, s=15, c='orangered', edgecolors='none', label='Predicted')
        ax5.set_xlabel(feature_names[1], fontsize=11)
        ax5.set_ylabel("Output value", fontsize=11)
        ax5.set_title(f"vs {feature_names[1]}", fontsize=11, fontweight='bold')
        ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.2)

    # Panel 6: Pareto front
    ax6 = fig.add_subplot(gs[1, 2])
    eqs = model.equations_
    ax6.semilogy(eqs["complexity"], eqs["loss"], 'o-', color='darkgreen', markersize=6)
    best_idx = eqs["loss"].idxmin()
    ax6.semilogy(eqs.loc[best_idx, "complexity"], eqs.loc[best_idx, "loss"],
                 '*', color='red', markersize=15, label='Best')
    ax6.set_xlabel("Complexity", fontsize=11)
    ax6.set_ylabel("Loss (log)", fontsize=11)
    ax6.set_title("Pareto Front", fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Best equation text
    best_eq_str = str(model.sympy())
    fig.text(0.5, 0.01, f"Best equation: y = {best_eq_str}",
             ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             wrap=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show(block=False)
    plt.pause(0.5)


def plot_layer_summary(all_results: list[dict], exp_name: str):
    """Summary plot across all layers."""
    if not all_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Best loss per layer per output PC
    ax1 = axes[0]
    for res in all_results:
        lid = res["output_layer"]
        for eq_info in res["equations"]:
            ax1.scatter(lid, eq_info["loss"], c='steelblue', s=40, alpha=0.7,
                       edgecolors='black', linewidth=0.5)
    ax1.set_xlabel("Output Layer", fontsize=11)
    ax1.set_ylabel("Best Loss (log)", fontsize=11)
    ax1.set_yscale('log')
    ax1.set_title("SR Loss per Layer", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: R² per layer
    ax2 = axes[1]
    for res in all_results:
        lid = res["output_layer"]
        for eq_info in res["equations"]:
            ax2.scatter(lid, eq_info.get("r2", 0), c='coral', s=40, alpha=0.7,
                       edgecolors='black', linewidth=0.5)
    ax2.set_xlabel("Output Layer", fontsize=11)
    ax2.set_ylabel("R²", fontsize=11)
    ax2.set_title("R² per Layer", fontsize=12, fontweight='bold')
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Complexity per layer
    ax3 = axes[2]
    for res in all_results:
        lid = res["output_layer"]
        for eq_info in res["equations"]:
            ax3.scatter(lid, eq_info["complexity"], c='teal', s=40, alpha=0.7,
                       edgecolors='black', linewidth=0.5)
    ax3.set_xlabel("Output Layer", fontsize=11)
    ax3.set_ylabel("Equation Complexity", fontsize=11)
    ax3.set_title("Complexity per Layer", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"Layer Approximation Summary — {exp_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


# =============================================================================
# Core SR
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
        batch_size=min(128, len(X)),
        random_state=42,
    )

    print(f"    [SR] Fitting: {label}")
    print(f"    [SR] X shape: {X.shape}, y shape: {y.shape}")
    print(f"    [SR] Features: {feature_names}")
    model.fit(X, y, variable_names=feature_names)
    return model


def print_equations_table(model, label: str):
    """Print the Pareto front."""
    eqs = model.equations_
    print(f"\n{'='*120}")
    print(f"PARETO FRONT — {label}")
    print(f"{'='*120}")
    print(f"{'Complexity':<12} {'Loss':<14} {'Score':<12} Equation")
    print(f"{'-'*120}")
    for _, row in eqs.iterrows():
        c = int(row["complexity"])
        loss = row["loss"]
        score = row.get("score", 0.0)
        eq = row["equation"]
        print(f"{c:<12} {loss:<14.6e} {score:<12.6e} y = {eq}")
    print(f"{'='*120}\n")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_layer_spec(spec: str, available_layers: list[int]) -> list[int]:
    """Parse layer specification. For this script, layers must be >= 1 (need L-1 as input)."""
    spec = spec.strip().lower()
    max_layer = max(available_layers)

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

    # Filter: must be in available AND must have L-1 also available
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
        description="Symbolic regression: approximate real LLM layers (input→output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Approximate layer 20 (using layer 19 as input)
  uv run symbolic_regression.py -l 20

  # Approximate the residual (what the layer adds) instead of full output
  uv run symbolic_regression.py -l 20 --residual

  # Multiple layers
  uv run symbolic_regression.py -l 10-15

  # More input PCs for richer symbolic expressions
  uv run symbolic_regression.py -l 20 --input-pcs 15

  # Specific experiment
  uv run symbolic_regression.py -e capital_paris_multilingual -l 32
""",
    )

    parser.add_argument("--experiment", "-e", type=str, default=None,
                        help="Experiment name (default: first found)")
    parser.add_argument("--results-dir", "-r", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--layers", "-l", type=str, default="20",
                        help="Layer spec: number, range (10-15), or group name. "
                             "Must be >= 1 (needs L-1 as input). Default: 20")

    # Mode
    parser.add_argument("--residual", action="store_true",
                        help="Model the residual (what layer adds) instead of full output. "
                             "Better for transformers since output ≈ input + layer(input).")

    # PCA configuration
    parser.add_argument("--input-pcs", type=int, default=10,
                        help="Number of input PCA components (features for SR). Default: 10")
    parser.add_argument("--output-pcs", type=int, default=5,
                        help="Number of output PCA components (targets for SR). Default: 5")
    parser.add_argument("--max-targets", type=int, default=3,
                        help="Max number of output PCs to fit with SR. Default: 3")

    # SR parameters
    parser.add_argument("--iterations", "-i", type=int, default=40)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--parsimony", type=float, default=0.0032)
    parser.add_argument("--maxsize", type=int, default=30)
    parser.add_argument("--binary-ops", type=str, default="+,*,/,^,-")
    parser.add_argument("--unary-ops", type=str, default="sin,cos,exp,log,sqrt,tanh,abs")

    # Output
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable all matplotlib plots")
    parser.add_argument("--list-experiments", action="store_true")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # =========================================================================
    # List experiments mode
    # =========================================================================
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
            print(f"  {exp}  (layers: {min(layers)}-{max(layers)}, count={len(layers)}, model: {model_name})")
        return

    # =========================================================================
    # Resolve experiment
    # =========================================================================
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
    layers = parse_layer_spec(args.layers, available_layers)

    if not layers:
        print(f"ERROR: No valid layers found. Available: {available_layers}", file=sys.stderr)
        print(f"  (Each layer L needs layer L-1 to also exist as input.)", file=sys.stderr)
        sys.exit(1)

    model_info = {}
    info_path = exp_dir / "model_info.json"
    if info_path.exists():
        model_info = json.loads(info_path.read_text())

    # =========================================================================
    # Print configuration
    # =========================================================================
    mode_str = "RESIDUAL (what layer adds)" if args.residual else "FULL OUTPUT"
    print("=" * 90)
    print("SYMBOLIC REGRESSION: LAYER INPUT → OUTPUT APPROXIMATION")
    print("=" * 90)
    print(f"  Experiment:     {exp_name}")
    print(f"  Model:          {model_info.get('model_name', 'unknown')}")
    print(f"  Layer(s):       {layers}")
    print(f"  Mode:           {mode_str}")
    print(f"  Input PCs:      {args.input_pcs} (features for SR)")
    print(f"  Output PCs:     {args.output_pcs} (targets, fitting top {args.max_targets})")
    print(f"  SR iterations:  {args.iterations}")
    print(f"  SR maxsize:     {args.maxsize}")
    print(f"  Operators:      binary={args.binary_ops}  unary={args.unary_ops}")
    print(f"  Plots:          {'disabled' if args.no_plots else 'enabled'}")
    print("=" * 90)
    print()

    # =========================================================================
    # Main loop: for each layer, build input→output, run SR
    # =========================================================================
    all_results = []

    for layer_idx in layers:
        print(f"\n{'#' * 90}")
        print(f"### LAYER TRANSITION: {layer_idx - 1} → {layer_idx}")
        print(f"{'#' * 90}")

        # Build input→output data
        try:
            if args.residual:
                io_data = build_residual_io(
                    exp_dir, layer_idx,
                    n_input_pcs=args.input_pcs,
                    n_output_pcs=args.output_pcs,
                )
            else:
                io_data = build_layer_io(
                    exp_dir, layer_idx,
                    n_input_pcs=args.input_pcs,
                    n_output_pcs=args.output_pcs,
                )
        except Exception as e:
            print(f"  ERROR loading layer pair: {e}")
            continue

        X_pca = io_data["X_pca"]
        Y_pca = io_data["Y_pca"]
        feat_names = io_data["input_feature_names"]
        in_var = io_data["input_var_explained"]
        out_var = io_data["output_var_explained"]

        print(f"  Samples:        {io_data['n_samples']}")
        print(f"  Raw dims:       {io_data['n_raw_dims']}")
        print(f"  Input PCA var:  {in_var.sum()*100:.1f}% total, top-3: {(in_var[:3]*100).round(1).tolist()}")
        print(f"  Output PCA var: {out_var.sum()*100:.1f}% total, top-3: {(out_var[:3]*100).round(1).tolist()}")

        # Show overview plot
        if not args.no_plots:
            plot_layer_overview(io_data, title_prefix=f"{exp_name} | ")

        # Run SR for each output PC
        n_targets = min(args.max_targets, Y_pca.shape[1])
        layer_result = {
            "input_layer": layer_idx - 1,
            "output_layer": layer_idx,
            "mode": "residual" if args.residual else "full",
            "input_var_explained": in_var.tolist(),
            "output_var_explained": out_var.tolist(),
            "equations": [],
        }

        for target_idx in range(n_targets):
            y_target = Y_pca[:, target_idx]
            target_label = f"out_pc{target_idx}"
            full_label = f"L{layer_idx-1}→L{layer_idx}_{target_label}"

            if np.std(y_target) < 1e-10:
                print(f"  {target_label}: SKIPPED (zero variance)")
                continue

            var_pct = out_var[target_idx] * 100
            print(f"\n  --- Fitting {full_label} (explains {var_pct:.1f}% of output variance) ---")
            print(f"      X: {X_pca.shape}, y std: {np.std(y_target):.4f}")

            # Run symbolic regression
            model = run_symbolic_regression(X_pca, y_target, feat_names, full_label, args)

            # Print equations
            print_equations_table(model, full_label)

            # Compute R²
            y_pred = model.predict(X_pca)
            ss_res = np.sum((y_target - y_pred) ** 2)
            ss_tot = np.sum((y_target - np.mean(y_target)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Plot
            if not args.no_plots:
                plot_sr_results(X_pca, y_target, model, feat_names, full_label)

            # Store result
            best_eq = model.sympy()
            eqs_df = model.equations_
            best_row = eqs_df.iloc[eqs_df["loss"].idxmin()]

            eq_info = {
                "target": target_label,
                "var_explained_pct": float(var_pct),
                "equation": str(best_eq),
                "loss": float(best_row["loss"]),
                "r2": float(r2),
                "complexity": int(best_row["complexity"]),
                "features": feat_names,
            }
            layer_result["equations"].append(eq_info)

            print(f"  ✓ {target_label}: y = {best_eq}")
            print(f"    R²={r2:.4f}  loss={eq_info['loss']:.4e}  complexity={eq_info['complexity']}")

        all_results.append(layer_result)

    # =========================================================================
    # Summary
    # =========================================================================
    if not args.no_plots and all_results:
        plot_layer_summary(all_results, exp_name)

    # Print text summary
    print("\n" + "=" * 90)
    print("SUMMARY: BEST EQUATIONS PER LAYER")
    print("=" * 90)
    for res in all_results:
        in_l = res["input_layer"]
        out_l = res["output_layer"]
        print(f"\n  Layer {in_l} → {out_l} ({res['mode']}):")
        for eq_info in res["equations"]:
            print(f"    {eq_info['target']} ({eq_info['var_explained_pct']:.1f}% var): "
                  f"y = {eq_info['equation']}")
            print(f"      R²={eq_info['r2']:.4f}  loss={eq_info['loss']:.4e}  "
                  f"complexity={eq_info['complexity']}")
    print("=" * 90)

    # =========================================================================
    # Save results JSON
    # =========================================================================
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
        print(f"\nResults saved to {output_path}")

    print("\nDone.")
    if not args.no_plots:
        plt.show(block=True)


if __name__ == "__main__":
    main()
