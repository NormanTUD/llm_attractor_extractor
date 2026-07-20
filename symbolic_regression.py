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

Bootstraps itself via `uv run symbolic_regression.py`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    """Build X (features) and y (activations) from a layer dataframe.

    Returns:
        X: (n_samples, n_features) - [prompt_idx, token_pos]
        y: (n_samples, d_model) - activation vectors
        token_texts: list of token strings for labeling
    """
    meta_cols = ["prompt_idx", "token_pos", "token_text"]
    X = df[["prompt_idx", "token_pos"]].values.astype(float)
    y = df.drop(columns=meta_cols).values.astype(float)
    token_texts = df["token_text"].tolist()
    return X, y, token_texts


def build_cross_layer_matrix(
    experiment_dir: Path, layer_indices: list[int]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build matrices across multiple layers.

    Returns:
        X: (n_samples, 3) - [layer_idx, prompt_idx, token_pos]
        y: (n_samples, d_model) - activation vectors
        labels: list of label strings
    """
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
        timeout_per_equation=args.timeout,
        deterministic=True,
        random_seed=42,
        progress=False,
        batching=True,
        batch_size=min(64, len(X)),
    )
    model.fit(X, y, variable_names=feature_names)
    return model


def parse_layer_spec(spec: str, max_layer: int = 63) -> list[int]:
    """Parse layer specification like '5', '10-20', 'early', '5,10,15'."""
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


def format_equation(eq_str: str, score: float, loss: float) -> str:
    return f"  {eq_str}  (loss={loss:.4e}, score={score:.4f})"


def main():
    parser = argparse.ArgumentParser(
        description="Symbolic regression on LLM layer activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults on first experiment
  uv run symbolic_regression.py

  # Specific experiment, specific layers
  uv run symbolic_regression.py --experiment capital_varied_english --layers early

  # Cross-layer analysis with PCA
  uv run symbolic_regression.py --mode cross-layer --layers 0-10 --pca-components 5

  # Custom operators and iterations
  uv run symbolic_regression.py --binary-ops "+,*,/,^,-" --iterations 50
""",
    )

    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default=None,
        help="Experiment directory name (default: first discovered)",
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=str(RESULTS_DIR),
        help=f"Results root directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--layers", "-l",
        type=str,
        default="all",
        help="Layer spec: integer, range (5-20), comma-separated (5,10,15), "
             "or group name: early/mid/late/first-half/second-half/all (default: all)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["per-layer", "cross-layer"],
        default="per-layer",
        help="Analysis mode: per-layer (separate fit per layer) or "
             "cross-layer (joint fit across layers) (default: per-layer)",
    )
    parser.add_argument(
        "--pca", "-p",
        type=int,
        default=10,
        help="Number of PCA components (0 to disable) (default: 10)",
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=30,
        help="PySR iterations (default: 30)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=30,
        help="PySR population size (default: 30)",
    )
    parser.add_argument(
        "--parsimony",
        type=float,
        default=0.0032,
        help="PySR parsimony coefficient (default: 0.0032)",
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        default=25,
        help="Max equation size for PySR (default: 25)",
    )
    parser.add_argument(
        "--binary-ops",
        type=str,
        default="+,*,/,^,-",
        help="Binary operators (default: +,*,/,^,-)",
    )
    parser.add_argument(
        "--unary-ops",
        type=str,
        default="sin,cos,exp,log,sqrt,tanh,abs",
        help="Unary operators (default: sin,cos,exp,log,sqrt,tanh,abs)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per equation in seconds (default: 60)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of best equations to show (default: 3)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results JSON (default: stdout summary)",
    )
    parser.add_argument(
        "--max-dims",
        type=int,
        default=None,
        help="Max dimensions to fit (default: all PCA components or all dims)",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiments and exit",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.list_experiments:
        for exp in discover_experiments(results_dir):
            exp_dir = results_dir / exp
            layers = discover_layers(exp_dir)
            info_path = exp_dir / "model_info.json"
            model = ""
            if info_path.exists():
                info = json.loads(info_path.read_text())
                model = info.get("model_name", "")
            print(f"  {exp}  (layers: {len(layers)}, model: {model})")
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

    print(f"Experiment: {exp_name}")
    print(f"Model:      {model_info.get('model_name', 'unknown')}")
    print(f"Layers:     {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Mode:       {args.mode}")
    print(f"PCA:        {args.pca} components" if args.pca > 0 else "PCA:        disabled")
    print(f"Iterations: {args.iterations}")
    print(f"Operators:  binary={args.binary_ops}  unary={args.unary_ops}")
    print()

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
                y_target = y_reduced[:, dim_idx]

                if np.std(y_target) < 1e-10:
                    continue

                model = run_symbolic_regression(
                    X_raw, y_target, feature_names, label, args
                )

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

                print(f"  {label}: {best_eq}")
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
            y_target = y_reduced[:, dim_idx]

            if np.std(y_target) < 1e-10:
                continue

            model = run_symbolic_regression(
                X_raw, y_target, feature_names, label, args
            )

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

            print(f"  {label}: {best_eq}")
            print(f"         loss={result_entry['loss']:.4e}  complexity={result_entry['complexity']}")

        all_results.append(cross_results)
        print()

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
