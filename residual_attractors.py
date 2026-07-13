# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "numpy",
#     "safetensors",
#     "huggingface_hub",
#     "tqdm",
#     "matplotlib",
#     "pandas",
#     "scikit-learn",
# ]
# ///
"""
Residual Stream Attractor Extraction Tool v2

Changes from v1:
- Fixed CSV encoding (UTF-8 enforced everywhere)
- Shows predicted next token prominently in all outputs
- Animated visualization of points converging to attractors across layers
- Attractor geometry extraction (point, ring, torus classification)
- New prompt groups designed to produce non-point attractors (ambiguous predictions)

Usage:
    uv run residual_attractors.py [--model-preset gpt2] [--device auto]
"""

import sys
import os
import shutil
import subprocess

# =============================================================================
# Auto-restart under `uv run` if invoked directly with python3
# =============================================================================

def _ensure_uv_run():
    if os.environ.get("_UV_RUN_ACTIVE") == "1":
        return

    uv_path = shutil.which("uv")

    if uv_path is None:
        print("=" * 60)
        print("ERROR: This script must be run with `uv run` but `uv` was")
        print("not found on your system.")
        print("=" * 60)
        print()
        print("To install uv:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print()
        print("Then run:")
        print(f"  uv run {os.path.basename(__file__)}")
        print("=" * 60)
        sys.exit(1)

    script_path = os.path.abspath(__file__)
    extra_args = sys.argv[1:]
    cmd = [uv_path, "run", script_path] + extra_args

    print(f"[auto-restart] Re-launching with: {' '.join(cmd)}")
    print()

    env = os.environ.copy()
    env["_UV_RUN_ACTIVE"] = "1"

    if sys.platform == "win32":
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    else:
        os.execvpe(uv_path, cmd, env)


_ensure_uv_run()

# =============================================================================
# Actual imports (only reached after uv installs dependencies)
# =============================================================================

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# =============================================================================
# Prompt Group Definitions
# =============================================================================

@dataclass
class PromptGroup:
    """A group of prompts that should all predict the same target token."""
    name: str
    target_token: str
    attractor_concept: str
    expected_geometry: str = "point"  # "point", "ring", "torus", "cloud"
    prompts: list[str] = field(default_factory=list)


ATTRACTOR_GROUPS: list[PromptGroup] = [
    # === EXISTING GROUPS (all point attractors) ===
    PromptGroup(
        name="paris_multilingual",
        target_token="Paris",
        attractor_concept="Paris (capital of France)",
        expected_geometry="point",
        prompts=[
            "The capital of France is",
            "Die Hauptstadt von Frankreich ist",
            "La capitale de la France est",
            "La capital de Francia es",
            "\u30d5\u30e9\u30f3\u30b9\u306e\u9996\u90fd\u306f",
            "The Eiffel Tower is located in",
            "Der Eiffelturm steht in",
            "The city of love is",
            "The Louvre museum is in",
            "France's largest city is",
            "The Seine river flows through",
            "Notre-Dame cathedral is in",
        ],
    ),
    PromptGroup(
        name="berlin_multilingual",
        target_token="Berlin",
        attractor_concept="Berlin (capital of Germany)",
        expected_geometry="point",
        prompts=[
            "The capital of Germany is",
            "Die Hauptstadt von Deutschland ist",
            "La capitale de l'Allemagne est",
            "La capital de Alemania es",
            "\u30c9\u30a4\u30c4\u306e\u9996\u90fd\u306f",
            "The Brandenburg Gate is in",
            "Das Brandenburger Tor steht in",
            "The Berlin Wall divided",
            "Germany's largest city is",
            "The Reichstag building is in",
            "The Spree river flows through",
            "Checkpoint Charlie was in",
        ],
    ),
    PromptGroup(
        name="tokyo_multilingual",
        target_token="Tokyo",
        attractor_concept="Tokyo (capital of Japan)",
        expected_geometry="point",
        prompts=[
            "The capital of Japan is",
            "Die Hauptstadt von Japan ist",
            "La capitale du Japon est",
            "\u65e5\u672c\u306e\u9996\u90fd\u306f",
            "The largest city in Japan is",
            "Mount Fuji can be seen from",
            "The Shibuya crossing is in",
            "The Tokyo Tower is located in",
            "Japan's imperial palace is in",
        ],
    ),
    PromptGroup(
        name="london_multilingual",
        target_token="London",
        attractor_concept="London (capital of UK)",
        expected_geometry="point",
        prompts=[
            "The capital of England is",
            "The capital of the United Kingdom is",
            "Die Hauptstadt von England ist",
            "La capitale de l'Angleterre est",
            "Big Ben is located in",
            "The Thames river flows through",
            "Buckingham Palace is in",
            "The Tower Bridge is in",
            "Baker Street 221B is in",
        ],
    ),
    PromptGroup(
        name="four_arithmetic",
        target_token="4",
        attractor_concept="The number 4",
        expected_geometry="point",
        prompts=[
            "2 + 2 =",
            "8 / 2 =",
            "1 + 3 =",
            "The square root of 16 is",
            "Two plus two equals",
            "Zwei plus zwei ist",
            "The number of seasons in a year is",
        ],
    ),
    PromptGroup(
        name="water_concept",
        target_token="water",
        attractor_concept="Water (H2O, the substance)",
        expected_geometry="point",
        prompts=[
            "H2O is commonly known as",
            "The chemical formula for water is H2O. Water is also called",
            "Humans need to drink",
            "The ocean is made of salt",
            "Ice melts and becomes",
            "Rain is falling",
            "The liquid that comes from a tap is",
        ],
    ),
    PromptGroup(
        name="sun_concept",
        target_token="Sun",
        attractor_concept="The Sun (our star)",
        expected_geometry="point",
        prompts=[
            "The star at the center of our solar system is the",
            "Earth orbits around the",
            "Der Stern im Zentrum unseres Sonnensystems ist die",
            "The largest object in our solar system is the",
            "Sunlight comes from the",
            "Solar energy is produced by the",
        ],
    ),
    PromptGroup(
        name="einstein_person",
        target_token="Einstein",
        attractor_concept="Albert Einstein",
        expected_geometry="point",
        prompts=[
            "E = mc\u00b2 was discovered by Albert",
            "The theory of relativity was developed by",
            "The most famous physicist of the 20th century is Albert",
            "Die Relativit\u00e4tstheorie wurde entwickelt von Albert",
            "The Nobel Prize in Physics 1921 was awarded to Albert",
        ],
    ),

    # === NEW GROUPS: Expected NON-POINT attractors ===

    # RING/ANNULAR attractor: Multiple equally-likely color tokens
    # The model should be uncertain between several colors → residual streams
    # should form a ring or cloud around the color subspace
    PromptGroup(
        name="color_ambiguous_ring",
        target_token="[ambiguous:color]",
        attractor_concept="An ambiguous color (red/blue/green/yellow equally likely)",
        expected_geometry="ring",
        prompts=[
            "My favorite color is",
            "The color I like most is",
            "She painted the wall",
            "His shirt was",
            "The car was painted",
            "I chose the",
            "The flower was",
            "Her dress was a beautiful shade of",
            "The sky turned a deep",
            "He picked up the",
            "The balloon was",
            "They decorated the room in",
            "The bird had bright",
            "The candy was colored",
            "She dyed her hair",
        ],
    ),

    # TORUS attractor: Two independent ambiguity axes
    # Axis 1: name (John/Mary/James/Sarah...) - who
    # Axis 2: action (said/went/took/saw...) - what they did
    # The combination should create a 2D manifold (torus-like)
    PromptGroup(
        name="name_action_torus",
        target_token="[ambiguous:name]",
        attractor_concept="Ambiguous person name (many equally likely names)",
        expected_geometry="torus",
        prompts=[
            "Once upon a time, there was a person named",
            "The story begins with a character called",
            "In the village, everyone knew",
            "The protagonist of the novel is",
            "Dear",
            "Hello, my name is",
            "The patient's name is",
            "The suspect was identified as",
            "The winner of the competition is",
            "Ladies and gentlemen, please welcome",
            "The new employee is called",
            "The teacher introduced herself as",
            "The detective's name was",
            "Born in 1990,",
            "The hero of our story,",
        ],
    ),

    # RING attractor: Day of the week (7 equally likely options forming a cycle)
    PromptGroup(
        name="weekday_ring",
        target_token="[ambiguous:weekday]",
        attractor_concept="Day of the week (cyclic, 7 options)",
        expected_geometry="ring",
        prompts=[
            "Today is",
            "The meeting is scheduled for",
            "I was born on a",
            "The appointment is on",
            "Let's meet on",
            "The deadline is",
            "School starts on",
            "The concert is on",
            "My day off is",
            "The flight departs on",
            "The exam is on",
            "We always go shopping on",
        ],
    ),

    # CLOUD attractor: Highly ambiguous continuation (many tokens possible)
    PromptGroup(
        name="open_ended_cloud",
        target_token="[ambiguous:anything]",
        attractor_concept="Maximally ambiguous continuation (high entropy)",
        expected_geometry="cloud",
        prompts=[
            "The",
            "I",
            "It",
            "There",
            "When",
            "After",
            "Before",
            "If",
            "Although",
            "However",
            "Meanwhile",
            "Furthermore",
        ],
    ),

    # RING attractor: Numbers 1-10 (ordered, cyclic-ish)
    PromptGroup(
        name="number_ambiguous_ring",
        target_token="[ambiguous:number]",
        attractor_concept="Ambiguous number (1-10 equally plausible)",
        expected_geometry="ring",
        prompts=[
            "Pick a number between 1 and 10:",
            "I'm thinking of a number. It is",
            "The winning lottery number is",
            "Roll the dice. The result is",
            "On a scale of 1 to 10, I'd rate it",
            "The answer to the riddle is the number",
            "Chapter",
            "Question number",
            "The score was",
            "He held up",
        ],
    ),
]


# =============================================================================
# Attractor Geometry Analyzer
# =============================================================================

class AttractorGeometryAnalyzer:
    """Analyzes the geometric structure of attractor clusters.
    
    Classifies attractors as:
    - point: All streams converge to a single point (1 dominant eigenvalue ~ 0)
    - ring: Streams form a 1D manifold (1 large eigenvalue, rest small)
    - torus: Streams form a 2D manifold (2 large eigenvalues, rest small)
    - cloud: No clear structure (many comparable eigenvalues)
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def analyze(self, streams: np.ndarray, group_name: str = "") -> dict:
        """
        Analyze geometry of a set of residual stream vectors.
        
        Args:
            streams: (n_prompts, d_model) array of final-layer residual streams
            group_name: name for reporting
            
        Returns:
            dict with geometry classification and metrics
        """
        n_samples, d_model = streams.shape

        if n_samples < 3:
            return {
                "group_name": group_name,
                "classification": "insufficient_data",
                "n_samples": n_samples,
            }

        # Center the data
        centroid = streams.mean(axis=0)
        centered = streams - centroid

        # PCA
        n_comp = min(self.n_components, n_samples - 1, d_model)
        pca = PCA(n_components=n_comp)
        projected = pca.fit_transform(centered)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        # Compute spread metrics
        total_variance = np.sum(pca.explained_variance_)
        mean_distance_to_centroid = np.mean(np.linalg.norm(centered, axis=1))
        max_distance_to_centroid = np.max(np.linalg.norm(centered, axis=1))

        # Classification heuristics
        # Point: very low total variance relative to centroid norm
        centroid_norm = np.linalg.norm(centroid)
        relative_spread = mean_distance_to_centroid / (centroid_norm + 1e-10)

        # Dimensionality estimation: how many PCs needed for 90% variance?
        dims_for_90 = int(np.searchsorted(cumulative_variance, 0.90)) + 1
        dims_for_95 = int(np.searchsorted(cumulative_variance, 0.95)) + 1

        # Eigenvalue ratios
        if len(explained_variance) >= 2:
            ratio_1_2 = explained_variance[0] / (explained_variance[1] + 1e-10)
        else:
            ratio_1_2 = float('inf')

        if len(explained_variance) >= 3:
            ratio_2_3 = explained_variance[1] / (explained_variance[2] + 1e-10)
        else:
            ratio_2_3 = float('inf')

        # Classification logic
        if relative_spread < 0.01:
            classification = "point"
            confidence = 1.0 - relative_spread * 100
        elif dims_for_90 == 1 and ratio_1_2 > 5.0:
            classification = "ring"  # 1D manifold
            confidence = min(ratio_1_2 / 10.0, 1.0)
        elif dims_for_90 <= 2 and ratio_2_3 > 3.0:
            classification = "torus"  # 2D manifold
            confidence = min(ratio_2_3 / 6.0, 1.0)
        elif dims_for_90 <= 3:
            classification = "low_dim_manifold"
            confidence = 0.5
        else:
            classification = "cloud"
            confidence = 1.0 - (3.0 / dims_for_90)

        # Check for ring structure: project onto first 2 PCs and check circularity
        ring_score = 0.0
        if n_samples >= 5 and n_comp >= 2:
            proj_2d = projected[:, :2]
            # Check if points form a ring: variance of distances from center should be low
            dists_from_center = np.linalg.norm(proj_2d, axis=1)
            if dists_from_center.mean() > 1e-10:
                ring_score = 1.0 - (dists_from_center.std() / dists_from_center.mean())
                ring_score = max(0, ring_score)

        # Check for torus: project onto first 3 PCs and check 2D manifold structure
        torus_score = 0.0
        if n_samples >= 8 and n_comp >= 3:
            proj_3d = projected[:, :3]
            # Simple heuristic: if 2 PCs explain most variance and they're comparable
            if explained_variance[0] > 0 and explained_variance[1] > 0:
                balance = min(explained_variance[0], explained_variance[1]) / max(explained_variance[0], explained_variance[1])
                torus_score = balance * (1.0 - explained_variance[2] / explained_variance[1]) if explained_variance[1] > 0 else 0

        return {
            "group_name": group_name,
            "classification": classification,
            "confidence": float(confidence),
            "n_samples": n_samples,
            "relative_spread": float(relative_spread),
            "mean_distance_to_centroid": float(mean_distance_to_centroid),
            "max_distance_to_centroid": float(max_distance_to_centroid),
            "centroid_norm": float(centroid_norm),
            "total_variance": float(total_variance),
            "explained_variance_ratios": explained_variance.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "dims_for_90_pct": dims_for_90,
            "dims_for_95_pct": dims_for_95,
            "eigenvalue_ratio_1_2": float(ratio_1_2),
            "eigenvalue_ratio_2_3": float(ratio_2_3),
            "ring_score": float(ring_score),
            "torus_score": float(torus_score),
            "pca_projected_2d": projected[:, :2].tolist() if n_comp >= 2 else None,
            "pca_projected_3d": projected[:, :3].tolist() if n_comp >= 3 else None,
        }


# =============================================================================
# CSV Data Saver (with UTF-8 fix and next-token display)
# =============================================================================

class RawDataSaver:
    """Saves all residual stream data as CSV files with proper UTF-8 encoding."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_group(self, group: PromptGroup, results: list[dict]):
        """Save all raw data for a prompt group."""
        group_dir = self.output_dir / group.name
        group_dir.mkdir(exist_ok=True)

        n_layers = results[0]["n_layers"]

        # --- 1. Raw streams: group/raw_streams/layer_XXX/prompt_YYY.csv ---
        raw_dir = group_dir / "raw_streams"
        raw_dir.mkdir(exist_ok=True)

        for layer_idx in tqdm(range(n_layers), desc=f"  Saving raw streams ({group.name})"):
            layer_dir = raw_dir / f"layer_{layer_idx:03d}"
            layer_dir.mkdir(exist_ok=True)

            for prompt_idx, result in enumerate(results):
                stream = result["full_streams"][layer_idx]  # (seq_len, d_model)
                d_model = stream.shape[1]
                columns = [f"dim_{d:04d}" for d in range(d_model)]

                df = pd.DataFrame(stream, columns=columns)
                df.insert(0, "token_pos", range(len(df)))
                tokens_list = result["tokens"][:len(df)] if len(result["tokens"]) >= len(df) else result["tokens"] + [""] * (len(df) - len(result["tokens"]))
                df.insert(1, "token", tokens_list)

                # UTF-8 encoding explicitly
                df.to_csv(layer_dir / f"prompt_{prompt_idx:03d}.csv", index=False, encoding="utf-8")

        # --- 2. Final token streams with NEXT TOKEN info ---
        final_dir = group_dir / "final_token_streams"
        final_dir.mkdir(exist_ok=True)

        d_model = results[0]["final_token_streams"][0].shape[0]
        dim_columns = [f"dim_{d:04d}" for d in range(d_model)]

        # Per-layer CSVs (one row per prompt)
        for layer_idx in range(n_layers):
            streams = np.stack([r["final_token_streams"][layer_idx] for r in results])
            df = pd.DataFrame(streams, columns=dim_columns)
            df.insert(0, "prompt_idx", range(len(results)))
            df.insert(1, "prompt", [r["prompt"] for r in results])
            df.insert(2, "predicted_next_token", [r["predicted_token"] for r in results])
            df.insert(3, "target_token", [group.target_token] * len(results))
            df.insert(4, "top3_predictions", [
                " | ".join([f"{t}({l:.1f})" for t, l in r["top_k_predictions"][:3]])
                for r in results
            ])
            df.to_csv(final_dir / f"layer_{layer_idx:03d}.csv", index=False, encoding="utf-8")

        # Combined CSV (all layers, all prompts) with next token
        all_rows = []
        for layer_idx in range(n_layers):
            for prompt_idx, result in enumerate(results):
                row = {
                    "layer": layer_idx,
                    "prompt_idx": prompt_idx,
                    "prompt": result["prompt"],
                    "predicted_next_token": result["predicted_token"],
                    "target_token": group.target_token,
                    "top3": " | ".join([f"{t}({l:.1f})" for t, l in result["top_k_predictions"][:3]]),
                }
                for d in range(d_model):
                    row[f"dim_{d:04d}"] = result["final_token_streams"][layer_idx][d]
                all_rows.append(row)

        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(final_dir / "all_layers_all_prompts.csv", index=False, encoding="utf-8")

        # --- 3. Centroids ---
        centroid_dir = group_dir / "centroids"
        centroid_dir.mkdir(exist_ok=True)

        centroid_rows = []
        for layer_idx in range(n_layers):
            streams = np.stack([r["final_token_streams"][layer_idx] for r in results])
            centroid = streams.mean(axis=0)
            row = {"layer": layer_idx}
            for d in range(d_model):
                row[f"dim_{d:04d}"] = centroid[d]
            centroid_rows.append(row)

        df_centroids = pd.DataFrame(centroid_rows)
        df_centroids.to_csv(centroid_dir / "centroids_all_layers.csv", index=False, encoding="utf-8")

        # --- 4. Prompts metadata with next token ---
        meta_rows = []
        for i, result in enumerate(results):
            meta_rows.append({
                "prompt_idx": i,
                "prompt": result["prompt"],
                "target_token": group.target_token,
                "predicted_next_token": result["predicted_token"],
                "target_match": group.target_token.lower() in result["predicted_token"].lower() or result["predicted_token"].lower() in group.target_token.lower(),
                "n_tokens": len(result["tokens"]),
                "tokens": " | ".join(result["tokens"]),
                "top1_logit": result["top_k_predictions"][0][1],
                "top3": str([(t, round(l, 2)) for t, l in result["top_k_predictions"][:3]]),
                "top10": str([(t, round(l, 2)) for t, l in result["top_k_predictions"][:10]]),
            })

        df_meta = pd.DataFrame(meta_rows)
        df_meta.to_csv(group_dir / "prompts_meta.csv", index=False, encoding="utf-8")

        print(f"    Saved: {group_dir}/")


# =============================================================================
# Attractor Metrics
# =============================================================================

def compute_group_metrics(results: list[dict], group_name: str) -> dict:
    """Compute convergence metrics for a group."""
    n_prompts = len(results)
    n_layers = results[0]["n_layers"]

    metrics = {
        "group_name": group_name,
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "per_layer": {},
    }

    for layer_idx in range(n_layers):
        streams = np.stack([r["final_token_streams"][layer_idx] for r in results])
        centroid = streams.mean(axis=0)
        diffs = streams - centroid[None, :]
        distances = np.linalg.norm(diffs, axis=1)

        norms = np.linalg.norm(streams, axis=1, keepdims=True)
        normalized = streams / (norms + 1e-10)
        cos_sim_matrix = normalized @ normalized.T
        triu_indices = np.triu_indices(n_prompts, k=1)
        pairwise_cos_sims = cos_sim_matrix[triu_indices]

        metrics["per_layer"][layer_idx] = {
            "mean_distance": float(distances.mean()),
            "std_distance": float(distances.std()),
            "max_distance": float(distances.max()),
            "min_distance": float(distances.min()),
            "mean_cosine_sim": float(pairwise_cos_sims.mean()),
            "std_cosine_sim": float(pairwise_cos_sims.std()),
            "min_cosine_sim": float(pairwise_cos_sims.min()),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "individual_distances": distances.tolist(),
        }

    mean_distances = [metrics["per_layer"][l]["mean_distance"] for l in range(n_layers)]
    mean_cosines = [metrics["per_layer"][l]["mean_cosine_sim"] for l in range(n_layers)]

    metrics["convergence"] = {
        "distance_trajectory": mean_distances,
        "cosine_trajectory": mean_cosines,
        "distance_ratio_last_vs_first": mean_distances[-1] / (mean_distances[0] + 1e-10),
        "cosine_improvement": mean_cosines[-1] - mean_cosines[0],
    }

    return metrics


# =============================================================================
# Visualization (Enhanced with animations and geometry plots)
# =============================================================================

class AttractorVisualizer:
    """Generates plots and animations for attractor analysis."""

    def __init__(self, output_dir: Path):
        self.viz_dir = output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def plot_convergence_trajectories(self, all_metrics: dict[str, dict]):
        """Plot distance and cosine trajectories across layers for all groups."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        for group_name, metrics in all_metrics.items():
            n_layers = metrics["n_layers"]
            layers = list(range(n_layers))

            axes[0].plot(layers, metrics["convergence"]["distance_trajectory"],
                        label=group_name, linewidth=1.5)
            axes[1].plot(layers, metrics["convergence"]["cosine_trajectory"],
                        label=group_name, linewidth=1.5)

        axes[0].set_ylabel("Mean Distance to Centroid")
        axes[0].set_title("Convergence: Distance to Group Centroid per Layer")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Layer Index")
        axes[1].set_ylabel("Mean Pairwise Cosine Similarity")
        axes[1].set_title("Convergence: Pairwise Cosine Similarity per Layer")
        axes[1].legend(loc="lower right", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "convergence_trajectories.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {self.viz_dir}/convergence_trajectories.png")

    def plot_cosine_heatmaps(self, all_results: dict[str, list[dict]], layer_indices: list[int] = None):
        """Plot cosine similarity heatmaps at selected layers for each group."""
        group_names = list(all_results.keys())
        n_groups = len(group_names)

        if layer_indices is None:
            n_layers = all_results[group_names[0]][0]["n_layers"]
            layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

        n_layer_plots = len(layer_indices)

        fig, axes = plt.subplots(n_groups, n_layer_plots, figsize=(4 * n_layer_plots, 3.5 * n_groups))
        if n_groups == 1:
            axes = axes[None, :]
        if n_layer_plots == 1:
            axes = axes[:, None]

        for row, group_name in enumerate(group_names):
            results = all_results[group_name]
            n_prompts = len(results)

            for col, layer_idx in enumerate(layer_indices):
                streams = np.stack([r["final_token_streams"][layer_idx] for r in results])
                norms = np.linalg.norm(streams, axis=1, keepdims=True)
                normalized = streams / (norms + 1e-10)
                cos_sim = normalized @ normalized.T

                im = axes[row, col].imshow(cos_sim, vmin=-1, vmax=1, cmap="RdBu_r")
                axes[row, col].set_title(f"L{layer_idx}", fontsize=9)

                if col == 0:
                    axes[row, col].set_ylabel(group_name, fontsize=8)

                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

        fig.colorbar(im, ax=axes, shrink=0.6, label="Cosine Similarity")
        fig.suptitle("Pairwise Cosine Similarity Heatmaps (within group, across layers)", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.viz_dir / "cosine_similarity_heatmaps.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {self.viz_dir}/cosine_similarity_heatmaps.png")

    def plot_pca_final_layer(self, all_results: dict[str, list[dict]]):
        """PCA projection of final-layer residual streams, colored by group, with next-token labels."""
        group_names = list(all_results.keys())
        n_layers = all_results[group_names[0]][0]["n_layers"]
        last_layer = n_layers - 1

        all_streams = []
        labels = []
        predicted_tokens = []
        for group_name in group_names:
            for result in all_results[group_name]:
                all_streams.append(result["final_token_streams"][last_layer])
                labels.append(group_name)
                predicted_tokens.append(result["predicted_token"])

        X = np.stack(all_streams)
        X_centered = X - X.mean(axis=0)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_centered)

        # 2D plot with token labels
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))

        offset = 0
        for i, group_name in enumerate(group_names):
            n = len(all_results[group_name])
            ax.scatter(X_pca[offset:offset+n, 0], X_pca[offset:offset+n, 1],
                      c=[colors[i]], label=group_name, s=60, alpha=0.8, edgecolors="k", linewidths=0.5)

            # Annotate with predicted next token
            for j in range(n):
                ax.annotate(predicted_tokens[offset + j],
                           (X_pca[offset + j, 0], X_pca[offset + j, 1]),
                           fontsize=6, alpha=0.7, ha="center", va="bottom",
                           xytext=(0, 5), textcoords="offset points")

            # Draw centroid
            centroid = X_pca[offset:offset+n].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c=[colors[i]], s=200, marker="*",
                      edgecolors="k", linewidths=1.5, zorder=10)

            offset += n

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        ax.set_title(f"PCA of Final-Layer (L{last_layer}) Residual Streams\n(labels = predicted next token, stars = centroids)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "pca_final_layer.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {self.viz_dir}/pca_final_layer.png")

    def plot_pca_3d_final_layer(self, all_results: dict[str, list[dict]]):
        """3D PCA projection for geometry inspection."""
        group_names = list(all_results.keys())
        n_layers = all_results[group_names[0]][0]["n_layers"]
        last_layer = n_layers - 1

        all_streams = []
        labels = []
        for group_name in group_names:
            for result in all_results[group_name]:
                all_streams.append(result["final_token_streams"][last_layer])
                labels.append(group_name)

        X = np.stack(all_streams)
        X_centered = X - X.mean(axis=0)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_centered)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))

        offset = 0
        for i, group_name in enumerate(group_names):
            n = len(all_results[group_name])
            ax.scatter(X_pca[offset:offset+n, 0], X_pca[offset:offset+n, 1], X_pca[offset:offset+n, 2],
                      c=[colors[i]], label=group_name, s=60, alpha=0.8, edgecolors="k", linewidths=0.3)
            centroid = X_pca[offset:offset+n].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], centroid[2], c=[colors[i]], s=200, marker="*",
                      edgecolors="k", linewidths=1.5)
            offset += n

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
        ax.set_title("3D PCA of Final-Layer Residual Streams")
        ax.legend(loc="best", fontsize=7)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "pca_3d_final_layer.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {self.viz_dir}/pca_3d_final_layer.png")

    def plot_cross_group_distances(self, all_results: dict[str, list[dict]]):
        """Bar chart of between-group vs within-group distances at final layer."""
        group_names = list(all_results.keys())
        n_layers = all_results[group_names[0]][0]["n_layers"]
        last_layer = n_layers - 1

        within_distances = {}
        centroids = {}
        for name in group_names:
            streams = np.stack([r["final_token_streams"][last_layer] for r in all_results[name]])
            centroid = streams.mean(axis=0)
            centroids[name] = centroid
            dists = np.linalg.norm(streams - centroid[None, :], axis=1)
            within_distances[name] = float(dists.mean())

        between_distances = {}
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                key = f"{group_names[i]}\nvs\n{group_names[j]}"
                d = np.linalg.norm(centroids[group_names[i]] - centroids[group_names[j]])
                between_distances[key] = float(d)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].bar(range(len(within_distances)), list(within_distances.values()), color="steelblue")
        axes[0].set_xticks(range(len(within_distances)))
        axes[0].set_xticklabels(list(within_distances.keys()), rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Mean L2 Distance to Centroid")
        axes[0].set_title("Within-Group Distances (should be SMALL)")
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].bar(range(len(between_distances)), list(between_distances.values()), color="coral")
        axes[1].set_xticks(range(len(between_distances)))
        axes[1].set_xticklabels(list(between_distances.keys()), rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("L2 Distance Between Centroids")
        axes[1].set_title("Between-Group Distances (should be LARGE)")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "cross_group_distances.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {self.viz_dir}/cross_group_distances.png")

    def plot_layer_trajectory_pca(self, all_results: dict[str, list[dict]], selected_groups: list[str] = None):
        """Show how points move through PCA space from early to late layers."""
        group_names = selected_groups or list(all_results.keys())[:4]
        n_layers = all_results[group_names[0]][0]["n_layers"]

        all_streams_final = []
        for name in group_names:
            for result in all_results[name]:
                all_streams_final.append(result["final_token_streams"][n_layers - 1])

        X_final = np.stack(all_streams_final)
        mean_vec = X_final.mean(axis=0)

        pca = PCA(n_components=2)
        pca.fit(X_final - mean_vec)

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))

        layer_samples = np.linspace(0, n_layers - 1, min(20, n_layers), dtype=int)

        for g_idx, name in enumerate(group_names):
            results = all_results[name]
            for p_idx, result in enumerate(results):
                trajectory = []
                for l in layer_samples:
                    vec = result["final_token_streams"][l]
                    projected = pca.transform((vec - mean_vec).reshape(1, -1))[0]
                    trajectory.append(projected)

                trajectory = np.array(trajectory)

                ax.plot(trajectory[:, 0], trajectory[:, 1],
                       color=colors[g_idx], alpha=0.3, linewidth=0.8)
                ax.scatter(trajectory[0, 0], trajectory[0, 1],
                          color=colors[g_idx], s=15, alpha=0.4, marker="o")
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                          color=colors[g_idx], s=50, alpha=0.8, marker="o",
                          edgecolors="k", linewidths=0.5)

            final_streams = np.stack([r["final_token_streams"][n_layers - 1] for r in results])
            centroid = final_streams.mean(axis=0)
            centroid_proj = pca.transform((centroid - mean_vec).reshape(1, -1))[0]
            ax.scatter(centroid_proj[0], centroid_proj[1], color=colors[g_idx],
                      s=300, marker="*", edgecolors="k", linewidths=1.5, zorder=10,
                      label=f"{name}")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Layer Trajectories in PCA Space\n(small dots = early layers, large dots = final layer, stars = centroids)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "layer_trajectories_pca.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {self.viz_dir}/layer_trajectories_pca.png")

    def create_convergence_animation(self, all_results: dict[str, list[dict]], selected_groups: list[str] = None):
        """Create animated GIF showing points converging to attractors across layers."""
        group_names = selected_groups or list(all_results.keys())[:5]
        n_layers = all_results[group_names[0]][0]["n_layers"]

        # Collect all final-layer streams for PCA basis
        all_streams_final = []
        for name in group_names:
            for result in all_results[name]:
                all_streams_final.append(result["final_token_streams"][n_layers - 1])

        X_final = np.stack(all_streams_final)
        mean_vec = X_final.mean(axis=0)

        pca = PCA(n_components=2)
        pca.fit(X_final - mean_vec)

        # Pre-compute all projections for all layers
        all_projections = {}  # group_name -> (n_prompts, n_layers, 2)
        for name in group_names:
            results = all_results[name]
            group_proj = np.zeros((len(results), n_layers, 2))
            for p_idx, result in enumerate(results):
                for l in range(n_layers):
                    vec = result["final_token_streams"][l]
                    group_proj[p_idx, l] = pca.transform((vec - mean_vec).reshape(1, -1))[0]
            all_projections[name] = group_proj

        # Compute axis limits
        all_points = np.concatenate([proj.reshape(-1, 2) for proj in all_projections.values()])
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        margin = 0.1 * max(x_max - x_min, y_max - y_min)

        # Create animation
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))

        def animate(frame):
            ax.clear()
            layer_idx = frame

            for g_idx, name in enumerate(group_names):
                proj = all_projections[name]  # (n_prompts, n_layers, 2)
                points = proj[:, layer_idx, :]  # (n_prompts, 2)

                ax.scatter(points[:, 0], points[:, 1],
                          c=[colors[g_idx]], s=60, alpha=0.8,
                          edgecolors="k", linewidths=0.5, label=name)

                # Draw centroid
                centroid = points.mean(axis=0)
                ax.scatter(centroid[0], centroid[1], c=[colors[g_idx]], s=200, marker="*",
                          edgecolors="k", linewidths=1.5, zorder=10)

                # Draw trails (last 5 layers)
                if layer_idx > 0:
                    trail_start = max(0, layer_idx - 5)
                    for p_idx in range(len(proj)):
                        trail = proj[p_idx, trail_start:layer_idx+1, :]
                        ax.plot(trail[:, 0], trail[:, 1],
                               color=colors[g_idx], alpha=0.2, linewidth=0.8)

            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"Residual Stream Dynamics — Layer {layer_idx}/{n_layers-1}\n"
                        f"(Points converging to attractors)")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.3)

            # Progress bar in plot
            progress = layer_idx / (n_layers - 1)
            ax.axhline(y=y_min - margin * 0.5, xmin=0, xmax=progress,
                      color="green", linewidth=3, alpha=0.7)

        # Use every layer or subsample if too many
        if n_layers > 60:
            frame_indices = np.linspace(0, n_layers - 1, 60, dtype=int)
        else:
            frame_indices = list(range(n_layers))

        anim = FuncAnimation(fig, animate, frames=frame_indices, interval=200, blit=False)

        gif_path = self.viz_dir / "convergence_animation.gif"
        anim.save(str(gif_path), writer=PillowWriter(fps=5))
        plt.close()
        print(f"    Saved: {gif_path}")

    def plot_attractor_geometry(self, geometry_results: list[dict]):
        """Plot attractor geometry analysis results."""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig)

        # --- Panel 1: Classification summary ---
        ax1 = fig.add_subplot(gs[0, 0])
        names = [g["group_name"] for g in geometry_results]
        classifications = [g["classification"] for g in geometry_results]
        confidences = [g.get("confidence", 0) for g in geometry_results]

        color_map = {"point": "blue", "ring": "orange", "torus": "red",
                    "cloud": "gray", "low_dim_manifold": "purple", "insufficient_data": "white"}
        bar_colors = [color_map.get(c, "gray") for c in classifications]

        bars = ax1.barh(range(len(names)), confidences, color=bar_colors, edgecolor="k")
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=7)
        ax1.set_xlabel("Confidence")
        ax1.set_title("Attractor Geometry Classification")

        # Add classification text
        for i, (cls, conf) in enumerate(zip(classifications, confidences)):
            ax1.text(conf + 0.02, i, cls, va="center", fontsize=7)

        ax1.set_xlim(0, 1.3)

        # --- Panel 2: Eigenvalue spectra ---
        ax2 = fig.add_subplot(gs[0, 1])
        for g in geometry_results:
            if "explained_variance_ratios" in g and g["explained_variance_ratios"]:
                evr = g["explained_variance_ratios"]
                ax2.plot(range(1, len(evr) + 1), evr, marker="o", markersize=4,
                        label=g["group_name"], linewidth=1.5)

        ax2.set_xlabel("Principal Component")
        ax2.set_ylabel("Explained Variance Ratio")
        ax2.set_title("PCA Eigenvalue Spectra")
        ax2.legend(fontsize=6, loc="upper right")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        # --- Panel 3: Ring scores vs Torus scores ---
        ax3 = fig.add_subplot(gs[0, 2])
        ring_scores = [g.get("ring_score", 0) for g in geometry_results]
        torus_scores = [g.get("torus_score", 0) for g in geometry_results]

        ax3.scatter(ring_scores, torus_scores, s=80, c=bar_colors, edgecolors="k", zorder=5)
        for i, name in enumerate(names):
            ax3.annotate(name, (ring_scores[i], torus_scores[i]),
                        fontsize=6, ha="center", va="bottom", xytext=(0, 5),
                        textcoords="offset points")

        ax3.set_xlabel("Ring Score")
        ax3.set_ylabel("Torus Score")
        ax3.set_title("Ring vs Torus Geometry Scores")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.1, 1.1)
        ax3.set_ylim(-0.1, 1.1)

        # --- Panel 4-6: 2D PCA projections for individual groups ---
        interesting_groups = [g for g in geometry_results
                            if g.get("pca_projected_2d") is not None and g["classification"] != "point"]
        # Also include some point groups for comparison
        point_groups = [g for g in geometry_results
                       if g.get("pca_projected_2d") is not None and g["classification"] == "point"]

        plot_groups = interesting_groups[:2] + point_groups[:1]
        if not plot_groups:
            plot_groups = [g for g in geometry_results if g.get("pca_projected_2d") is not None][:3]

        for idx, g in enumerate(plot_groups[:3]):
            ax = fig.add_subplot(gs[1, idx])
            proj = np.array(g["pca_projected_2d"])
            ax.scatter(proj[:, 0], proj[:, 1], s=50, alpha=0.8, edgecolors="k", linewidths=0.5)

            # Draw circle for ring reference
            if g["classification"] in ("ring", "torus"):
                theta = np.linspace(0, 2 * np.pi, 100)
                r = np.linalg.norm(proj, axis=1).mean()
                ax.plot(r * np.cos(theta), r * np.sin(theta), "r--", alpha=0.4, linewidth=1)

            ax.set_title(f"{g['group_name']}\n[{g['classification']}] (conf={g.get('confidence', 0):.2f})",
                        fontsize=9)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "attractor_geometry_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {self.viz_dir}/attractor_geometry_analysis.png")

    def plot_per_group_geometry_detail(self, all_results: dict[str, list[dict]], geometry_results: list[dict]):
        """Detailed per-group geometry plots showing 2D and 3D structure."""
        n_layers = list(all_results.values())[0][0]["n_layers"]
        last_layer = n_layers - 1

        for g_result in geometry_results:
            name = g_result["group_name"]
            if name not in all_results:
                continue

            results = all_results[name]
            streams = np.stack([r["final_token_streams"][last_layer] for r in results])
            predicted_tokens = [r["predicted_token"] for r in results]

            if streams.shape[0] < 3:
                continue

            centered = streams - streams.mean(axis=0)
            n_comp = min(3, streams.shape[0] - 1)
            pca = PCA(n_components=n_comp)
            projected = pca.fit_transform(centered)

            fig = plt.figure(figsize=(16, 5))

            # 2D scatter with token labels
            ax1 = fig.add_subplot(131)
            ax1.scatter(projected[:, 0], projected[:, 1], s=60, alpha=0.8, edgecolors="k", linewidths=0.5)
            for i, tok in enumerate(predicted_tokens):
                ax1.annotate(tok, (projected[i, 0], projected[i, 1]),
                            fontsize=7, ha="center", va="bottom", xytext=(0, 4),
                            textcoords="offset points")
            ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax1.set_title(f"{name} — 2D PCA\n[{g_result['classification']}]")
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect("equal")

            # 3D scatter if possible
            if n_comp >= 3:
                ax2 = fig.add_subplot(132, projection='3d')
                ax2.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                           s=60, alpha=0.8, edgecolors="k", linewidths=0.3)
                ax2.set_xlabel(f"PC1")
                ax2.set_ylabel(f"PC2")
                ax2.set_zlabel(f"PC3")
                ax2.set_title(f"{name} — 3D PCA")
            else:
                ax2 = fig.add_subplot(132)
                ax2.text(0.5, 0.5, "Not enough\nsamples for 3D", ha="center", va="center", fontsize=12)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)

            # Distance from centroid histogram
            ax3 = fig.add_subplot(133)
            dists = np.linalg.norm(centered, axis=1)
            ax3.hist(dists, bins=max(5, len(dists) // 2), color="steelblue", edgecolor="k", alpha=0.7)
            ax3.axvline(dists.mean(), color="red", linestyle="--", label=f"mean={dists.mean():.2f}")
            ax3.set_xlabel("Distance from Centroid")
            ax3.set_ylabel("Count")
            ax3.set_title(f"Distance Distribution\nspread={g_result.get('relative_spread', 0):.4f}")
            ax3.legend()

            plt.tight_layout()
            plt.savefig(self.viz_dir / f"geometry_detail_{name}.png", dpi=150, bbox_inches="tight")
            plt.close()

        print(f"    Saved: {self.viz_dir}/geometry_detail_*.png")

    def generate_all_plots(self, all_results: dict[str, list[dict]], all_metrics: dict[str, dict],
                          geometry_results: list[dict] = None):
        """Generate all visualization plots."""
        print("\n  Generating visualizations...")
        self.plot_convergence_trajectories(all_metrics)
        self.plot_cosine_heatmaps(all_results)
        self.plot_pca_final_layer(all_results)
        self.plot_pca_3d_final_layer(all_results)
        self.plot_cross_group_distances(all_results)
        self.plot_layer_trajectory_pca(all_results)
        self.create_convergence_animation(all_results)

        if geometry_results:
            self.plot_attractor_geometry(geometry_results)
            self.plot_per_group_geometry_detail(all_results, geometry_results)

        print("  All visualizations complete.")


# =============================================================================
# Residual Stream Extractor
# =============================================================================

class ResidualStreamExtractor:
    def __init__(self, model, tokenizer, device: str = "cuda", max_tokens: Optional[int] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.hooks = []
        self.residual_streams = {}

    def _get_layer_modules(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            return self.model.gpt_neox.layers
        else:
            raise ValueError(
                f"Cannot find transformer layers in model of type {type(self.model)}. "
                f"Top-level attributes: {[a for a in dir(self.model) if not a.startswith('_')]}"
            )

    def _register_hooks(self):
        self.residual_streams = {}
        self.hooks = []
        layers = self._get_layer_modules()

        for layer_idx, layer in enumerate(layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    self.residual_streams[idx] = hidden.detach().cpu()
                return hook_fn
            h = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def extract(self, prompt: str) -> dict:
        """Run prompt, capture full residual streams at all layers and positions."""
        self._register_hooks()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # --- Truncate to max_tokens if specified ---
            if self.max_tokens is not None:
                input_ids = inputs["input_ids"][:, :self.max_tokens]
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :self.max_tokens]
                inputs = {"input_ids": input_ids}
                if attention_mask is not None:
                    inputs["attention_mask"] = attention_mask

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"][0].tolist()
            tokens = [self.tokenizer.decode([tid]) for tid in input_ids]

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits[0, -1, :]
            top_k = torch.topk(logits, k=10)
            top_k_predictions = [
                (self.tokenizer.decode([idx.item()]).strip(), float(logits[idx].item()))
                for idx in top_k.indices
            ]
            predicted_token = top_k_predictions[0][0]

            # Full streams: dict[layer_idx] -> np.ndarray (seq_len, d_model)
            full_streams = {}
            final_token_streams = {}
            for layer_idx, stream in self.residual_streams.items():
                full_streams[layer_idx] = stream[0].float().numpy()
                final_token_streams[layer_idx] = stream[0, -1, :].float().numpy()

            return {
                "prompt": prompt,
                "input_ids": input_ids,
                "tokens": tokens,
                "n_layers": len(self.residual_streams),
                "full_streams": full_streams,
                "final_token_streams": final_token_streams,
                "predicted_token": predicted_token,
                "top_k_predictions": top_k_predictions,
            }
        finally:
            self._remove_hooks()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Residual Stream Attractor Extraction v2")

    MODEL_PRESETS = {
        "deepseek": "deepseek-ai/deepseek-llm-7b-base",
        "gpt2": "gpt2",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
        "gpt2-xl": "gpt2-xl",
        "distilgpt2": "distilgpt2",
        "pythia-70m": "EleutherAI/pythia-70m",
        "pythia-160m": "EleutherAI/pythia-160m",
        "pythia-410m": "EleutherAI/pythia-410m",
        "pythia-1b": "EleutherAI/pythia-1b",
    }

    parser.add_argument(
        "--model-preset", type=str, default="deepseek",
        choices=list(MODEL_PRESETS.keys()),
        help=f"Model preset to use (default: deepseek). Available: {list(MODEL_PRESETS.keys())}",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model ID (overrides --model-preset if given)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--output", type=str, default="attractor_data_v2",
        help="Output directory for saved data (default: attractor_data_v2)",
    )
    parser.add_argument(
        "--groups", type=str, nargs="*", default=None,
        help="Which groups to run (by name). Default: all groups.",
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: float16)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Maximum number of tokens to process per prompt. Truncates input if exceeded.",
    )

    args = parser.parse_args()

    # --- Resolve model ID ---
    if args.model is not None:
        model_id = args.model
    else:
        model_id = MODEL_PRESETS[args.model_preset]

    # --- Device ---
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cpu" and args.dtype == "float16":
        print(f"[INFO] Switching dtype from float16 to float32 for CPU execution.")
        args.dtype = "float32"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"{'='*60}")
    print(f"Residual Stream Attractor Extraction Tool v2")
    print(f"{'='*60}")
    print(f"Model preset: {args.model_preset}")
    print(f"Model ID: {model_id}")
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Max tokens: {args.max_tokens or 'unlimited'}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")

    # Load model and tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device == "auto" else {"": device},
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup extractor
    extractor = ResidualStreamExtractor(model, tokenizer, device, max_tokens=args.max_tokens)
    layers = extractor._get_layer_modules()
    n_layers = len(layers)
    print(f"Transformer layers: {n_layers}")

    # Select groups
    if args.groups:
        groups = [g for g in ATTRACTOR_GROUPS if g.name in args.groups]
        if not groups:
            print(f"ERROR: No matching groups. Available: {[g.name for g in ATTRACTOR_GROUPS]}")
            sys.exit(1)
    else:
        groups = ATTRACTOR_GROUPS

    print(f"\nGroups: {[g.name for g in groups]}")
    print(f"Total prompts: {sum(len(g.prompts) for g in groups)}")

    # Process all groups
    output_path = Path(args.output)
    saver = RawDataSaver(output_path)
    visualizer = AttractorVisualizer(output_path)
    geometry_analyzer = AttractorGeometryAnalyzer(n_components=10)

    all_results = {}
    all_metrics = {}
    geometry_results = []

    for group in groups:
        print(f"\n{'='*60}")
        print(f"Processing: {group.name} (target: '{group.target_token}', expected geometry: {group.expected_geometry})")
        print(f"{'='*60}")

        results = []
        for prompt in tqdm(group.prompts, desc=group.name):
            result = extractor.extract(prompt)
            result["target_token"] = group.target_token
            result["group_name"] = group.name

            predicted = result["predicted_token"]
            target_match = (
                group.target_token.lower() in predicted.lower() or
                predicted.lower() in group.target_token.lower()
            )
            result["target_match"] = target_match

            match_str = "\u2713" if target_match else "\u2717"
            top3_str = " | ".join([f"{t}({l:.1f})" for t, l in result["top_k_predictions"][:3]])
            print(f"  [{match_str}] '{prompt}' -> '{predicted}' [top3: {top3_str}]")
            results.append(result)

        all_results[group.name] = results

        # Compute metrics
        metrics = compute_group_metrics(results, group.name)
        all_metrics[group.name] = metrics

        conv = metrics["convergence"]
        print(f"\n  Convergence metrics:")
        print(f"    Distance ratio (last/first): {conv['distance_ratio_last_vs_first']:.4f}")
        print(f"    Cosine improvement: {conv['cosine_improvement']:.4f}")
        print(f"    Final layer cosine sim: {conv['cosine_trajectory'][-1]:.4f}")

        # Geometry analysis on final layer
        last_layer = metrics["n_layers"] - 1
        final_streams = np.stack([r["final_token_streams"][last_layer] for r in results])
        geo_result = geometry_analyzer.analyze(final_streams, group_name=group.name)
        geo_result["expected_geometry"] = group.expected_geometry
        geometry_results.append(geo_result)

        print(f"\n  Geometry analysis:")
        print(f"    Classification: {geo_result['classification']} (expected: {group.expected_geometry})")
        print(f"    Confidence: {geo_result.get('confidence', 0):.3f}")
        print(f"    Relative spread: {geo_result.get('relative_spread', 0):.6f}")
        print(f"    Ring score: {geo_result.get('ring_score', 0):.3f}")
        print(f"    Torus score: {geo_result.get('torus_score', 0):.3f}")
        print(f"    Dims for 90% variance: {geo_result.get('dims_for_90_pct', '?')}")

        # Save raw data as CSV (UTF-8)
        saver.save_group(group, results)

        # Save metrics JSON
        group_dir = output_path / group.name
        metrics_save = {
            "group_name": metrics["group_name"],
            "n_prompts": metrics["n_prompts"],
            "n_layers": metrics["n_layers"],
            "convergence": metrics["convergence"],
            "geometry": {k: v for k, v in geo_result.items() if k != "pca_projected_2d" and k != "pca_projected_3d"},
            "per_layer": {
                str(l): {k: v for k, v in data.items()}
                for l, data in metrics["per_layer"].items()
            },
        }
        with open(group_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_save, f, indent=2, ensure_ascii=False)

    # Cross-group comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"Cross-Group Comparison (final layer)")
        print(f"{'='*60}")

        group_names = list(all_results.keys())
        last_layer = all_results[group_names[0]][0]["n_layers"] - 1

        centroids = {}
        for name in group_names:
            streams = np.stack([r["final_token_streams"][last_layer] for r in all_results[name]])
            centroids[name] = streams.mean(axis=0)

        comparison = {"layer": last_layer, "between_distances": {}, "between_cosines": {}}
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                key = f"{group_names[i]} vs {group_names[j]}"
                d = float(np.linalg.norm(centroids[group_names[i]] - centroids[group_names[j]]))
                cos = float(
                    np.dot(centroids[group_names[i]], centroids[group_names[j]]) /
                    (np.linalg.norm(centroids[group_names[i]]) * np.linalg.norm(centroids[group_names[j]]) + 1e-10)
                )
                comparison["between_distances"][key] = d
                comparison["between_cosines"][key] = cos
                print(f"  {key}: dist={d:.4f}, cos={cos:.4f}")

        with open(output_path / "cross_group_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Save geometry summary
    geometry_summary = []
    for g in geometry_results:
        summary_entry = {k: v for k, v in g.items() if k not in ("pca_projected_2d", "pca_projected_3d")}
        geometry_summary.append(summary_entry)

    with open(output_path / "geometry_summary.json", "w", encoding="utf-8") as f:
        json.dump(geometry_summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Geometry Summary:")
    print(f"  {'Group':<30} {'Expected':<10} {'Classified':<15} {'Confidence':<10} {'Ring':<6} {'Torus':<6}")
    print(f"  {'-'*77}")
    for g in geometry_results:
        print(f"  {g['group_name']:<30} {g.get('expected_geometry','?'):<10} {g['classification']:<15} "
              f"{g.get('confidence',0):<10.3f} {g.get('ring_score',0):<6.3f} {g.get('torus_score',0):<6.3f}")

    # Generate visualizations (including geometry and animation)
    visualizer.generate_all_plots(all_results, all_metrics, geometry_results)

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}/")
    print(f"")
    print(f"Structure:")
    print(f"  {{group}}/raw_streams/layer_XXX/prompt_YYY.csv  <- full residual streams (UTF-8)")
    print(f"  {{group}}/final_token_streams/layer_XXX.csv     <- final-pos streams + next token")
    print(f"  {{group}}/final_token_streams/all_layers_all_prompts.csv")
    print(f"  {{group}}/centroids/centroids_all_layers.csv")
    print(f"  {{group}}/prompts_meta.csv                      <- includes predicted next token + top10")
    print(f"  {{group}}/metrics.json                          <- includes geometry classification")
    print(f"  visualizations/convergence_trajectories.png")
    print(f"  visualizations/cosine_similarity_heatmaps.png")
    print(f"  visualizations/pca_final_layer.png              <- with next-token labels")
    print(f"  visualizations/pca_3d_final_layer.png")
    print(f"  visualizations/cross_group_distances.png")
    print(f"  visualizations/layer_trajectories_pca.png")
    print(f"  visualizations/convergence_animation.gif        <- animated attractor convergence")
    print(f"  visualizations/attractor_geometry_analysis.png  <- geometry classification")
    print(f"  visualizations/geometry_detail_{{group}}.png     <- per-group geometry detail")
    print(f"  geometry_summary.json")
    print(f"  cross_group_comparison.json")


if __name__ == "__main__":
    main()
