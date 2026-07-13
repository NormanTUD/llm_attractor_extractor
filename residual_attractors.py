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
# ]
# ///
"""
Residual Stream Attractor Extraction Tool

Loads DeepSeek (or compatible model) and runs predefined prompt groups
through it, capturing the FULL residual stream at every layer for every
token position. Saves raw data as CSV files organized by group and layer.

Hypothesis: In later layers, residual stream vectors for prompts that
should predict the same token (e.g., "Berlin") are attracted toward
a common point (attractor) regardless of the input language or framing.

Output structure:
    output_dir/
    +-- {group_name}/
    |   +-- raw_streams/
    |   |   +-- layer_000/
    |   |   |   +-- prompt_000.csv    # columns: dim_0, dim_1, ..., dim_N; rows: token positions
    |   |   |   +-- prompt_001.csv
    |   |   |   +-- ...
    |   |   +-- layer_001/
    |   |   +-- ...
    |   +-- final_token_streams/
    |   |   +-- layer_000.csv          # columns: dim_0..dim_N; rows: one per prompt
    |   |   +-- layer_001.csv
    |   |   +-- ...
    |   |   +-- all_layers.csv         # columns: layer, prompt_idx, dim_0..dim_N
    |   +-- centroids/
    |   |   +-- centroids_all_layers.csv  # columns: layer, dim_0..dim_N
    |   +-- metrics.json
    |   +-- prompts_meta.csv
    +-- visualizations/
    |   +-- convergence_trajectories.png
    |   +-- cosine_similarity_heatmaps.png
    |   +-- cross_group_distances.png
    |   +-- pca_final_layer.png
    +-- cross_group_comparison.json

Usage:
    uv run residual_attractors.py [--model deepseek-ai/deepseek-llm-7b-base] [--device cuda]
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
matplotlib.use("Agg")  # Non-interactive backend for cluster/headless
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    prompts: list[str] = field(default_factory=list)


ATTRACTOR_GROUPS: list[PromptGroup] = [
    PromptGroup(
        name="paris_multilingual",
        target_token="Paris",
        attractor_concept="Paris (capital of France)",
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
        prompts=[
            "E = mc\u00b2 was discovered by Albert",
            "The theory of relativity was developed by",
            "The most famous physicist of the 20th century is Albert",
            "Die Relativit\u00e4tstheorie wurde entwickelt von Albert",
            "The Nobel Prize in Physics 1921 was awarded to Albert",
        ],
    ),
]


# =============================================================================
# Residual Stream Extraction
# =============================================================================

class ResidualStreamExtractor:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
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
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
# CSV Data Saver
# =============================================================================

class RawDataSaver:
    """Saves all residual stream data as CSV files in a structured directory layout."""

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
                df.insert(1, "token", result["tokens"][:len(df)] if len(result["tokens"]) >= len(df) else result["tokens"] + [""] * (len(df) - len(result["tokens"])))

                df.to_csv(layer_dir / f"prompt_{prompt_idx:03d}.csv", index=False)

        # --- 2. Final token streams: group/final_token_streams/layer_XXX.csv ---
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
            df.to_csv(final_dir / f"layer_{layer_idx:03d}.csv", index=False)

        # Combined CSV (all layers, all prompts)
        all_rows = []
        for layer_idx in range(n_layers):
            for prompt_idx, result in enumerate(results):
                row = {
                    "layer": layer_idx,
                    "prompt_idx": prompt_idx,
                    "prompt": result["prompt"],
                    "predicted_token": result["predicted_token"],
                }
                for d in range(d_model):
                    row[f"dim_{d:04d}"] = result["final_token_streams"][layer_idx][d]
                all_rows.append(row)

        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(final_dir / "all_layers_all_prompts.csv", index=False)

        # --- 3. Centroids: group/centroids/centroids_all_layers.csv ---
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
        df_centroids.to_csv(centroid_dir / "centroids_all_layers.csv", index=False)

        # --- 4. Prompts metadata ---
        meta_rows = []
        for i, result in enumerate(results):
            meta_rows.append({
                "prompt_idx": i,
                "prompt": result["prompt"],
                "target_token": group.target_token,
                "predicted_token": result["predicted_token"],
                "target_match": group.target_token.lower() in result["predicted_token"].lower() or result["predicted_token"].lower() in group.target_token.lower(),
                "n_tokens": len(result["tokens"]),
                "tokens": " | ".join(result["tokens"]),
                "top1_logit": result["top_k_predictions"][0][1],
                "top3": str([(t, round(l, 2)) for t, l in result["top_k_predictions"][:3]]),
            })

        df_meta = pd.DataFrame(meta_rows)
        df_meta.to_csv(group_dir / "prompts_meta.csv", index=False)

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
# Visualization
# =============================================================================

class AttractorVisualizer:
    """Generates plots for attractor analysis."""

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
        """PCA projection of final-layer residual streams, colored by group."""
        from numpy.linalg import svd

        group_names = list(all_results.keys())
        n_layers = all_results[group_names[0]][0]["n_layers"]
        last_layer = n_layers - 1

        # Collect all final-layer streams
        all_streams = []
        labels = []
        for group_name in group_names:
            for result in all_results[group_name]:
                all_streams.append(result["final_token_streams"][last_layer])
                labels.append(group_name)

        X = np.stack(all_streams)  # (total_prompts, d_model)
        X_centered = X - X.mean(axis=0)

        # PCA via SVD
        U, S, Vt = svd(X_centered, full_matrices=False)
        X_pca = X_centered @ Vt[:2].T  # project onto first 2 PCs

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))

        offset = 0
        for i, group_name in enumerate(group_names):
            n = len(all_results[group_name])
            ax.scatter(X_pca[offset:offset+n, 0], X_pca[offset:offset+n, 1],
                      c=[colors[i]], label=group_name, s=60, alpha=0.8, edgecolors="k", linewidths=0.5)

            # Draw centroid
            centroid = X_pca[offset:offset+n].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], c=[colors[i]], s=200, marker="*",
                      edgecolors="k", linewidths=1.5, zorder=10)

            offset += n

        ax.set_xlabel(f"PC1 (var explained: {S[0]**2 / (S**2).sum():.1%})")
        ax.set_ylabel(f"PC2 (var explained: {S[1]**2 / (S**2).sum():.1%})")
        ax.set_title(f"PCA of Final-Layer (L{last_layer}) Residual Streams\n(stars = group centroids)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "pca_final_layer.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {self.viz_dir}/pca_final_layer.png")

    def plot_cross_group_distances(self, all_results: dict[str, list[dict]]):
        """Bar chart of between-group vs within-group distances at final layer."""
        group_names = list(all_results.keys())
        n_layers = all_results[group_names[0]][0]["n_layers"]
        last_layer = n_layers - 1

        # Within-group mean distances
        within_distances = {}
        centroids = {}
        for name in group_names:
            streams = np.stack([r["final_token_streams"][last_layer] for r in all_results[name]])
            centroid = streams.mean(axis=0)
            centroids[name] = centroid
            dists = np.linalg.norm(streams - centroid[None, :], axis=1)
            within_distances[name] = float(dists.mean())

        # Between-group distances
        between_distances = {}
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                key = f"{group_names[i]}\nvs\n{group_names[j]}"
                d = np.linalg.norm(centroids[group_names[i]] - centroids[group_names[j]])
                between_distances[key] = float(d)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Within-group
        bars = axes[0].bar(range(len(within_distances)), list(within_distances.values()), color="steelblue")
        axes[0].set_xticks(range(len(within_distances)))
        axes[0].set_xticklabels(list(within_distances.keys()), rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Mean L2 Distance to Centroid")
        axes[0].set_title("Within-Group Distances (should be SMALL)")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Between-group
        bars = axes[1].bar(range(len(between_distances)), list(between_distances.values()), color="coral")
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
        from numpy.linalg import svd

        group_names = selected_groups or list(all_results.keys())[:3]  # Limit for readability
        n_layers = all_results[group_names[0]][0]["n_layers"]

        # Collect streams at multiple layers for selected groups
        # We'll do PCA on the combined final-layer space, then show trajectories
        all_streams_final = []
        all_labels = []
        for name in group_names:
            for result in all_results[name]:
                all_streams_final.append(result["final_token_streams"][n_layers - 1])
                all_labels.append(name)

        X_final = np.stack(all_streams_final)
        X_centered = X_final - X_final.mean(axis=0)
        U, S, Vt = svd(X_centered, full_matrices=False)
        # Use the PCA basis from the final layer to project all layers
        pca_basis = Vt[:2]  # (2, d_model)

        # Now plot trajectories: for each prompt, project its residual stream at each layer
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_names)))

        # Sample layers to show (not all, too cluttered)
        layer_samples = np.linspace(0, n_layers - 1, min(20, n_layers), dtype=int)

        for g_idx, name in enumerate(group_names):
            results = all_results[name]
            for p_idx, result in enumerate(results):
                trajectory = []
                for l in layer_samples:
                    vec = result["final_token_streams"][l]
                    projected = (vec - X_final.mean(axis=0)) @ pca_basis.T
                    trajectory.append(projected)

                trajectory = np.array(trajectory)  # (n_layer_samples, 2)

                # Plot trajectory as a line with arrow
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                       color=colors[g_idx], alpha=0.3, linewidth=0.8)
                # Mark start (early layer) with a small dot
                ax.scatter(trajectory[0, 0], trajectory[0, 1],
                          color=colors[g_idx], s=15, alpha=0.4, marker="o")
                # Mark end (final layer) with a larger dot
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                          color=colors[g_idx], s=50, alpha=0.8, marker="o",
                          edgecolors="k", linewidths=0.5)

            # Plot centroid at final layer
            final_streams = np.stack([r["final_token_streams"][n_layers - 1] for r in results])
            centroid = final_streams.mean(axis=0)
            centroid_proj = (centroid - X_final.mean(axis=0)) @ pca_basis.T
            ax.scatter(centroid_proj[0], centroid_proj[1], color=colors[g_idx],
                      s=300, marker="*", edgecolors="k", linewidths=1.5, zorder=10,
                      label=f"{name} (centroid)")

        ax.set_xlabel(f"PC1")
        ax.set_ylabel(f"PC2")
        ax.set_title("Layer Trajectories in PCA Space\n(small dots = early layers, large dots = final layer, stars = centroids)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "layer_trajectories_pca.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {self.viz_dir}/layer_trajectories_pca.png")

    def generate_all_plots(self, all_results: dict[str, list[dict]], all_metrics: dict[str, dict]):
        """Generate all visualization plots."""
        print("\n  Generating visualizations...")
        self.plot_convergence_trajectories(all_metrics)
        self.plot_cosine_heatmaps(all_results)
        self.plot_pca_final_layer(all_results)
        self.plot_cross_group_distances(all_results)
        self.plot_layer_trajectory_pca(all_results)
        print("  All visualizations complete.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Residual Stream Attractor Extraction")
    parser.add_argument(
        "--model", type=str, default="deepseek-ai/deepseek-llm-7b-base",
        help="HuggingFace model ID (default: deepseek-ai/deepseek-llm-7b-base)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--output", type=str, default="attractor_data",
        help="Output directory for saved data (default: attractor_data)",
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

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"{'='*60}")
    print(f"Residual Stream Attractor Extraction Tool")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")

    # Load model and tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device == "auto" else {"": device},
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup extractor
    extractor = ResidualStreamExtractor(model, tokenizer, device)
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

    all_results = {}
    all_metrics = {}

    for group in groups:
        print(f"\n{'='*60}")
        print(f"Processing: {group.name} (target: '{group.target_token}')")
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

            match_str = "+" if target_match else "X"
            print(f"  [{match_str}] '{prompt}' -> '{predicted}' (target: '{group.target_token}')")
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

        # Save raw data as CSV
        saver.save_group(group, results)

        # Save metrics JSON
        group_dir = output_path / group.name
        metrics_save = {
            "group_name": metrics["group_name"],
            "n_prompts": metrics["n_prompts"],
            "n_layers": metrics["n_layers"],
            "convergence": metrics["convergence"],
            "per_layer": {
                str(l): {k: v for k, v in data.items()}
                for l, data in metrics["per_layer"].items()
            },
        }
        with open(group_dir / "metrics.json", "w") as f:
            json.dump(metrics_save, f, indent=2)

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

        with open(output_path / "cross_group_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    # Generate visualizations
    visualizer.generate_all_plots(all_results, all_metrics)

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}/")
    print(f"")
    print(f"Structure:")
    print(f"  {{group}}/raw_streams/layer_XXX/prompt_YYY.csv  <- full residual streams")
    print(f"  {{group}}/final_token_streams/layer_XXX.csv     <- final-pos streams per prompt")
    print(f"  {{group}}/final_token_streams/all_layers_all_prompts.csv")
    print(f"  {{group}}/centroids/centroids_all_layers.csv")
    print(f"  {{group}}/prompts_meta.csv")
    print(f"  {{group}}/metrics.json")
    print(f"  visualizations/*.png")
    print(f"  cross_group_comparison.json")


if __name__ == "__main__":
    main()
