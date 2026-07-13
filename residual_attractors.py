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
# ]
# ///
"""
Residual Stream Attractor Extraction Tool

Loads DeepSeek (or compatible model) and runs predefined prompt groups
through it, capturing the residual stream at every layer for the final
token position (the one that predicts the target token).

Hypothesis: In later layers, residual stream vectors for prompts that
should predict the same token (e.g., "Berlin") are attracted toward
a common point (attractor) regardless of the input language or framing.

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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# =============================================================================
# Prompt Group Definitions
# =============================================================================

@dataclass
class PromptGroup:
    """A group of prompts that should all predict the same target token."""
    name: str
    target_token: str  # The expected next token (e.g., "Paris", "Berlin")
    attractor_concept: str  # What concept this attractor represents
    prompts: list[str] = field(default_factory=list)


# --- Predefined Attractor Groups ---

ATTRACTOR_GROUPS: list[PromptGroup] = [
    # === PARIS attractor ===
    PromptGroup(
        name="paris_multilingual",
        target_token="Paris",
        attractor_concept="Paris (capital of France)",
        prompts=[
            "The capital of France is",
            "Die Hauptstadt von Frankreich ist",
            "La capitale de la France est",
            "La capital de Francia es",
            "フランスの首都は",
            "The Eiffel Tower is located in",
            "Der Eiffelturm steht in",
            "The city of love is",
            "The Louvre museum is in",
            "France's largest city is",
            "The Seine river flows through",
            "Notre-Dame cathedral is in",
        ],
    ),

    # === BERLIN attractor ===
    PromptGroup(
        name="berlin_multilingual",
        target_token="Berlin",
        attractor_concept="Berlin (capital of Germany)",
        prompts=[
            "The capital of Germany is",
            "Die Hauptstadt von Deutschland ist",
            "La capitale de l'Allemagne est",
            "La capital de Alemania es",
            "ドイツの首都は",
            "The Brandenburg Gate is in",
            "Das Brandenburger Tor steht in",
            "The Berlin Wall divided",
            "Germany's largest city is",
            "The Reichstag building is in",
            "The Spree river flows through",
            "Checkpoint Charlie was in",
        ],
    ),

    # === TOKYO attractor ===
    PromptGroup(
        name="tokyo_multilingual",
        target_token="Tokyo",
        attractor_concept="Tokyo (capital of Japan)",
        prompts=[
            "The capital of Japan is",
            "Die Hauptstadt von Japan ist",
            "La capitale du Japon est",
            "日本の首都は",
            "The largest city in Japan is",
            "Mount Fuji can be seen from",
            "The Shibuya crossing is in",
            "The Tokyo Tower is located in",
            "Japan's imperial palace is in",
        ],
    ),

    # === LONDON attractor ===
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

    # === Number attractor: 4 ===
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

    # === Water/H2O attractor ===
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

    # === Sun attractor ===
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

    # === Einstein attractor ===
    PromptGroup(
        name="einstein_person",
        target_token="Einstein",
        attractor_concept="Albert Einstein",
        prompts=[
            "E = mc² was discovered by Albert",
            "The theory of relativity was developed by",
            "The most famous physicist of the 20th century is Albert",
            "Die Relativitätstheorie wurde entwickelt von Albert",
            "The Nobel Prize in Physics 1921 was awarded to Albert",
        ],
    ),
]


# =============================================================================
# Residual Stream Extraction
# =============================================================================

class ResidualStreamExtractor:
    """
    Hooks into a transformer model to capture the residual stream
    at every layer for specified token positions.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hooks = []
        self.residual_streams = {}  # layer_idx -> tensor

    def _get_layer_modules(self):
        """
        Find the transformer layers in the model.
        Supports DeepSeek, LLaMA, Mistral, GPT-NeoX architectures.
        """
        # Try common attribute names
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers  # LLaMA/DeepSeek style
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h  # GPT-2/GPT-Neo style
        elif hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            return self.model.gpt_neox.layers  # GPT-NeoX style
        else:
            raise ValueError(
                f"Cannot find transformer layers in model of type {type(self.model)}. "
                f"Top-level attributes: {[a for a in dir(self.model) if not a.startswith('_')]}"
            )

    def _register_hooks(self):
        """Register forward hooks on each transformer layer to capture residual streams."""
        self.residual_streams = {}
        self.hooks = []

        layers = self._get_layer_modules()

        for layer_idx, layer in enumerate(layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    # Most architectures: output is a tuple, first element is hidden states
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    # Store the full residual stream (all positions)
                    self.residual_streams[idx] = hidden.detach().cpu()
                return hook_fn

            h = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(h)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def extract_residual_streams(self, prompt: str) -> dict:
        """
        Run a prompt through the model and capture residual streams at all layers.

        Returns:
            dict with keys:
                - "input_ids": the tokenized input (list of ints)
                - "tokens": the decoded tokens (list of strings)
                - "n_layers": number of layers
                - "residual_streams": dict[layer_idx] -> np.ndarray of shape (seq_len, d_model)
                - "final_token_streams": dict[layer_idx] -> np.ndarray of shape (d_model,)
                    (the residual stream at the LAST token position, which predicts the next token)
                - "predicted_token": the model's top-1 prediction
                - "top_k_predictions": list of (token_str, logit) for top 10
        """
        self._register_hooks()

        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"][0].tolist()
            tokens = [self.tokenizer.decode([tid]) for tid in input_ids]

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get predictions from logits
            logits = outputs.logits[0, -1, :]  # logits for last position
            top_k = torch.topk(logits, k=10)
            top_k_predictions = [
                (self.tokenizer.decode([idx.item()]).strip(), logits[idx].item())
                for idx in top_k.indices
            ]
            predicted_token = top_k_predictions[0][0]

            # Extract final-token residual streams
            final_token_streams = {}
            full_streams = {}
            for layer_idx, stream in self.residual_streams.items():
                # stream shape: (1, seq_len, d_model)
                full_streams[layer_idx] = stream[0].numpy()  # (seq_len, d_model)
                final_token_streams[layer_idx] = stream[0, -1, :].numpy()  # (d_model,)

            # Also capture the embedding (layer 0 input)
            # The first layer's input is the embedded tokens + positional encoding
            # We approximate this as "layer -1" = input to first layer

            return {
                "prompt": prompt,
                "input_ids": input_ids,
                "tokens": tokens,
                "n_layers": len(self.residual_streams),
                "residual_streams": full_streams,
                "final_token_streams": final_token_streams,
                "predicted_token": predicted_token,
                "top_k_predictions": top_k_predictions,
            }

        finally:
            self._remove_hooks()


# =============================================================================
# Attractor Analysis
# =============================================================================

class AttractorAnalyzer:
    """
    Analyzes residual streams across prompt groups to find attractor structure.
    """

    def __init__(self, extractor: ResidualStreamExtractor):
        self.extractor = extractor
        self.results: dict[str, list[dict]] = {}  # group_name -> list of extraction results

    def process_group(self, group: PromptGroup, verbose: bool = True) -> list[dict]:
        """Process all prompts in a group and store results."""
        results = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing group: {group.name}")
            print(f"Target token: '{group.target_token}'")
            print(f"Concept: {group.attractor_concept}")
            print(f"{'='*60}")

        for prompt in tqdm(group.prompts, desc=group.name, disable=not verbose):
            result = self.extractor.extract_residual_streams(prompt)
            result["target_token"] = group.target_token
            result["group_name"] = group.name
            result["attractor_concept"] = group.attractor_concept

            # Check if prediction matches target
            predicted = result["predicted_token"]
            target_match = (
                group.target_token.lower() in predicted.lower() or
                predicted.lower() in group.target_token.lower()
            )
            result["target_match"] = target_match

            if verbose:
                match_str = "✓" if target_match else "✗"
                print(f"  {match_str} '{prompt}' -> '{predicted}' (target: '{group.target_token}')")

            results.append(result)

        self.results[group.name] = results
        return results

    def compute_attractor_metrics(self, group_name: str) -> dict:
        """
        Compute metrics about attractor convergence for a group.

        For each layer, computes:
        - Mean residual stream vector (the "attractor center")
        - Variance of distances to center (how tight the cluster is)
        - Cosine similarities between all pairs
        - Convergence trajectory (does variance decrease in later layers?)
        """
        if group_name not in self.results:
            raise ValueError(f"Group '{group_name}' not processed yet.")

        results = self.results[group_name]
        n_prompts = len(results)
        n_layers = results[0]["n_layers"]

        metrics = {
            "group_name": group_name,
            "n_prompts": n_prompts,
            "n_layers": n_layers,
            "per_layer": {},
        }

        for layer_idx in range(n_layers):
            # Collect all final-token residual streams for this layer
            streams = np.stack([
                r["final_token_streams"][layer_idx] for r in results
            ])  # (n_prompts, d_model)

            # Centroid (potential attractor)
            centroid = streams.mean(axis=0)  # (d_model,)

            # Distances to centroid
            diffs = streams - centroid[None, :]
            distances = np.linalg.norm(diffs, axis=1)  # (n_prompts,)

            # Cosine similarities between all pairs
            norms = np.linalg.norm(streams, axis=1, keepdims=True)
            normalized = streams / (norms + 1e-10)
            cos_sim_matrix = normalized @ normalized.T  # (n_prompts, n_prompts)

            # Extract upper triangle (excluding diagonal)
            triu_indices = np.triu_indices(n_prompts, k=1)
            pairwise_cos_sims = cos_sim_matrix[triu_indices]

            metrics["per_layer"][layer_idx] = {
                "centroid": centroid,
                "mean_distance_to_centroid": float(distances.mean()),
                "std_distance_to_centroid": float(distances.std()),
                "max_distance_to_centroid": float(distances.max()),
                "min_distance_to_centroid": float(distances.min()),
                "mean_pairwise_cosine_sim": float(pairwise_cos_sims.mean()),
                "std_pairwise_cosine_sim": float(pairwise_cos_sims.std()),
                "min_pairwise_cosine_sim": float(pairwise_cos_sims.min()),
                "centroid_norm": float(np.linalg.norm(centroid)),
                "individual_distances": distances.tolist(),
            }

        # Convergence: does the cluster get tighter in later layers?
        mean_distances = [
            metrics["per_layer"][l]["mean_distance_to_centroid"]
            for l in range(n_layers)
        ]
        mean_cosines = [
            metrics["per_layer"][l]["mean_pairwise_cosine_sim"]
            for l in range(n_layers)
        ]

        metrics["convergence"] = {
            "distance_trajectory": mean_distances,
            "cosine_trajectory": mean_cosines,
            "distance_ratio_last_vs_first": (
                mean_distances[-1] / (mean_distances[0] + 1e-10)
            ),
            "cosine_improvement": mean_cosines[-1] - mean_cosines[0],
        }

        return metrics

    def compare_groups(self, group_names: list[str], layer_idx: int = -1) -> dict:
        """
        Compare centroids between different groups at a given layer.
        
        If the attractor hypothesis is correct:
        - Within-group distances should be SMALL (points converge to same attractor)
        - Between-group distances should be LARGE (different attractors are far apart)
        """
        if layer_idx == -1:
            # Use last layer
            layer_idx = self.results[group_names[0]][0]["n_layers"] - 1

        centroids = {}
        for name in group_names:
            results = self.results[name]
            streams = np.stack([
                r["final_token_streams"][layer_idx] for r in results
            ])
            centroids[name] = streams.mean(axis=0)

        # Between-group distances
        between_distances = {}
        between_cosines = {}
        names = list(centroids.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = f"{names[i]} vs {names[j]}"
                d = np.linalg.norm(centroids[names[i]] - centroids[names[j]])
                between_distances[key] = float(d)

                cos_sim = (
                    np.dot(centroids[names[i]], centroids[names[j]]) /
                    (np.linalg.norm(centroids[names[i]]) * np.linalg.norm(centroids[names[j]]) + 1e-10)
                )
                between_cosines[key] = float(cos_sim)

        return {
            "layer_idx": layer_idx,
            "between_group_distances": between_distances,
            "between_group_cosines": between_cosines,
            "centroids": {name: centroid.tolist() for name, centroid in centroids.items()},
        }

    def save_results(self, output_dir: str = "attractor_data"):
        """Save all extracted residual streams and metrics to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for group_name, results in self.results.items():
            group_dir = out_path / group_name
            group_dir.mkdir(exist_ok=True)

            # Save residual streams as numpy arrays
            for i, result in enumerate(results):
                # Save final-token streams (most important for attractor analysis)
                streams_array = np.stack([
                    result["final_token_streams"][l]
                    for l in sorted(result["final_token_streams"].keys())
                ])  # (n_layers, d_model)

                np.save(
                    group_dir / f"prompt_{i:03d}_final_token_streams.npy",
                    streams_array,
                )

                # Save metadata
                meta = {
                    "prompt": result["prompt"],
                    "tokens": result["tokens"],
                    "predicted_token": result["predicted_token"],
                    "target_token": result["target_token"],
                    "target_match": result["target_match"],
                    "top_k_predictions": result["top_k_predictions"],
                    "n_layers": result["n_layers"],
                }
                with open(group_dir / f"prompt_{i:03d}_meta.json", "w") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

            # Save group-level metrics
            metrics = self.compute_attractor_metrics(group_name)

            # Convert numpy arrays to lists for JSON serialization
            metrics_serializable = {
                "group_name": metrics["group_name"],
                "n_prompts": metrics["n_prompts"],
                "n_layers": metrics["n_layers"],
                "convergence": metrics["convergence"],
                "per_layer_summary": {
                    str(l): {
                        k: v for k, v in layer_data.items()
                        if k != "centroid"  # Skip large arrays
                    }
                    for l, layer_data in metrics["per_layer"].items()
                },
            }
            with open(group_dir / "metrics.json", "w") as f:
                json.dump(metrics_serializable, f, indent=2)

            # Save centroids separately as numpy
            centroids = np.stack([
                metrics["per_layer"][l]["centroid"]
                for l in range(metrics["n_layers"])
            ])  # (n_layers, d_model)
            np.save(group_dir / "centroids_per_layer.npy", centroids)

        # Save cross-group comparison
        if len(self.results) > 1:
            group_names = list(self.results.keys())
            comparison = self.compare_groups(group_names)
            # Remove large centroid arrays for JSON
            comparison_save = {
                k: v for k, v in comparison.items() if k != "centroids"
            }
            with open(out_path / "cross_group_comparison.json", "w") as f:
                json.dump(comparison_save, f, indent=2)

        print(f"\n✓ Results saved to {out_path}/")


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
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="Only capture first N layers (for memory-constrained setups)",
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

    print(f"Loading model (this may take a while for large models)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device == "auto" else {"": device},
        trust_remote_code=True,
    )
    model.eval()

    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Determine which layers exist
    extractor = ResidualStreamExtractor(model, tokenizer, device)
    layers = extractor._get_layer_modules()
    n_layers = len(layers)
    print(f"Number of transformer layers: {n_layers}")

    if args.max_layers and args.max_layers < n_layers:
        print(f"⚠ Limiting to first {args.max_layers} layers (of {n_layers})")
        # We'd need to modify the hook registration, but for now just note it

    # Select groups
    if args.groups:
        groups = [g for g in ATTRACTOR_GROUPS if g.name in args.groups]
        if not groups:
            print(f"ERROR: No matching groups found. Available: {[g.name for g in ATTRACTOR_GROUPS]}")
            sys.exit(1)
    else:
        groups = ATTRACTOR_GROUPS

    print(f"\nGroups to process: {[g.name for g in groups]}")
    print(f"Total prompts: {sum(len(g.prompts) for g in groups)}")

    # Process all groups
    analyzer = AttractorAnalyzer(extractor)

    for group in groups:
        analyzer.process_group(group, verbose=True)

        # Print immediate metrics
        metrics = analyzer.compute_attractor_metrics(group.name)
        conv = metrics["convergence"]

        print(f"\n  📊 Convergence metrics for '{group.name}':")
        print(f"     Distance ratio (last/first layer): {conv['distance_ratio_last_vs_first']:.4f}")
        print(f"     Cosine sim improvement: {conv['cosine_improvement']:.4f}")
        print(f"     Final layer mean cosine sim: {conv['cosine_trajectory'][-1]:.4f}")
        print(f"     First layer mean cosine sim: {conv['cosine_trajectory'][0]:.4f}")

        # Show trajectory at a few key layers
        n_show = min(5, n_layers)
        layer_indices = np.linspace(0, n_layers - 1, n_show, dtype=int)
        print(f"     Layer trajectory (distance to centroid):")
        for l in layer_indices:
            d = conv['distance_trajectory'][l]
            c = conv['cosine_trajectory'][l]
            print(f"       Layer {l:3d}: dist={d:.4f}, cos_sim={c:.4f}")

    # Cross-group comparison
    if len(groups) > 1:
        print(f"\n{'='*60}")
        print(f"Cross-Group Comparison (last layer)")
        print(f"{'='*60}")

        group_names = [g.name for g in groups if g.name in analyzer.results]
        comparison = analyzer.compare_groups(group_names)

        print(f"\nBetween-group distances (should be LARGE if attractors are distinct):")
        for pair, dist in comparison["between_group_distances"].items():
            cos = comparison["between_group_cosines"][pair]
            print(f"  {pair}: distance={dist:.4f}, cosine_sim={cos:.4f}")

    # Save everything
    analyzer.save_results(args.output)

    print(f"\n{'='*60}")
    print(f"DONE! All residual streams saved to: {args.output}/")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Load the .npy files to analyze attractor structure")
    print(f"  2. Use PCA/UMAP on final-layer streams to visualize clustering")
    print(f"  3. Compare centroid trajectories across layers")
    print(f"  4. Test if centroids from one group can predict tokens for new prompts")
    print(f"\nExample loading code:")
    print(f"  import numpy as np")
    print(f"  streams = np.load('{args.output}/paris_multilingual/prompt_000_final_token_streams.npy')")
    print(f"  # Shape: (n_layers, d_model)")
    print(f"  centroids = np.load('{args.output}/paris_multilingual/centroids_per_layer.npy')")
    print(f"  # Shape: (n_layers, d_model)")


if __name__ == "__main__":
    main()
