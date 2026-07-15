# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "numpy",
#     "accelerate",
# ]
# ///
"""
Residual Stream Attractor Extractor

Extracts raw residual stream activations at EVERY token position in EVERY layer
for grouped prompt sets. Outputs pure CSV for attractor analysis.

Models:
  - gpt2 (default, runs on CPU or single GPU)
  - deepseek (deepseek-ai/DeepSeek-V2-Lite or configurable, needs GPU)

Usage:
  uv run residual_attractors.py results/
  uv run residual_attractors.py results/ --model deepseek
  uv run residual_attractors.py results/ --model gpt2 --groups capital_multilingual arithmetic
  uv run residual_attractors.py results/ --model deepseek --model-name deepseek-ai/DeepSeek-V2-Lite

Self-bootstrapping: uses uv inline metadata for dependencies.
SLURM: see run.sbatch.sh (just `sbatch run.sbatch.sh`)
"""

import sys
import os
import shutil
import subprocess
import argparse
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field

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
        # Update PATH
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

# =============================================================================
# Now we're running under uv with all dependencies available
# =============================================================================

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# Prompt Groups — designed for attractor hunting
# =============================================================================

PROMPT_GROUPS: dict[str, dict] = {
    # --- Same concept, multiple languages (should converge to same attractor) ---
    "capital_berlin_multilingual": {
        "description": "Capital of Germany in multiple languages",
        "prompts": [
            "The capital of Germany is",
            "Die Hauptstadt von Deutschland ist",
            "La capitale de l'Allemagne est",
            "La capital de Alemania es",
            "La capitale della Germania è",
            "A capital da Alemanha é",
            "Столица Германии —",
            "ドイツの首都は",
            "德国的首都是",
            "독일의 수도는",
            "Hoofdstad van Duitsland is",
            "Tysklands huvudstad är",
        ],
    },
    "capital_paris_multilingual": {
        "description": "Capital of France in multiple languages",
        "prompts": [
            "The capital of France is",
            "Die Hauptstadt von Frankreich ist",
            "La capitale de la France est",
            "La capital de Francia es",
            "La capitale della Francia è",
            "A capital da França é",
            "Столица Франции —",
            "フランスの首都は",
            "法国的首都是",
            "프랑스의 수도는",
            "De hoofdstad van Frankrijk is",
            "Frankrikes huvudstad är",
        ],
    },
    "capital_tokyo_multilingual": {
        "description": "Capital of Japan in multiple languages",
        "prompts": [
            "The capital of Japan is",
            "Die Hauptstadt von Japan ist",
            "La capitale du Japon est",
            "La capital de Japón es",
            "La capitale del Giappone è",
            "A capital do Japão é",
            "Столица Японии —",
            "日本の首都は",
            "日本的首都是",
            "일본의 수도는",
        ],
    },

    # --- Same structure, different answers (should go to DIFFERENT attractors) ---
    "capital_varied_english": {
        "description": "Different capitals, same English structure",
        "prompts": [
            "The capital of Germany is",
            "The capital of France is",
            "The capital of Japan is",
            "The capital of Italy is",
            "The capital of Spain is",
            "The capital of Brazil is",
            "The capital of Russia is",
            "The capital of China is",
            "The capital of Australia is",
            "The capital of Canada is",
            "The capital of Egypt is",
            "The capital of India is",
        ],
    },

    # --- Arithmetic (same result, different paths) ---
    "arithmetic_result_4": {
        "description": "Arithmetic expressions that equal 4",
        "prompts": [
            "2 + 2 =",
            "1 + 3 =",
            "8 - 4 =",
            "8 / 2 =",
            "2 * 2 =",
            "The result of adding 2 and 2 is",
            "If you add one and three you get",
            "Two plus two equals",
            "Four minus zero is",
            "The square root of sixteen is",
        ],
    },
    "arithmetic_result_7": {
        "description": "Arithmetic expressions that equal 7",
        "prompts": [
            "3 + 4 =",
            "5 + 2 =",
            "14 / 2 =",
            "10 - 3 =",
            "1 + 6 =",
            "The result of adding 3 and 4 is",
            "Three plus four equals",
            "Seven minus zero is",
            "If you subtract 3 from 10 you get",
        ],
    },

    # --- Sentiment (same valence, different content) ---
    "sentiment_positive": {
        "description": "Strongly positive sentiment",
        "prompts": [
            "This movie was absolutely fantastic and I",
            "I am so incredibly happy because",
            "The best day of my life was when I",
            "Everything is wonderful and I feel",
            "This is the most amazing thing I have ever",
            "I love this so much, it makes me feel",
            "What a beautiful and perfect",
            "I'm thrilled and overjoyed because",
        ],
    },
    "sentiment_negative": {
        "description": "Strongly negative sentiment",
        "prompts": [
            "This movie was absolutely terrible and I",
            "I am so incredibly sad because",
            "The worst day of my life was when I",
            "Everything is horrible and I feel",
            "This is the most awful thing I have ever",
            "I hate this so much, it makes me feel",
            "What a ugly and terrible",
            "I'm devastated and heartbroken because",
        ],
    },

    # --- Factual completion (same domain, different facts) ---
    "color_of_things": {
        "description": "Colors of well-known things",
        "prompts": [
            "The color of the sky is",
            "The color of grass is",
            "The color of blood is",
            "The color of snow is",
            "The color of coal is",
            "The color of gold is",
            "The color of the sun is",
            "The color of the ocean is",
        ],
    },

    # --- Syntactic completion (same structure, tests grammar attractor) ---
    "plural_completion": {
        "description": "Plural noun completions",
        "prompts": [
            "The dogs are running in the",
            "The cats are sleeping on the",
            "The birds are flying over the",
            "The children are playing in the",
            "The cars are driving on the",
            "The books are sitting on the",
            "The flowers are growing in the",
            "The students are studying in the",
        ],
    },
    "singular_completion": {
        "description": "Singular noun completions",
        "prompts": [
            "The dog is running in the",
            "The cat is sleeping on the",
            "The bird is flying over the",
            "The child is playing in the",
            "The car is driving on the",
            "The book is sitting on the",
            "The flower is growing in the",
            "The student is studying in the",
        ],
    },

    # --- Next-word prediction convergence (famous quotes) ---
    "famous_beginnings": {
        "description": "Famous text beginnings that strongly predict next words",
        "prompts": [
            "To be or not to",
            "I think therefore I",
            "One small step for man, one giant leap for",
            "In the beginning God created the",
            "It was the best of times, it was the worst of",
            "All animals are equal, but some animals are more",
            "The only thing we have to fear is",
            "I have a dream that one day",
        ],
    },
}


# =============================================================================
# Model Loading
# =============================================================================

MODEL_CONFIGS = {
    "gpt2": {
        "name": "gpt2",
        "dtype": torch.float32,
        "device_map": None,  # will use cuda if available, else cpu
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "dtype": torch.float32,
        "device_map": None,
    },
    "gpt2-large": {
        "name": "gpt2-large",
        "dtype": torch.float32,
        "device_map": None,
    },
    "deepseek": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",  # klein, nativ supported
        "dtype": torch.bfloat16,
        "device_map": "auto",
        },
}


def load_model(model_key: str, model_name_override: str = None):
    """Load model and tokenizer."""
    if model_key not in MODEL_CONFIGS:
        print(f"Unknown model key '{model_key}'. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    config = MODEL_CONFIGS[model_key]
    model_name = model_name_override or config["name"]
    dtype = config["dtype"]
    device_map = config["device_map"]

    print(f"\nLoading model: {model_name}")
    print(f"  dtype: {dtype}")
    print(f"  device_map: {device_map or 'auto-detect'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "output_hidden_states": True,
    }
    if device_map:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # If no device_map, move to best available device
    if device_map is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model = model.to(device)
        print(f"  device: {device}")
    else:
        print(f"  device: distributed ({device_map})")

    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params / 1e6:.1f}M")
    print(f"  layers: {model.config.num_hidden_layers}")
    print(f"  d_model: {model.config.hidden_size}")

    return model, tokenizer


# =============================================================================
# Residual Stream Extraction
# =============================================================================

@dataclass
class TokenResidual:
    """Residual stream data for a single token at all layers."""
    prompt_idx: int
    token_pos: int
    token_text: str
    # shape: (n_layers+1, d_model) — layer 0 = embedding, layer N = after last transformer block
    residuals: np.ndarray = field(repr=False)


@dataclass
class PromptResult:
    """Full extraction result for one prompt."""
    prompt_idx: int
    prompt_text: str
    predicted_next_token: str
    token_texts: list[str]
    token_residuals: list[TokenResidual]


def extract_residual_streams(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 4,
) -> list[PromptResult]:
    """
    Extract residual stream at every token position, every layer.
    
    Returns one PromptResult per prompt, containing per-token residuals
    across all layers (embedding + all transformer layers).
    """
    results = []
    device = next(model.parameters()).device

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors, each (batch, seq_len, d_model)
        # hidden_states[0] = embedding output
        # hidden_states[i] = output of transformer layer i
        hidden_states = outputs.hidden_states

        n_layers_plus_one = len(hidden_states)
        attention_mask = inputs["attention_mask"]

        # Get predictions (next token after last real token)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        for batch_idx, prompt_text in enumerate(batch_prompts):
            prompt_global_idx = batch_start + batch_idx

            # Find actual token length (excluding padding)
            mask = attention_mask[batch_idx]
            seq_len = mask.sum().item()

            # Get token texts
            input_ids = inputs["input_ids"][batch_idx][:seq_len]
            token_texts = [tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                          for tid in input_ids]

            # Predicted next token
            last_logits = logits[batch_idx, seq_len - 1, :]
            pred_id = last_logits.argmax().item()
            predicted_next = tokenizer.decode([pred_id], clean_up_tokenization_spaces=False)

            # Extract residuals for each token position
            token_residuals = []
            for pos in range(seq_len):
                # Stack all layers for this token position
                # shape: (n_layers+1, d_model)
                residual_stack = np.stack([
                    hidden_states[layer_idx][batch_idx, pos, :].cpu().float().numpy()
                    for layer_idx in range(n_layers_plus_one)
                ])

                token_residuals.append(TokenResidual(
                    prompt_idx=prompt_global_idx,
                    token_pos=pos,
                    token_text=token_texts[pos],
                    residuals=residual_stack,
                ))

            results.append(PromptResult(
                prompt_idx=prompt_global_idx,
                prompt_text=prompt_text,
                predicted_next_token=predicted_next,
                token_texts=token_texts,
                token_residuals=token_residuals,
            ))

            print(f"    [{prompt_global_idx+1}/{len(prompts)}] "
                  f"'{prompt_text[:50]}...' → '{predicted_next}' "
                  f"({seq_len} tokens, {n_layers_plus_one} layers)")

    return results


# =============================================================================
# CSV Output
# =============================================================================

def save_group_csv(
    output_dir: Path,
    group_name: str,
    results: list[PromptResult],
):
    """
    Save extraction results as CSV files.
    
    Output structure:
      output_dir/
        group_name/
          prompts_meta.csv              — prompt texts and predictions
          all_token_streams/
            all_layers_all_tokens.csv   — FULL data: every token, every layer, all dims
            layer_000.csv               — per-layer files (for quick access)
            layer_001.csv
            ...
          final_token_streams/
            all_layers_all_prompts.csv  — only final token per prompt (backward compat)
            layer_000.csv
            ...
    """
    group_dir = output_dir / group_name
    all_token_dir = group_dir / "all_token_streams"
    final_token_dir = group_dir / "final_token_streams"

    all_token_dir.mkdir(parents=True, exist_ok=True)
    final_token_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        print(f"  WARNING: No results for group '{group_name}'")
        return

    n_layers = results[0].token_residuals[0].residuals.shape[0]
    d_model = results[0].token_residuals[0].residuals.shape[1]

    dim_headers = [f"dim_{i:04d}" for i in range(d_model)]

    # --- 1. Metadata CSV ---
    meta_path = group_dir / "prompts_meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "prompt", "predicted_next_token", "n_tokens", "tokens"])
        for r in results:
            writer.writerow([
                r.prompt_idx,
                r.prompt_text,
                r.predicted_next_token,
                len(r.token_texts),
                "|".join(r.token_texts),
            ])
    print(f"    Saved: {meta_path}")

    # --- 2. All tokens, all layers (combined CSV) ---
    all_csv_path = all_token_dir / "all_layers_all_tokens.csv"
    with open(all_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "prompt_idx", "token_pos", "token_text"] + dim_headers)

        for layer_idx in range(n_layers):
            for r in results:
                for tok_res in r.token_residuals:
                    row = [
                        layer_idx,
                        tok_res.prompt_idx,
                        tok_res.token_pos,
                        tok_res.token_text,
                    ] + tok_res.residuals[layer_idx].tolist()
                    writer.writerow(row)

    print(f"    Saved: {all_csv_path} "
          f"({n_layers} layers × {sum(len(r.token_residuals) for r in results)} tokens × {d_model} dims)")

    # --- 3. Per-layer CSVs (all tokens) ---
    for layer_idx in range(n_layers):
        layer_path = all_token_dir / f"layer_{layer_idx:03d}.csv"
        with open(layer_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt_idx", "token_pos", "token_text"] + dim_headers)
            for r in results:
                for tok_res in r.token_residuals:
                    row = [
                        tok_res.prompt_idx,
                        tok_res.token_pos,
                        tok_res.token_text,
                    ] + tok_res.residuals[layer_idx].tolist()
                    writer.writerow(row)

    print(f"    Saved: {n_layers} per-layer CSVs in {all_token_dir}")

    # --- 4. Final token only (combined CSV, backward compat) ---
    final_csv_path = final_token_dir / "all_layers_all_prompts.csv"
    with open(final_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "prompt_idx"] + dim_headers)

        for layer_idx in range(n_layers):
            for r in results:
                # Last token residual
                last_tok = r.token_residuals[-1]
                row = [layer_idx, r.prompt_idx] + last_tok.residuals[layer_idx].tolist()
                writer.writerow(row)

    print(f"    Saved: {final_csv_path}")

    # --- 5. Final token per-layer CSVs ---
    for layer_idx in range(n_layers):
        layer_path = final_token_dir / f"layer_{layer_idx:03d}.csv"
        with open(layer_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt_idx"] + dim_headers)
            for r in results:
                last_tok = r.token_residuals[-1]
                row = [r.prompt_idx] + last_tok.residuals[layer_idx].tolist()
                writer.writerow(row)

    print(f"    Saved: {n_layers} final-token per-layer CSVs in {final_token_dir}")

    # --- 6. Group info JSON ---
    info = {
        "group_name": group_name,
        "n_prompts": len(results),
        "n_layers": n_layers,
        "d_model": d_model,
        "total_tokens": sum(len(r.token_residuals) for r in results),
        "prompts": [r.prompt_text for r in results],
        "predictions": [r.predicted_next_token for r in results],
    }
    info_path = group_dir / "group_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract residual stream data for attractor analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run residual_attractors.py results/
  uv run residual_attractors.py results/ --model deepseek
  uv run residual_attractors.py results/ --model gpt2-large --groups capital_berlin_multilingual sentiment_positive
  uv run residual_attractors.py results/ --model deepseek --model-name deepseek-ai/DeepSeek-V2-Lite-Chat

Available groups:
""" + "\n".join(f"  {k}: {v['description']}" for k, v in PROMPT_GROUPS.items())
    )

    parser.add_argument("output_dir", type=str,
                       help="Output directory for CSV data")
    parser.add_argument("--model", type=str, default="gpt2",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to use (default: gpt2)")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Override model name/path (e.g. for custom DeepSeek variant)")
    parser.add_argument("--groups", type=str, nargs="*", default=None,
                       help="Specific prompt groups to run (default: all)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for inference (default: 4)")
    parser.add_argument("--dims", type=int, default=None,
                       help="Limit output dimensions (default: all). "
                            "Saves only first N dims to reduce file size.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select groups
    if args.groups:
        groups_to_run = {}
        for g in args.groups:
            if g in PROMPT_GROUPS:
                groups_to_run[g] = PROMPT_GROUPS[g]
            else:
                print(f"WARNING: Unknown group '{g}'. Available: {list(PROMPT_GROUPS.keys())}")
        if not groups_to_run:
            print("ERROR: No valid groups specified.")
            sys.exit(1)
    else:
        groups_to_run = PROMPT_GROUPS

    print("=" * 70)
    print("  RESIDUAL STREAM ATTRACTOR EXTRACTOR")
    print("=" * 70)
    print(f"  Model:      {args.model}" + (f" ({args.model_name})" if args.model_name else ""))
    print(f"  Output:     {output_dir}")
    print(f"  Groups:     {len(groups_to_run)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Dims limit: {args.dims or 'all'}")
    total_prompts = sum(len(g["prompts"]) for g in groups_to_run.values())
    print(f"  Total prompts: {total_prompts}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model(args.model, args.model_name)

    # Process each group
    for group_idx, (group_name, group_data) in enumerate(groups_to_run.items()):
        prompts = group_data["prompts"]
        print(f"\n{'─' * 60}")
        print(f"  Group [{group_idx+1}/{len(groups_to_run)}]: {group_name}")
        print(f"  Description: {group_data['description']}")
        print(f"  Prompts: {len(prompts)}")
        print(f"{'─' * 60}")

        # Extract
        results = extract_residual_streams(
            model, tokenizer, prompts,
            batch_size=args.batch_size,
        )

        # Optionally truncate dimensions
        if args.dims:
            for r in results:
                for tok_res in r.token_residuals:
                    tok_res.residuals = tok_res.residuals[:, :args.dims]

        # Save
        save_group_csv(output_dir, group_name, results)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Output directory: {output_dir}")
    print(f"  Groups extracted: {len(groups_to_run)}")
    print(f"{'=' * 70}")

    # List output
    print(f"\n  Directory structure:")
    for group_name in groups_to_run:
        gdir = output_dir / group_name
        if gdir.exists():
            all_csv = gdir / "all_token_streams" / "all_layers_all_tokens.csv"
            if all_csv.exists():
                size_mb = all_csv.stat().st_size / (1024 * 1024)
                print(f"    {group_name}/all_token_streams/all_layers_all_tokens.csv ({size_mb:.1f} MB)")

    print(f"\n  To visualize:")
    print(f"    uv run viewer.py {output_dir}")
    print(f"    uv run viewer.py {output_dir}/{list(groups_to_run.keys())[0]}")


if __name__ == "__main__":
    main()
