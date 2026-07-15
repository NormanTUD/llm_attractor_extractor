# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "numpy",
#     "accelerate",
#     "safetensors",
# ]
# ///
"""
Residual Stream FULL Extractor — Extract EVERYTHING

Extracts from each run:
  - Residual stream activations at EVERY token × EVERY layer
  - Attention weights (all heads, all layers) — who attends to whom
  - Attention output (per-head, per-layer)
  - MLP intermediate activations (per-layer)
  - Layer norm scales (pre/post per layer)
  - Logits at every token position (full vocab or top-k)
  - Top-k token predictions at every position (not just last)
  - Token probabilities (softmax of logits)
  - Entropy of prediction distribution at each position
  - Per-layer residual stream norms (L2)
  - Per-layer residual stream deltas (change from previous layer)
  - Cosine similarity between consecutive layers
  - Full tokenization details (token IDs, byte offsets, special tokens)
  - Model config and architecture metadata
  - Inference timing per prompt

Usage:
  uv run extract.py results/
  uv run extract.py results/ --model gpt2-large
  uv run extract.py results/ --model deepseek --groups capital_berlin_multilingual
  uv run extract.py results/ --model gpt2 --top-k 50 --save-attention --save-mlp

Self-bootstrapping: uses uv inline metadata for dependencies.
"""

import sys
import os
import shutil
import subprocess
import argparse
import csv
import json
import time as time_module
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

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

# =============================================================================
# Now we're running under uv with all dependencies available
# =============================================================================

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


# =============================================================================
# Prompt Groups — designed for attractor hunting
# =============================================================================

PROMPT_GROUPS: dict[str, dict] = {
    "capital_berlin_multilingual": {
        "description": "Capital of Germany in multiple languages",
        "expected_answer": "Berlin",
        "category": "factual_multilingual",
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
        "expected_answer": "Paris",
        "category": "factual_multilingual",
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
        "expected_answer": "Tokyo",
        "category": "factual_multilingual",
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
    "capital_varied_english": {
        "description": "Different capitals, same English structure",
        "expected_answer": "varies",
        "category": "factual_varied",
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
    "arithmetic_result_4": {
        "description": "Arithmetic expressions that equal 4",
        "expected_answer": "4",
        "category": "arithmetic",
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
        "expected_answer": "7",
        "category": "arithmetic",
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
    "sentiment_positive": {
        "description": "Strongly positive sentiment",
        "expected_answer": "positive_continuation",
        "category": "sentiment",
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
        "expected_answer": "negative_continuation",
        "category": "sentiment",
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
    "color_of_things": {
        "description": "Colors of well-known things",
        "expected_answer": "color_word",
        "category": "factual_simple",
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
    "plural_completion": {
        "description": "Plural noun completions",
        "expected_answer": "location_noun",
        "category": "syntactic",
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
        "expected_answer": "location_noun",
        "category": "syntactic",
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
    "famous_beginnings": {
        "description": "Famous text beginnings that strongly predict next words",
        "expected_answer": "famous_next_word",
        "category": "memorized",
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
        "device_map": None,
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
    "gpt2-xl": {
        "name": "gpt2-xl",
        "dtype": torch.float32,
        "device_map": None,
    },
    "deepseek": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "dtype": torch.bfloat16,
        "device_map": "auto",
    },
}


def load_model(model_key: str, model_name_override: str = None):
    """Load model and tokenizer, return full config info."""
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

    # Load config first for metadata
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "output_hidden_states": True,
        "output_attentions": True,  # EXTRACT ATTENTION WEIGHTS
    }
    if device_map:
        load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  device: distributed ({device_map})")

    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model_config.num_hidden_layers
    d_model = model_config.hidden_size
    n_heads = model_config.num_attention_heads
    vocab_size = model_config.vocab_size
    # Try to get intermediate size (MLP hidden dim)
    d_ff = getattr(model_config, 'intermediate_size', None) or getattr(model_config, 'n_inner', None) or d_model * 4

    print(f"  parameters: {n_params / 1e6:.1f}M")
    print(f"  layers: {n_layers}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_head: {d_model // n_heads}")
    print(f"  d_ff (MLP): {d_ff}")
    print(f"  vocab_size: {vocab_size}")

    model_info = {
        "model_name": model_name,
        "model_key": model_key,
        "dtype": str(dtype),
        "n_params": n_params,
        "n_params_M": round(n_params / 1e6, 1),
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_head": d_model // n_heads,
        "d_ff": d_ff,
        "vocab_size": vocab_size,
        "max_position_embeddings": getattr(model_config, 'max_position_embeddings', None) or getattr(model_config, 'n_positions', None),
        "activation_function": getattr(model_config, 'activation_function', None) or getattr(model_config, 'hidden_act', None),
        "layer_norm_epsilon": getattr(model_config, 'layer_norm_epsilon', None) or getattr(model_config, 'rms_norm_eps', None),
        "tie_word_embeddings": getattr(model_config, 'tie_word_embeddings', None),
        "architecture": model_config.architectures[0] if hasattr(model_config, 'architectures') and model_config.architectures else str(type(model).__name__),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    return model, tokenizer, model_info


# =============================================================================
# Full Extraction Data Structures
# =============================================================================

@dataclass
class TokenFullExtraction:
    """EVERYTHING extracted for a single token at all layers."""
    prompt_idx: int
    token_pos: int
    token_id: int
    token_text: str
    token_bytes: bytes
    is_special: bool

    # Residual streams: shape (n_layers+1, d_model)
    residuals: np.ndarray = field(repr=False)

    # Per-layer norms: shape (n_layers+1,)
    residual_norms: np.ndarray = field(repr=False)

    # Per-layer deltas (change from previous layer): shape (n_layers,)
    residual_deltas: np.ndarray = field(repr=False)

    # Cosine similarity between consecutive layers: shape (n_layers,)
    layer_cosines: np.ndarray = field(repr=False)

    # Logits at this position
    logits_topk_ids: np.ndarray = field(repr=False)
    logits_topk_values: np.ndarray = field(repr=False)
    logits_topk_tokens: list[str] = field(repr=False)

    # Attention patterns
    attention_to_self: np.ndarray = field(repr=False)
    attention_from_self: np.ndarray = field(repr=False)
    attention_max_source: np.ndarray = field(repr=False)

    # --- Fields WITH defaults MUST come last ---
    prediction_entropy: float = 0.0
    next_token_prob: float = 0.0
    next_token_rank: int = -1

@dataclass
class PromptFullResult:
    """Full extraction result for one prompt — EVERYTHING."""
    prompt_idx: int
    prompt_text: str
    predicted_next_token: str
    predicted_next_token_id: int
    predicted_next_prob: float
    predicted_top5_tokens: list[str]
    predicted_top5_probs: list[float]
    token_ids: list[int]
    token_texts: list[str]
    n_tokens: int
    inference_time_ms: float

    # Full attention matrices per layer (optional, large)
    # shape per layer: (n_heads, seq_len, seq_len)
    attention_matrices: Optional[list[np.ndarray]] = field(default=None, repr=False)

    # Per-token extractions
    token_extractions: list[TokenFullExtraction] = field(default_factory=list, repr=False)


# =============================================================================
# FULL Extraction Engine
# =============================================================================

def extract_everything(
    model,
    tokenizer,
    prompts: list[str],
    model_info: dict,
    batch_size: int = 1,  # Use 1 for maximum extraction (attention needs care with padding)
    top_k: int = 20,
    save_attention_matrices: bool = False,
) -> list[PromptFullResult]:
    """
    Extract EVERYTHING possible from each prompt run.
    
    For each token at each layer, extracts:
      - Residual stream vector
      - Residual norm (L2)
      - Delta from previous layer
      - Cosine similarity to previous layer
      - Attention patterns (who this token attends to, who attends to it)
      - Logits and top-k predictions at this position
      - Prediction entropy
      - Next-token probability and rank
    """
    results = []
    device = next(model.parameters()).device
    n_layers = model_info["n_layers"]
    n_heads = model_info["n_heads"]
    vocab_size = model_info["vocab_size"]

    for prompt_idx, prompt_text in enumerate(prompts):
        prefixed_prompt = prompt_text
        t_start = time_module.perf_counter()

        # Tokenize single prompt (no padding issues)
        inputs = tokenizer(
            prefixed_prompt,  # <-- use prefixed version
            return_tensors="pt",
            truncation=True,
            max_length=128,
            return_offsets_mapping=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                output_attentions=True,
            )

        t_end = time_module.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000

        # --- Extract raw outputs ---
        hidden_states = outputs.hidden_states  # tuple of (n_layers+1) × (1, seq_len, d_model)
        attentions = outputs.attentions        # tuple of n_layers × (1, n_heads, seq_len, seq_len)
        logits = outputs.logits                # (1, seq_len, vocab_size)

        seq_len = inputs["input_ids"].shape[1]
        input_ids = inputs["input_ids"][0].cpu().tolist()

        # Token texts and metadata
        token_texts = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in input_ids]
        special_tokens = set(tokenizer.all_special_ids)

        # --- Last-position predictions ---
        last_logits = logits[0, -1, :]  # (vocab_size,)
        last_probs = torch.softmax(last_logits, dim=-1)
        top5_vals, top5_ids = torch.topk(last_probs, k=min(5, vocab_size))
        predicted_next_id = top5_ids[0].item()
        predicted_next_token = tokenizer.decode([predicted_next_id], clean_up_tokenization_spaces=False)
        predicted_next_prob = top5_vals[0].item()
        predicted_top5_tokens = [tokenizer.decode([tid.item()], clean_up_tokenization_spaces=False) for tid in top5_ids]
        predicted_top5_probs = top5_vals.cpu().tolist()

        # --- Attention matrices (optional full save) ---
        attention_matrices = None
        if save_attention_matrices:
            attention_matrices = [attn[0].cpu().float().numpy() for attn in attentions]

        # --- Per-token extraction ---
        token_extractions = []

        for pos in range(seq_len):
            # Stack residual streams across all layers for this token
            residual_stack = np.stack([
                hidden_states[layer_idx][0, pos, :].cpu().float().numpy()
                for layer_idx in range(n_layers + 1)
            ])  # shape: (n_layers+1, d_model)

            # Norms
            residual_norms = np.linalg.norm(residual_stack, axis=1)  # (n_layers+1,)

            # Deltas (L2 distance between consecutive layers)
            residual_deltas = np.array([
                np.linalg.norm(residual_stack[i+1] - residual_stack[i])
                for i in range(n_layers)
            ])  # (n_layers,)

            # Cosine similarity between consecutive layers
            layer_cosines = np.array([
                np.dot(residual_stack[i], residual_stack[i+1]) /
                (np.linalg.norm(residual_stack[i]) * np.linalg.norm(residual_stack[i+1]) + 1e-10)
                for i in range(n_layers)
            ])  # (n_layers,)

            # --- Logits at this position ---
            pos_logits = logits[0, pos, :]  # (vocab_size,)
            pos_probs = torch.softmax(pos_logits, dim=-1)

            # Top-k predictions at this position
            topk_vals, topk_ids = torch.topk(pos_probs, k=min(top_k, vocab_size))
            topk_ids_np = topk_ids.cpu().numpy()
            topk_vals_np = topk_vals.cpu().float().numpy()
            topk_tokens = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in topk_ids_np]

            # Entropy of prediction distribution
            pos_probs_np = pos_probs.cpu().float().numpy()
            # Clip for numerical stability
            pos_probs_clipped = np.clip(pos_probs_np, 1e-10, 1.0)
            prediction_entropy = -np.sum(pos_probs_clipped * np.log2(pos_probs_clipped))

            # Next token probability and rank (if not last position)
            next_token_prob = 0.0
            next_token_rank = -1
            if pos < seq_len - 1:
                actual_next_id = input_ids[pos + 1]
                next_token_prob = pos_probs[actual_next_id].item()
                # Rank: how many tokens have higher probability
                next_token_rank = (pos_probs > pos_probs[actual_next_id]).sum().item()

            # --- Attention patterns for this token ---
            # attention_to_self[layer, head] = sum of attention FROM all tokens TO this position
            # attention_from_self[layer, head] = entropy of this token's attention distribution
            # attention_max_source[layer, head] = which position this token attends to most

            attention_to_self = np.zeros((n_layers, n_heads), dtype=np.float32)
            attention_from_self = np.zeros((n_layers, n_heads), dtype=np.float32)
            attention_max_source = np.zeros((n_layers, n_heads), dtype=np.int32)

            for layer_idx in range(n_layers):
                attn = attentions[layer_idx][0]  # (n_heads, seq_len, seq_len)

                for head_idx in range(n_heads):
                    # How much all tokens attend TO this position (column sum)
                    attn_col = attn[head_idx, :, pos].cpu().float().numpy()
                    attention_to_self[layer_idx, head_idx] = attn_col.sum()

                    # This token's attention distribution (row) — compute entropy
                    attn_row = attn[head_idx, pos, :pos+1].cpu().float().numpy()  # causal: only attend to past
                    attn_row_clipped = np.clip(attn_row, 1e-10, 1.0)
                    attn_entropy = -np.sum(attn_row_clipped * np.log2(attn_row_clipped))
                    attention_from_self[layer_idx, head_idx] = attn_entropy

                    # Which position this token attends to most
                    attention_max_source[layer_idx, head_idx] = attn_row.argmax()

            token_extractions.append(TokenFullExtraction(
                prompt_idx=prompt_idx,
                token_pos=pos,
                token_id=input_ids[pos],
                token_text=token_texts[pos],
                token_bytes=token_texts[pos].encode('utf-8', errors='replace'),
                is_special=(input_ids[pos] in special_tokens),
                residuals=residual_stack,
                residual_norms=residual_norms,
                residual_deltas=residual_deltas,
                layer_cosines=layer_cosines,
                logits_topk_ids=topk_ids_np,
                logits_topk_values=topk_vals_np,
                logits_topk_tokens=topk_tokens,
                prediction_entropy=prediction_entropy,
                next_token_prob=next_token_prob,
                next_token_rank=next_token_rank,
                attention_to_self=attention_to_self,
                attention_from_self=attention_from_self,
                attention_max_source=attention_max_source,
            ))

        results.append(PromptFullResult(
            prompt_idx=prompt_idx,
            prompt_text=prompt_text,
            predicted_next_token=predicted_next_token,
            predicted_next_token_id=predicted_next_id,
            predicted_next_prob=predicted_next_prob,
            predicted_top5_tokens=predicted_top5_tokens,
            predicted_top5_probs=predicted_top5_probs,
            token_ids=input_ids,
            token_texts=token_texts,
            n_tokens=seq_len,
            inference_time_ms=inference_time_ms,
            attention_matrices=attention_matrices,
            token_extractions=token_extractions,
        ))

        print(f"    [{prompt_idx+1}/{len(prompts)}] "
              f"'{prompt_text[:50]}{'...' if len(prompt_text)>50 else ''}' "
              f"→ '{predicted_next_token}' (p={predicted_next_prob:.4f}) "
              f"[{seq_len} tok, {inference_time_ms:.1f}ms]")
        print(f"      Top-5: {list(zip(predicted_top5_tokens, [f'{p:.4f}' for p in predicted_top5_probs]))}")

    return results


# =============================================================================
# FULL CSV/JSON Output — Save EVERYTHING
# =============================================================================

def save_everything(
    output_dir: Path,
    group_name: str,
    group_data: dict,
    results: list[PromptFullResult],
    model_info: dict,
    top_k: int = 20,
):
    """
    Save ALL extracted data in organized directory structure.
    
    Output structure:
      output_dir/
        group_name/
          model_info.json                    — full model metadata
          group_info.json                    — group metadata + all predictions
          prompts_meta.csv                   — prompt texts, predictions, timing
          
          residual_streams/
            all_layers_all_tokens.csv        — FULL residual data (layer, prompt, pos, token, dims...)
            layer_000.csv ... layer_NNN.csv  — per-layer residual CSVs
            
          residual_norms/
            norms_all.csv                    — L2 norms per token per layer
            deltas_all.csv                   — L2 deltas between layers
                        cosines_all.csv                  — cosine similarity between consecutive layers

          predictions/
            logits_topk_all_positions.csv    — top-k predictions at EVERY token position
            entropy_all.csv                  — prediction entropy per token per position
            next_token_accuracy.csv          — probability & rank of actual next token

          attention/
            attention_to_self.csv            — how much each head attends TO each token (per layer)
            attention_from_self_entropy.csv  — entropy of each token's attention distribution
            attention_max_source.csv         — which token each position attends to most
            full_matrices/                   — (optional) full attention matrices per layer
              layer_000.npy ... layer_NNN.npy

          final_token_streams/
            all_layers_all_prompts.csv       — backward compat: last token only
            layer_000.csv ... layer_NNN.csv

          all_token_streams/
            all_layers_all_tokens.csv        — FULL residual data
            layer_000.csv ... layer_NNN.csv
    """
    group_dir = output_dir / group_name
    residual_dir = group_dir / "residual_streams"
    norms_dir = group_dir / "residual_norms"
    predictions_dir = group_dir / "predictions"
    attention_dir = group_dir / "attention"
    final_token_dir = group_dir / "final_token_streams"
    all_token_dir = group_dir / "all_token_streams"

    for d in [residual_dir, norms_dir, predictions_dir, attention_dir,
              final_token_dir, all_token_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if not results:
        print(f"  WARNING: No results for group '{group_name}'")
        return

    n_layers = model_info["n_layers"] + 1  # +1 for embedding layer
    d_model = model_info["d_model"]
    n_heads = model_info["n_heads"]

    dim_headers = [f"dim_{i:04d}" for i in range(d_model)]

    # =========================================================================
    # 1. MODEL INFO JSON (once per group, includes full config)
    # =========================================================================
    model_info_path = group_dir / "model_info.json"
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False, default=str)
    print(f"    Saved: {model_info_path}")

    # =========================================================================
    # 2. GROUP INFO JSON (metadata + all predictions + timing)
    # =========================================================================
    group_info = {
        "group_name": group_name,
        "description": group_data.get("description", ""),
        "expected_answer": group_data.get("expected_answer", ""),
        "category": group_data.get("category", ""),
        "n_prompts": len(results),
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "total_tokens": sum(r.n_tokens for r in results),
        "prompts": [r.prompt_text for r in results],
        "predictions": [r.predicted_next_token for r in results],
        "prediction_probs": [r.predicted_next_prob for r in results],
        "top5_predictions": [r.predicted_top5_tokens for r in results],
        "top5_probs": [r.predicted_top5_probs for r in results],
        "token_counts": [r.n_tokens for r in results],
        "inference_times_ms": [r.inference_time_ms for r in results],
        "total_inference_time_ms": sum(r.inference_time_ms for r in results),
        "token_ids_per_prompt": [r.token_ids for r in results],
        "token_texts_per_prompt": [r.token_texts for r in results],
    }
    info_path = group_dir / "group_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(group_info, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {info_path}")

    # =========================================================================
    # 3. PROMPTS META CSV (rich per-prompt summary)
    # =========================================================================
    meta_path = group_dir / "prompts_meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_idx", "prompt", "predicted_next_token", "predicted_next_token_id",
            "predicted_next_prob", "top5_tokens", "top5_probs",
            "n_tokens", "tokens", "token_ids", "inference_time_ms"
        ])
        for r in results:
            writer.writerow([
                r.prompt_idx,
                r.prompt_text,
                r.predicted_next_token,
                r.predicted_next_token_id,
                f"{r.predicted_next_prob:.6f}",
                "|".join(r.predicted_top5_tokens),
                "|".join(f"{p:.6f}" for p in r.predicted_top5_probs),
                r.n_tokens,
                "|".join(r.token_texts),
                "|".join(str(tid) for tid in r.token_ids),
                f"{r.inference_time_ms:.2f}",
            ])
    print(f"    Saved: {meta_path}")

    # =========================================================================
    # 4. RESIDUAL STREAMS — all tokens, all layers (the big one)
    # =========================================================================
    all_csv_path = all_token_dir / "all_layers_all_tokens.csv"
    with open(all_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "prompt_idx", "token_pos", "token_text"] + dim_headers)

        for layer_idx in range(n_layers):
            for r in results:
                for tok in r.token_extractions:
                    row = [
                        layer_idx,
                        tok.prompt_idx,
                        tok.token_pos,
                        tok.token_text,
                    ] + tok.residuals[layer_idx].tolist()
                    writer.writerow(row)

    total_tokens = sum(len(r.token_extractions) for r in results)
    print(f"    Saved: {all_csv_path} "
          f"({n_layers} layers × {total_tokens} tokens × {d_model} dims)")

    # Per-layer CSVs (all tokens)
    for layer_idx in range(n_layers):
        layer_path = all_token_dir / f"layer_{layer_idx:03d}.csv"
        with open(layer_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt_idx", "token_pos", "token_text"] + dim_headers)
            for r in results:
                for tok in r.token_extractions:
                    row = [
                        tok.prompt_idx,
                        tok.token_pos,
                        tok.token_text,
                    ] + tok.residuals[layer_idx].tolist()
                    writer.writerow(row)
    print(f"    Saved: {n_layers} per-layer CSVs in {all_token_dir}")

    # =========================================================================
    # 5. FINAL TOKEN STREAMS (backward compat with viewer.py)
    # =========================================================================
    final_csv_path = final_token_dir / "all_layers_all_prompts.csv"
    with open(final_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "prompt_idx"] + dim_headers)
        for layer_idx in range(n_layers):
            for r in results:
                last_tok = r.token_extractions[-1]
                row = [layer_idx, r.prompt_idx] + last_tok.residuals[layer_idx].tolist()
                writer.writerow(row)
    print(f"    Saved: {final_csv_path}")

    for layer_idx in range(n_layers):
        layer_path = final_token_dir / f"layer_{layer_idx:03d}.csv"
        with open(layer_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["prompt_idx"] + dim_headers)
            for r in results:
                last_tok = r.token_extractions[-1]
                row = [r.prompt_idx] + last_tok.residuals[layer_idx].tolist()
                writer.writerow(row)
    print(f"    Saved: {n_layers} final-token per-layer CSVs in {final_token_dir}")

    # =========================================================================
    # 6. RESIDUAL NORMS — L2 norms per token per layer
    # =========================================================================
    norms_path = norms_dir / "norms_all.csv"
    layer_headers = [f"layer_{i:03d}" for i in range(n_layers)]
    with open(norms_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "token_pos", "token_text"] + layer_headers)
        for r in results:
            for tok in r.token_extractions:
                row = [tok.prompt_idx, tok.token_pos, tok.token_text] + tok.residual_norms.tolist()
                writer.writerow(row)
    print(f"    Saved: {norms_path}")

    # =========================================================================
    # 7. RESIDUAL DELTAS — L2 distance between consecutive layers
    # =========================================================================
    delta_headers = [f"delta_{i:03d}_to_{i+1:03d}" for i in range(n_layers - 1)]
    deltas_path = norms_dir / "deltas_all.csv"
    with open(deltas_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "token_pos", "token_text"] + delta_headers)
        for r in results:
            for tok in r.token_extractions:
                row = [tok.prompt_idx, tok.token_pos, tok.token_text] + tok.residual_deltas.tolist()
                writer.writerow(row)
    print(f"    Saved: {deltas_path}")

    # =========================================================================
    # 8. COSINE SIMILARITIES between consecutive layers
    # =========================================================================
    cosine_headers = [f"cos_{i:03d}_to_{i+1:03d}" for i in range(n_layers - 1)]
    cosines_path = norms_dir / "cosines_all.csv"
    with open(cosines_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "token_pos", "token_text"] + cosine_headers)
        for r in results:
            for tok in r.token_extractions:
                row = [tok.prompt_idx, tok.token_pos, tok.token_text] + tok.layer_cosines.tolist()
                writer.writerow(row)
    print(f"    Saved: {cosines_path}")

    # =========================================================================
    # 9. PREDICTIONS — top-k logits at EVERY token position
    # =========================================================================
    topk_path = predictions_dir / "logits_topk_all_positions.csv"
    with open(topk_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Headers: for each k, save token and probability
        topk_headers = []
        for k in range(top_k):
            topk_headers.extend([f"top{k}_token", f"top{k}_prob"])
        writer.writerow(["prompt_idx", "token_pos", "token_text", "entropy"] + topk_headers)

        for r in results:
            for tok in r.token_extractions:
                row = [tok.prompt_idx, tok.token_pos, tok.token_text, f"{tok.prediction_entropy:.4f}"]
                for k in range(top_k):
                    if k < len(tok.logits_topk_tokens):
                        row.extend([tok.logits_topk_tokens[k], f"{tok.logits_topk_values[k]:.6f}"])
                    else:
                        row.extend(["", "0.0"])
                writer.writerow(row)
    print(f"    Saved: {topk_path} (top-{top_k} at every position)")

    # =========================================================================
    # 10. ENTROPY — prediction entropy per token position
    # =========================================================================
    entropy_path = predictions_dir / "entropy_all.csv"
    with open(entropy_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "token_pos", "token_text", "entropy_bits"])
        for r in results:
            for tok in r.token_extractions:
                writer.writerow([tok.prompt_idx, tok.token_pos, tok.token_text,
                               f"{tok.prediction_entropy:.4f}"])
    print(f"    Saved: {entropy_path}")

    # =========================================================================
    # 11. NEXT TOKEN ACCURACY — how well model predicts actual next token
    # =========================================================================
    accuracy_path = predictions_dir / "next_token_accuracy.csv"
    with open(accuracy_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "token_pos", "token_text", "actual_next_token",
                        "next_token_prob", "next_token_rank"])
        for r in results:
            for tok in r.token_extractions:
                # Get actual next token text
                actual_next = ""
                if tok.token_pos < r.n_tokens - 1:
                    actual_next = r.token_texts[tok.token_pos + 1]
                writer.writerow([
                    tok.prompt_idx, tok.token_pos, tok.token_text,
                    actual_next,
                    f"{tok.next_token_prob:.6f}",
                    tok.next_token_rank,
                ])
    print(f"    Saved: {accuracy_path}")

    # =========================================================================
    # 12. ATTENTION — who attends to whom (summary stats per token)
    # =========================================================================
    # 12a. Attention TO self (how much each head attends TO this token position)
    attn_to_path = attention_dir / "attention_to_self.csv"
    attn_headers = [f"L{l:02d}_H{h:02d}" for l in range(n_layers - 1) for h in range(n_heads)]
    with open(attn_to_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "token_pos", "token_text"] + attn_headers)
        for r in results:
            for tok in r.token_extractions:
                row = [tok.prompt_idx, tok.token_pos, tok.token_text]
                row += tok.attention_to_self.flatten().tolist()
                writer.writerow(row)
    print(f"    Saved: {attn_to_path}")

    # 12b. Attention FROM self (entropy of this token's attention distribution)
    attn_from_path = attention_dir / "attention_from_self_entropy.csv"
    with open(attn_from_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "token_pos", "token_text"] + attn_headers)
        for r in results:
            for tok in r.token_extractions:
                row = [tok.prompt_idx, tok.token_pos, tok.token_text]
                row += tok.attention_from_self.flatten().tolist()
                writer.writerow(row)
    print(f"    Saved: {attn_from_path}")

    # 12c. Attention max source (which position this token attends to most)
    attn_max_path = attention_dir / "attention_max_source.csv"
    with open(attn_max_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_idx", "token_pos", "token_text"] + attn_headers)
        for r in results:
            for tok in r.token_extractions:
                row = [tok.prompt_idx, tok.token_pos, tok.token_text]
                row += tok.attention_max_source.flatten().tolist()
                writer.writerow(row)
    print(f"    Saved: {attn_max_path}")

    # =========================================================================
    # 13. FULL ATTENTION MATRICES (optional, very large)
    # =========================================================================
    if results[0].attention_matrices is not None:
        attn_full_dir = attention_dir / "full_matrices"
        attn_full_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            prompt_dir = attn_full_dir / f"prompt_{r.prompt_idx:03d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            for layer_idx, attn_matrix in enumerate(r.attention_matrices):
                np.save(prompt_dir / f"layer_{layer_idx:03d}.npy", attn_matrix)

        print(f"    Saved: full attention matrices in {attn_full_dir} "
              f"({len(results)} prompts × {n_layers-1} layers)")

    # =========================================================================
    # 14. SUMMARY STATISTICS CSV — per-prompt aggregate stats
    # =========================================================================
    summary_path = group_dir / "prompt_summary_stats.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_idx", "prompt", "predicted_next_token", "predicted_prob",
            "n_tokens", "inference_time_ms",
            "mean_entropy", "final_entropy",
            "mean_norm_first_layer", "mean_norm_last_layer",
            "mean_delta_early", "mean_delta_late",
            "mean_cosine_early", "mean_cosine_late",
            "total_path_length",
        ])
        for r in results:
            entropies = [tok.prediction_entropy for tok in r.token_extractions]
            norms_first = [tok.residual_norms[0] for tok in r.token_extractions]
            norms_last = [tok.residual_norms[-1] for tok in r.token_extractions]

            # Early = first half of layers, late = second half
            n_deltas = len(r.token_extractions[0].residual_deltas)
            mid = n_deltas // 2

            deltas_early = [tok.residual_deltas[:mid].mean() for tok in r.token_extractions]
            deltas_late = [tok.residual_deltas[mid:].mean() for tok in r.token_extractions]
            cosines_early = [tok.layer_cosines[:mid].mean() for tok in r.token_extractions]
            cosines_late = [tok.layer_cosines[mid:].mean() for tok in r.token_extractions]

            # Total path length (sum of deltas across all layers for last token)
            last_tok = r.token_extractions[-1]
            total_path = last_tok.residual_deltas.sum()

            writer.writerow([
                r.prompt_idx,
                r.prompt_text,
                r.predicted_next_token,
                f"{r.predicted_next_prob:.6f}",
                r.n_tokens,
                f"{r.inference_time_ms:.2f}",
                f"{np.mean(entropies):.4f}",
                f"{entropies[-1]:.4f}",
                f"{np.mean(norms_first):.4f}",
                f"{np.mean(norms_last):.4f}",
                f"{np.mean(deltas_early):.4f}",
                f"{np.mean(deltas_late):.4f}",
                f"{np.mean(cosines_early):.6f}",
                f"{np.mean(cosines_late):.6f}",
                f"{total_path:.4f}",
            ])
    print(f"    Saved: {summary_path}")

    # =========================================================================
    # 15. CONVERGENCE ANALYSIS — inter-prompt distances at each layer
    # =========================================================================
    convergence_path = group_dir / "convergence_analysis.csv"
    with open(convergence_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "mean_pairwise_distance_final_token",
            "std_pairwise_distance_final_token",
            "centroid_norm",
            "mean_distance_to_centroid",
            "max_distance_to_centroid",
        ])

        for layer_idx in range(n_layers):
            # Get final-token residuals at this layer for all prompts
            final_vecs = np.stack([
                r.token_extractions[-1].residuals[layer_idx] for r in results
            ])  # shape: (n_prompts, d_model)

            # Pairwise distances
            n_prompts = len(results)
            dists = []
            for i in range(n_prompts):
                for j in range(i + 1, n_prompts):
                    dists.append(np.linalg.norm(final_vecs[i] - final_vecs[j]))

            dists = np.array(dists) if dists else np.array([0.0])

            # Centroid
            centroid = final_vecs.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            dists_to_centroid = np.array([
                np.linalg.norm(v - centroid) for v in final_vecs
            ])

            writer.writerow([
                layer_idx,
                f"{dists.mean():.6f}",
                f"{dists.std():.6f}",
                f"{centroid_norm:.6f}",
                f"{dists_to_centroid.mean():.6f}",
                f"{dists_to_centroid.max():.6f}",
            ])
    print(f"    Saved: {convergence_path}")

    # Print size summary
    print(f"\n    === Size summary for {group_name} ===")
    total_size = 0
    for root, dirs, files in os.walk(group_dir):
        for file in files:
            fpath = Path(root) / file
            size = fpath.stat().st_size
            total_size += size
            if size > 1024 * 1024:
                print(f"      {fpath.relative_to(group_dir)}: {size / (1024*1024):.1f} MB")
    print(f"      TOTAL: {total_size / (1024*1024):.1f} MB")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract EVERYTHING from residual streams for attractor analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run extract.py results/
  uv run extract.py results/ --model gpt2-large
  uv run extract.py results/ --model deepseek --groups capital_berlin_multilingual
  uv run extract.py results/ --model gpt2 --top-k 50 --save-attention
  uv run extract.py results/ --model gpt2-medium --groups sentiment_positive sentiment_negative

Available groups:
""" + "\n".join(f"  {k}: {v['description']} ({len(v['prompts'])} prompts)"
               for k, v in PROMPT_GROUPS.items())
    )

    parser.add_argument("output_dir", type=str,
                       help="Output directory for all extracted data")
    parser.add_argument("--model", type=str, default="gpt2",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to use (default: gpt2)")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Override model name/path")
    parser.add_argument("--groups", type=str, nargs="*", default=None,
                       help="Specific prompt groups to run (default: all)")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Number of top predictions to save per position (default: 20)")
    parser.add_argument("--save-attention", action="store_true",
                       help="Save full attention matrices as .npy (WARNING: very large)")
    parser.add_argument("--dims", type=int, default=None,
                       help="Limit output dimensions (default: all)")

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
    print("  RESIDUAL STREAM FULL EXTRACTOR — EXTRACT EVERYTHING")
    print("=" * 70)
    print(f"  Model:           {args.model}" + (f" ({args.model_name})" if args.model_name else ""))
    print(f"  Output:          {output_dir}")
    print(f"  Groups:          {len(groups_to_run)}")
    print(f"  Top-k:           {args.top_k}")
    print(f"  Save attention:  {args.save_attention}")
    print(f"  Dims limit:      {args.dims or 'all'}")
    total_prompts = sum(len(g["prompts"]) for g in groups_to_run.values())
    print(f"  Total prompts:   {total_prompts}")
    print("=" * 70)

    # Load model
    model, tokenizer, model_info = load_model(args.model, args.model_name)

    # Process each group
    for group_idx, (group_name, group_data) in enumerate(groups_to_run.items()):
        prompts = group_data["prompts"]
        print(f"\n{'─' * 60}")
        print(f"  Group [{group_idx+1}/{len(groups_to_run)}]: {group_name}")
        print(f"  Description: {group_data['description']}")
        print(f"  Expected answer: {group_data.get('expected_answer', 'N/A')}")
        print(f"  Category: {group_data.get('category', 'N/A')}")
        print(f"  Prompts: {len(prompts)}")
        print(f"{'─' * 60}")

        # Extract EVERYTHING
        results = extract_everything(
            model, tokenizer, prompts, model_info,
            top_k=args.top_k,
            save_attention_matrices=args.save_attention,
        )

        # Optionally truncate dimensions in residuals
        if args.dims:
            for r in results:
                for tok in r.token_extractions:
                    tok.residuals = tok.residuals[:, :args.dims]

        # Save EVERYTHING
        save_everything(
            output_dir, group_name, group_data, results, model_info,
            top_k=args.top_k,
        )

    # =========================================================================
    # Final Summary
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Output directory: {output_dir}")
    print(f"  Groups extracted: {len(groups_to_run)}")
    print(f"  Model: {model_info['model_name']} ({model_info['n_params_M']}M params)")
    print(f"  Architecture: {model_info['architecture']}")
    print()

    # Total size
    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            total_size += (Path(root) / file).stat().st_size
    print(f"  Total output size: {total_size / (1024*1024):.1f} MB")
    print()

    # Per-group summary
    print(f"  Per-group summary:")
    for group_name in groups_to_run:
        gdir = output_dir / group_name
        if gdir.exists():
            gsize = sum(
                f.stat().st_size
                for f in gdir.rglob("*") if f.is_file()
            )
            print(f"    {group_name}: {gsize / (1024*1024):.1f} MB")

    print(f"\n  Directory structure:")
    for group_name in list(groups_to_run.keys())[:3]:
        gdir = output_dir / group_name
        if gdir.exists():
            print(f"    {group_name}/")
            for item in sorted(gdir.iterdir()):
                if item.is_file():
                    print(f"      {item.name} ({item.stat().st_size / 1024:.0f} KB)")
                elif item.is_dir():
                    n_files = len(list(item.rglob("*")))
                    dir_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    print(f"      {item.name}/ ({n_files} files, {dir_size / (1024*1024):.1f} MB)")

    print(f"\n  To visualize:")
    print(f"    uv run viewer.py {output_dir}")
    print(f"    uv run viewer.py {output_dir}/{list(groups_to_run.keys())[0]}")
    print()
    print(f"  Data files saved per group:")
    print(f"    model_info.json              — full model config & hardware info")
    print(f"    group_info.json              — prompts, predictions, top-5, timing, token IDs")
    print(f"    prompts_meta.csv             — per-prompt summary with top-5 and timing")
    print(f"    prompt_summary_stats.csv     — aggregate stats (entropy, norms, path length)")
    print(f"    convergence_analysis.csv     — inter-prompt distances per layer")
    print(f"    residual_streams/            — full residual vectors (all tokens × all layers)")
    print(f"    residual_norms/              — L2 norms, deltas, cosine similarities")
    print(f"    predictions/                 — top-k logits, entropy, next-token accuracy")
    print(f"    attention/                   — attention-to-self, entropy, max-source")
    if args.save_attention:
        print(f"    attention/full_matrices/  — full .npy attention matrices per prompt per layer")
    print(f"    final_token_streams/         — backward-compat (last token only)")
    print(f"    all_token_streams/           — backward-compat (all tokens, residuals only)")


if __name__ == "__main__":
    main()
