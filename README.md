# Residual Stream Attractor Extraction Tool

## Hypothesis

In large language models (LLMs), dynamic attractors exist in the residual stream: points in high-dimensional space toward which hidden states converge in later layers, regardless of the linguistic or contextual direction from which they approach.

Example: The prompts

- "The capital of France is"
- "Die Hauptstadt von Frankreich ist"
- "The Eiffel Tower is located in"
- "The Louvre museum is in"

should all predict the token "Paris". The hypothesis is that their residual stream vectors at the final token position are pulled toward a common point (the "Paris attractor") in later layers — no matter how different the input representations are in early layers.

Similarly, prompts like "Die Hauptstadt von Deutschland ist" and "Das Brandenburger Tor steht in" should both converge toward a "Berlin attractor", even though they approach from completely different semantic regions.

If this is true, it implies that LLM inference in later layers functions like a dynamical system with basins of attraction — and that these attractors can be extracted, catalogued, and potentially used to understand or steer model behavior.


## What This Tool Does

1. Loads DeepSeek (or any compatible HuggingFace causal LM) locally
2. Runs predefined prompt groups through the model — each group contains sentences in different languages and framings that should all predict the same next token
3. Captures the residual stream at every layer boundary, specifically at the final token position (the one that predicts the next token)
4. Computes attractor metrics — centroid (potential attractor point), distances, cosine similarities, convergence trajectories across layers
5. Compares groups — within-group vs. between-group distances to verify cluster separation
6. Saves everything as .npy and .json for downstream analysis, visualization, and reconstruction


## Predefined Prompt Groups

| Group | Target Token | Languages/Variants |
|-------|-------------|-------------------|
| paris_multilingual | Paris | EN, DE, FR, ES, JP + landmarks |
| berlin_multilingual | Berlin | EN, DE, FR, ES, JP + landmarks |
| tokyo_multilingual | Tokyo | EN, DE, FR, JP + landmarks |
| london_multilingual | London | EN, DE, FR + landmarks |
| four_arithmetic | 4 | arithmetic, language, facts |
| water_concept | water | chemistry, everyday, nature |
| sun_concept | Sun | astronomy, physics, multilingual |
| einstein_person | Einstein | physics, history, multilingual |

Each group contains prompts that approach the same concept from different angles — different languages, different factual framings, different contextual setups — all converging on the same predicted token.


## Installation and Usage

### Prerequisites

- Python 3.11+
- uv (used for dependency management via inline script metadata)
- GPU with sufficient VRAM for DeepSeek-7B (~14GB in float16) — or CPU (slow but works)

### Running

Option 1 — directly with uv:

```
    uv run residual_attractors.py
```

Option 2 — directly with python (auto-bootstraps to uv):

```
    python3 residual_attractors.py
```

With a different model:

```
    uv run residual_attractors.py --model deepseek-ai/deepseek-llm-7b-base --device cuda
```

Only specific groups:

```
    uv run residual_attractors.py --groups paris_multilingual berlin_multilingual
```

On CPU with float32 (slow, but no GPU required):

```
    uv run residual_attractors.py --device cpu --dtype float32
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| --model | deepseek-ai/deepseek-llm-7b-base | HuggingFace model ID |
| --device | auto | cuda, cpu, or auto |
| --output | attractor_data | Output directory |
| --groups | all | Which groups to run (by name) |
| --dtype | float16 | float16, bfloat16, float32 |
| --max-layers | all | Only capture first N layers (memory-constrained setups) |


## Output Structure

```
    attractor_data/
    +-- paris_multilingual/
    |   +-- prompt_000_final_token_streams.npy   # shape: (n_layers, d_model)
    |   +-- prompt_000_meta.json                 # prompt, prediction, target match
    |   +-- prompt_001_final_token_streams.npy
    |   +-- prompt_001_meta.json
    |   +-- ...
    |   +-- centroids_per_layer.npy              # shape: (n_layers, d_model) — the attractor per layer
    |   +-- metrics.json                         # convergence metrics
    +-- berlin_multilingual/
    |   +-- ...
    |   +-- centroids_per_layer.npy
    |   +-- metrics.json
    +-- ...
    +-- cross_group_comparison.json              # between-group distances and cosines
```

## Loading and Analyzing Data

```python
    import numpy as np
    import json

    # Load residual streams for a single prompt
    streams = np.load("attractor_data/paris_multilingual/prompt_000_final_token_streams.npy")
    # Shape: (n_layers, d_model) — residual stream at final token position per layer

    # Load centroid (= attractor candidate) per layer
    centroids = np.load("attractor_data/paris_multilingual/centroids_per_layer.npy")
    # Shape: (n_layers, d_model)

    # Load convergence metrics
    with open("attractor_data/paris_multilingual/metrics.json") as f:
        metrics = json.load(f)

    # Inspect convergence trajectory
    print("Cosine similarity trajectory (should increase in later layers):")
    for i, cos_sim in enumerate(metrics["convergence"]["cosine_trajectory"]):
        print(f"  Layer {i:3d}: {cos_sim:.4f}")
```

## Convergence Metrics

For each group and each layer, the following metrics are computed:

| Metric | Meaning |
|--------|---------|
| mean_distance_to_centroid | How far are the points on average from the attractor? |
| mean_pairwise_cosine_sim | How similar are the directions of the residual streams? |
| distance_ratio_last_vs_first | Ratio of last-layer to first-layer distance (less than 1 = convergence) |
| cosine_improvement | Cosine similarity gain from first to last layer |

If the attractor hypothesis holds:

- distance_ratio_last_vs_first should be significantly less than 1
- cosine_improvement should be positive
- mean_pairwise_cosine_sim should approach 1 in later layers
- Between-group distances should remain large (different attractors are well-separated)


## How It Works Internally

### Residual Stream Extraction

The tool registers PyTorch forward hooks on every transformer layer. After each layer processes the input, the hook captures the hidden state tensor. We extract specifically the vector at the final sequence position — this is the representation that the model uses to predict the next token.

### Attractor Analysis

For each prompt group at each layer:
1. Collect all final-token residual stream vectors (one per prompt in the group)
2. Compute the centroid (mean vector) — this is the attractor candidate
3. Measure how tightly the points cluster around the centroid (L2 distances)
4. Measure directional alignment (pairwise cosine similarities)
5. Track how these metrics evolve from early to late layers

### Cross-Group Comparison

At the final layer, compute distances and cosine similarities between centroids of different groups. If attractors are real and distinct, within-group distances should be small while between-group distances should be large.


## Next Steps

1. Visualization: Apply PCA or UMAP to the final-layer streams across all groups to visualize cluster formation in 2D/3D
2. Centroid-to-token reconstruction: Feed the centroid vector through the model's LM head (unembedding layer) and check whether it decodes to the correct target token
3. Layer-wise animation: Visualize how points migrate from scattered positions in early layers toward the attractor in later layers
4. New groups: Add a PromptGroup object to ATTRACTOR_GROUPS — no other changes needed
5. Cross-model comparison: Run the same prompts through different models (DeepSeek, LLaMA, Mistral) and compare attractor geometry
6. Attractor arithmetic: Test whether attractor centroids compose linearly (e.g., does the "Berlin" attractor minus the "Germany" direction plus the "France" direction approximate the "Paris" attractor?)
7. Causal intervention: Inject an attractor centroid into the residual stream at a middle layer and verify that the model then predicts the corresponding token


## Bootstrapping

The script uses the same auto-bootstrapping mechanism as grok.py: if invoked directly with python3 instead of uv run, it detects this, locates the uv binary, and re-execs itself under uv run with the inline-declared dependencies. No manual pip install required — just run the script and dependencies are resolved automatically.


## Theoretical Background

### Residual Stream as Dynamical System

Each transformer layer can be viewed as one step of a discrete dynamical system operating on the residual stream vector:

    x_{l+1} = x_l + f_l(x_l)

where f_l is the combined attention + MLP update at layer l. The attractor hypothesis posits that for prompts requiring the same next-token prediction, the trajectories x_0, x_1, ..., x_L converge to a common fixed region regardless of x_0.

### Why This Might Work

- The unembedding matrix W_U maps the final residual stream to logits. For the model to predict token t, the final residual stream must have high dot product with the t-th row of W_U.
- If multiple prompts must predict the same token, their final residual streams must all point in a similar direction (at minimum, they must all have high projection onto the same W_U row).
- The question is whether this convergence happens gradually across layers (attractor dynamics) or only at the very last layer.

### Related Work

- Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021) — formalizes the residual stream
- Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability" (2023) — Fourier structure in residual streams
- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (2021) — MLP layers as memory lookup
- Dar et al., "Analyzing Transformers in Embedding Space" (2022) — interpreting intermediate representations via the unembedding


## License

Research tool. Use freely for mechanistic interpretability research.
