# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "flask",
# ]
# ///
"""
Attractor Viewer — Browser-based Token Trajectory Analysis

Detects and visualizes attractors in residual stream data:
  - Fixed point attractors (convergence to single point)
  - Limit cycles (periodic orbits)
  - Torus attractors (quasi-periodic)
  - Strange/chaotic attractors (positive Lyapunov exponent, fractal dimension)

Runs a local web server, opens browser automatically.
All computation from raw CSV data, no PCA by default.

Usage:
    python viewer.py attractor_data/
    python viewer.py attractor_data/berlin_multilingual
    python viewer.py attractor_data/ --port 8899
"""

import sys
import os
import shutil
import subprocess
import signal
import argparse
import json
import csv
import threading
import webbrowser
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# Auto-restart under `uv run`
# =============================================================================

def _ensure_uv_run():
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

signal.signal(signal.SIGINT, lambda *_: os._exit(0))

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from flask import Flask, jsonify, send_from_directory, request

# =============================================================================
# Data Loading
# =============================================================================

class DataLoader:
    """Loads token trajectory data from CSV files dynamically."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.groups = self._discover_groups()
        print(f"Discovered {len(self.groups)} group(s): {list(self.groups.keys())}")

    def _discover_groups(self) -> dict:
        """Find all available groups in the data directory."""
        groups = {}
        data_dir = self.data_dir

        # Check if this IS a group directory
        if (data_dir / "all_token_streams").is_dir() or (data_dir / "final_token_streams").is_dir():
            groups[data_dir.name] = data_dir
        else:
            # Look for subdirectories that are groups
            for d in sorted(data_dir.iterdir()):
                if d.is_dir() and d.name != "visualizations":
                    if (d / "all_token_streams").is_dir() or (d / "final_token_streams").is_dir():
                        groups[d.name] = d
        return groups

    def get_group_info(self, group_name: str) -> dict:
        """Get metadata about a group."""
        group_dir = self.groups[group_name]
        info = {"name": group_name, "layers": 0, "dims": 0, "tokens": 0}

        # Try group_info.json
        info_json = group_dir / "group_info.json"
        if info_json.exists():
            with open(info_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                info["description"] = data.get("description", "")
                info["n_layers"] = data.get("n_layers", 0)
                info["d_model"] = data.get("d_model", 0)
                info["n_prompts"] = data.get("n_prompts", 0)
                info["total_tokens"] = data.get("total_tokens", 0)
                info["expected_answer"] = data.get("expected_answer", "")
                info["predictions"] = data.get("predictions", [])
                info["prompts"] = data.get("prompts", [])

        # Count layer files
        token_dir = group_dir / "all_token_streams"
        final_dir = group_dir / "final_token_streams"
        if token_dir.is_dir():
            layer_files = list(token_dir.glob("layer_*.csv"))
            info["n_layer_files"] = len(layer_files)
            info["has_all_tokens"] = True
        elif final_dir.is_dir():
            layer_files = list(final_dir.glob("layer_*.csv"))
            info["n_layer_files"] = len(layer_files)
            info["has_all_tokens"] = False

        return info

    def load_trajectories(self, group_name: str, layer_start: int = 0,
                          layer_end: int = -1, token_filter: str = "all") -> dict:
        """
        Load trajectory data for a specific group and layer range.
        
        Returns dict with:
          - trajectories: list of {token_text, prompt_idx, token_pos, points: [[x,y,z,...], ...]}
          - n_layers: int
          - d_model: int
          - layer_range: [start, end]
        """
        group_dir = self.groups[group_name]
        token_dir = group_dir / "all_token_streams"
        final_dir = group_dir / "final_token_streams"

        if token_dir.is_dir():
            return self._load_all_tokens(group_dir, token_dir, layer_start, layer_end, token_filter)
        elif final_dir.is_dir():
            return self._load_final_tokens(group_dir, final_dir, layer_start, layer_end)
        else:
            return {"error": f"No data found for group {group_name}"}

    def _load_all_tokens(self, group_dir, token_dir, layer_start, layer_end, token_filter):
        """Load from all_token_streams."""
        all_csv = token_dir / "all_layers_all_tokens.csv"

        # Read prompts meta
        prompts, predictions = self._read_meta(group_dir)

        token_data = {}  # key: (prompt_idx, token_pos) -> {text, layers: {layer: vec}}
        dim_keys = None

        if all_csv.exists():
            with open(all_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if dim_keys is None:
                        dim_keys = sorted([k for k in row.keys() if k.startswith("dim_")])
                    layer = int(row["layer"])
                    prompt_idx = int(row["prompt_idx"])
                    token_pos = int(row.get("token_pos", 0))
                    token_text = row.get("token_text", f"tok_{token_pos}")

                    key = (prompt_idx, token_pos)
                    if key not in token_data:
                        token_data[key] = {"text": token_text, "prompt_idx": prompt_idx,
                                          "token_pos": token_pos, "layers": {}}
                    token_data[key]["layers"][layer] = [float(row[k]) for k in dim_keys]
        else:
            # Load from per-layer files
            layer_files = sorted(token_dir.glob("layer_*.csv"))
            for layer_idx, layer_file in enumerate(layer_files):
                with open(layer_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if dim_keys is None:
                            dim_keys = sorted([k for k in row.keys() if k.startswith("dim_")])
                        prompt_idx = int(row.get("prompt_idx", 0))
                        token_pos = int(row.get("token_pos", 0))
                        token_text = row.get("token_text", f"tok_{token_pos}")

                        key = (prompt_idx, token_pos)
                        if key not in token_data:
                            token_data[key] = {"text": token_text, "prompt_idx": prompt_idx,
                                              "token_pos": token_pos, "layers": {}}
                        token_data[key]["layers"][layer_idx] = [float(row[k]) for k in dim_keys]

        if not token_data:
            return {"error": "No token data found"}

        # Determine layer range
        all_layers = set()
        for td in token_data.values():
            all_layers.update(td["layers"].keys())
        max_layer = max(all_layers)
        if layer_end < 0:
            layer_end = max_layer

        layer_start = max(0, layer_start)
        layer_end = min(max_layer, layer_end)

        # Filter tokens
        if token_filter == "last":
            # Only last token per prompt
            last_pos = {}
            for (pidx, tpos) in token_data.keys():
                if pidx not in last_pos or tpos > last_pos[pidx]:
                    last_pos[pidx] = tpos
            filtered_keys = [(pidx, last_pos[pidx]) for pidx in last_pos]
        else:
            filtered_keys = list(token_data.keys())

        # Build trajectories
        trajectories = []
        for key in sorted(filtered_keys):
            td = token_data[key]
            points = []
            for layer in range(layer_start, layer_end + 1):
                if layer in td["layers"]:
                    points.append(td["layers"][layer])
                else:
                    points.append([0.0] * len(dim_keys))

            sentence = prompts[td["prompt_idx"]] if td["prompt_idx"] < len(prompts) else ""
            prediction = predictions[td["prompt_idx"]] if td["prompt_idx"] < len(predictions) else ""

            trajectories.append({
                "token_text": td["text"],
                "prompt_idx": td["prompt_idx"],
                "token_pos": td["token_pos"],
                "sentence": sentence,
                "prediction": prediction,
                "points": points,
            })

        return {
            "trajectories": trajectories,
            "n_layers": layer_end - layer_start + 1,
            "total_layers": max_layer + 1,
            "d_model": len(dim_keys) if dim_keys else 0,
            "layer_range": [layer_start, layer_end],
            "n_tokens": len(trajectories),
        }

    def _load_final_tokens(self, group_dir, final_dir, layer_start, layer_end):
        """Load from final_token_streams (one trajectory per prompt)."""
        prompts, predictions = self._read_meta(group_dir)

        all_csv = final_dir / "all_layers_all_prompts.csv"
        trajectories_data = {}
        dim_keys = None

        if all_csv.exists():
            with open(all_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if dim_keys is None:
                        dim_keys = sorted([k for k in row.keys() if k.startswith("dim_")])
                    layer = int(row["layer"])
                    prompt_idx = int(row["prompt_idx"])
                    if prompt_idx not in trajectories_data:
                        trajectories_data[prompt_idx] = {}
                    trajectories_data[prompt_idx][layer] = [float(row[k]) for k in dim_keys]
        else:
            layer_files = sorted(final_dir.glob("layer_*.csv"))
            for layer_idx, layer_file in enumerate(layer_files):
                with open(layer_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if dim_keys is None:
                        # peek
                        temp = csv.DictReader(open(layer_file, "r", encoding="utf-8"))
                        first = next(temp)
                        dim_keys = sorted([k for k in first.keys() if k.startswith("dim_")])
                    for prompt_idx, row in enumerate(reader):
                        if prompt_idx not in trajectories_data:
                            trajectories_data[prompt_idx] = {}
                        trajectories_data[prompt_idx][layer_idx] = [float(row[k]) for k in dim_keys]

        if not trajectories_data:
            return {"error": "No data found"}

        max_layer = max(max(layers.keys()) for layers in trajectories_data.values())
        if layer_end < 0:
            layer_end = max_layer
        layer_start = max(0, layer_start)
        layer_end = min(max_layer, layer_end)

        trajectories = []
        for pidx in sorted(trajectories_data.keys()):
            points = []
            for layer in range(layer_start, layer_end + 1):
                if layer in trajectories_data[pidx]:
                    points.append(trajectories_data[pidx][layer])
                else:
                    points.append([0.0] * len(dim_keys))

            sentence = prompts[pidx] if pidx < len(prompts) else ""
            prediction = predictions[pidx] if pidx < len(predictions) else ""
            token_text = prediction if prediction else f"prompt_{pidx}"

            trajectories.append({
                "token_text": token_text,
                "prompt_idx": pidx,
                "token_pos": pidx,
                "sentence": sentence,
                "prediction": prediction,
                "points": points,
            })

        return {
            "trajectories": trajectories,
            "n_layers": layer_end - layer_start + 1,
            "total_layers": max_layer + 1,
            "d_model": len(dim_keys) if dim_keys else 0,
            "layer_range": [layer_start, layer_end],
            "n_tokens": len(trajectories),
        }

    def _read_meta(self, group_dir: Path):
        meta_csv = group_dir / "prompts_meta.csv"
        prompts = []
        predictions = []
        if meta_csv.exists():
            with open(meta_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompts.append(row.get("prompt", ""))
                    predictions.append(row.get("predicted_next_token", ""))
        return prompts, predictions

# =============================================================================
# Attractor Detection Engine
# =============================================================================

class AttractorDetector:
    """
    Detects attractor types from trajectory data.
    
    Types detected:
      - FIXED_POINT: trajectories converge to a single point
      - LIMIT_CYCLE: trajectories form periodic orbits
      - TORUS: quasi-periodic motion on a torus surface
      - STRANGE: chaotic attractor (positive Lyapunov exponent, fractal dimension)
      - NONE: no clear attractor structure
    """

    @staticmethod
    def analyze(trajectories: list, layer_range: list, dims: list = None) -> dict:
        """
        Analyze trajectories for attractor structure.
        
        Args:
            trajectories: list of trajectory dicts with 'points' field
            layer_range: [start, end] layer indices
            dims: which dimensions to analyze (None = all, or list of indices)
        
        Returns dict with attractor analysis results.
        """
        if not trajectories:
            return {"type": "NONE", "confidence": 0, "details": "No data"}

        # Extract point arrays
        all_points = []
        for traj in trajectories:
            pts = np.array(traj["points"], dtype=np.float64)
            if dims is not None:
                pts = pts[:, dims]
            all_points.append(pts)

        all_points = np.array(all_points)  # shape: (n_traj, n_layers, n_dims)
        n_traj, n_layers, n_dims = all_points.shape

        results = {
            "layer_range": layer_range,
            "n_trajectories": n_traj,
            "n_layers": n_layers,
            "n_dims": n_dims,
            "attractors": [],
        }

        # === Analysis at different layer slices ===
        # Analyze final state (last few layers)
        final_points = all_points[:, -1, :]  # shape: (n_traj, n_dims)

        # 1. FIXED POINT detection
        fixed_point_result = AttractorDetector._detect_fixed_point(all_points, final_points)
        results["fixed_point"] = fixed_point_result

        # 2. LIMIT CYCLE detection
        cycle_result = AttractorDetector._detect_limit_cycle(all_points)
        results["limit_cycle"] = cycle_result

        # 3. CONVERGENCE analysis (how trajectories approach attractor)
        convergence = AttractorDetector._analyze_convergence(all_points)
        results["convergence"] = convergence

        # 4. LYAPUNOV EXPONENT estimation (chaos indicator)
        lyapunov = AttractorDetector._estimate_lyapunov(all_points)
        results["lyapunov"] = lyapunov

        # 5. FRACTAL DIMENSION estimation (correlation dimension)
        fractal = AttractorDetector._estimate_fractal_dimension(final_points)
        results["fractal_dimension"] = fractal

        # 6. CLUSTERING of final states (multiple attractors?)
        clusters = AttractorDetector._detect_clusters(final_points)
        results["clusters"] = clusters

        # 7. RECURRENCE analysis (for torus/cycle detection)
        recurrence = AttractorDetector._analyze_recurrence(all_points)
        results["recurrence"] = recurrence

        # === Classify attractor type ===
        results["classification"] = AttractorDetector._classify(results)

        return results

    @staticmethod
    def _detect_fixed_point(all_points, final_points):
        """Detect if trajectories converge to fixed point(s)."""
        n_traj, n_layers, n_dims = all_points.shape

        # Compute centroid of final points
        centroid = final_points.mean(axis=0)

        # Distances to centroid
        dists_to_centroid = np.linalg.norm(final_points - centroid, axis=1)
        mean_dist = dists_to_centroid.mean()
        max_dist = dists_to_centroid.max()

        # Compare with spread at earlier layers
        early_points = all_points[:, 0, :]
        early_centroid = early_points.mean(axis=0)
        early_dists = np.linalg.norm(early_points - early_centroid, axis=1)
        early_mean_dist = early_dists.mean()

        # Convergence ratio
        convergence_ratio = mean_dist / (early_mean_dist + 1e-10)

        # Check if last N layers are stable (velocity → 0)
        if n_layers >= 3:
            velocities = np.linalg.norm(np.diff(all_points, axis=1), axis=2)  # (n_traj, n_layers-1)
            late_velocity = velocities[:, -3:].mean() if n_layers > 3 else velocities[:, -1:].mean()
            early_velocity = velocities[:, :3].mean()
            velocity_ratio = late_velocity / (early_velocity + 1e-10)
        else:
            velocity_ratio = 1.0
            late_velocity = 0.0

        is_fixed_point = convergence_ratio < 0.3 and velocity_ratio < 0.2

        return {
            "detected": bool(is_fixed_point),
            "centroid": centroid.tolist(),
            "mean_distance_to_centroid": float(mean_dist),
            "max_distance_to_centroid": float(max_dist),
            "convergence_ratio": float(convergence_ratio),
            "velocity_ratio": float(velocity_ratio),
            "late_mean_velocity": float(late_velocity),
            "confidence": float(max(0, 1 - convergence_ratio) * max(0, 1 - velocity_ratio)),
        }

    @staticmethod
    def _detect_limit_cycle(all_points):
        """Detect periodic orbits (limit cycles)."""
        n_traj, n_layers, n_dims = all_points.shape

        if n_layers < 6:
            return {"detected": False, "reason": "too few layers", "confidence": 0}

        # For each trajectory, check if it returns close to earlier states
        periodicities = []
        for traj_idx in range(min(n_traj, 20)):  # sample
            traj = all_points[traj_idx]  # (n_layers, n_dims)

            # Compute self-distance matrix
            dists = squareform(pdist(traj))

            # Look for recurrence: points that are close to earlier points
            # (excluding immediate neighbors)
            min_period = 3
            recurrence_strengths = []

            for period in range(min_period, n_layers // 2):
                # Check if points at distance 'period' are close
                diag_dists = []
                for i in range(n_layers - period):
                    diag_dists.append(dists[i, i + period])
                mean_recurrence = np.mean(diag_dists)
                # Normalize by typical distance
                typical_dist = np.median(dists[dists > 0])
                recurrence_strengths.append(mean_recurrence / (typical_dist + 1e-10))

            if recurrence_strengths:
                best_period = np.argmin(recurrence_strengths) + min_period
                best_strength = min(recurrence_strengths)
                periodicities.append({
                    "period": int(best_period),
                    "strength": float(best_strength),
                })

        if not periodicities:
            return {"detected": False, "confidence": 0}

        # Consensus period
        periods = [p["period"] for p in periodicities]
        strengths = [p["strength"] for p in periodicities]
        mean_strength = np.mean(strengths)

        # Detected if recurrence is strong (low strength = high recurrence)
        is_cycle = mean_strength < 0.5

        from collections import Counter
        period_counts = Counter(periods)
        dominant_period = period_counts.most_common(1)[0][0] if period_counts else 0

        return {
            "detected": bool(is_cycle),
            "dominant_period": int(dominant_period),
            "mean_recurrence_strength": float(mean_strength),
            "period_distribution": dict(period_counts),
            "confidence": float(max(0, 1 - mean_strength)),
        }

    @staticmethod
    def _analyze_convergence(all_points):
        """Analyze how trajectories converge over layers."""
        n_traj, n_layers, n_dims = all_points.shape

        # Pairwise distances between trajectories at each layer
        layer_spreads = []
        layer_centroids = []

        for layer in range(n_layers):
            pts = all_points[:, layer, :]
            centroid = pts.mean(axis=0)
            dists = np.linalg.norm(pts - centroid, axis=1)
            layer_spreads.append(float(dists.mean()))
            layer_centroids.append(centroid.tolist())

        # Compute convergence rate (exponential fit)
        if len(layer_spreads) > 3:
            log_spreads = np.log(np.array(layer_spreads) + 1e-10)
            x = np.arange(len(log_spreads))
            slope, intercept, r_value, p_value, std_err = linregress(x, log_spreads)
            convergence_rate = float(slope)
        else:
            convergence_rate = 0.0
            r_value = 0.0

        return {
            "layer_spreads": layer_spreads,
            "layer_centroids": layer_centroids,
            "convergence_rate": convergence_rate,
            "convergence_r_squared": float(r_value ** 2) if r_value else 0.0,
            "initial_spread": layer_spreads[0] if layer_spreads else 0,
            "final_spread": layer_spreads[-1] if layer_spreads else 0,
            "spread_ratio": float(layer_spreads[-1] / (layer_spreads[0] + 1e-10)) if layer_spreads else 1.0,
        }

    @staticmethod
    def _estimate_lyapunov(all_points):
        """
        Estimate largest Lyapunov exponent.
        Positive = chaos, negative = convergence, zero = neutral/periodic.
        """
        n_traj, n_layers, n_dims = all_points.shape

        if n_traj < 2 or n_layers < 4:
            return {"exponent": 0.0, "confidence": 0.0, "interpretation": "insufficient data"}

        # Find pairs of initially close trajectories
        initial_dists = squareform(pdist(all_points[:, 0, :]))
        np.fill_diagonal(initial_dists, np.inf)

        # For each trajectory, find nearest neighbor
        lyapunov_estimates = []
        for i in range(min(n_traj, 30)):
            j = np.argmin(initial_dists[i])
            d0 = initial_dists[i, j]
            if d0 < 1e-10:
                continue

            # Track divergence over layers
            divergences = []
            for layer in range(n_layers):
                d = np.linalg.norm(all_points[i, layer] - all_points[j, layer])
                divergences.append(d)

            # Lyapunov = rate of log(distance) growth
            log_divs = np.log(np.array(divergences) + 1e-10)
            if len(log_divs) > 2:
                x = np.arange(len(log_divs))
                slope, _, r_value, _, _ = linregress(x, log_divs)
                lyapunov_estimates.append(slope)

        if not lyapunov_estimates:
            return {"exponent": 0.0, "confidence": 0.0, "interpretation": "no valid pairs"}

        mean_lyapunov = float(np.mean(lyapunov_estimates))
        std_lyapunov = float(np.std(lyapunov_estimates))

        if mean_lyapunov > 0.1:
            interpretation = "CHAOTIC (diverging trajectories)"
        elif mean_lyapunov < -0.1:
            interpretation = "CONVERGENT (attracting fixed point/cycle)"
        else:
            interpretation = "NEUTRAL/PERIODIC (marginal stability)"

        return {
            "exponent": mean_lyapunov,
            "std": std_lyapunov,
            "n_pairs": len(lyapunov_estimates),
            "interpretation": interpretation,
            "confidence": float(min(1.0, len(lyapunov_estimates) / 10)),
        }

    @staticmethod
    def _estimate_fractal_dimension(points):
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.
        Integer dimension = regular attractor, non-integer = strange attractor.
        """
        n_points, n_dims = points.shape

        if n_points < 10:
            return {"dimension": 0.0, "confidence": 0.0, "is_fractal": False}

        # Compute pairwise distances
        dists = pdist(points)
        if len(dists) == 0:
            return {"dimension": 0.0, "confidence": 0.0, "is_fractal": False}

        dists = dists[dists > 0]  # remove zeros
        if len(dists) == 0:
            return {"dimension": 0.0, "confidence": 0.0, "is_fractal": False}

        # Correlation integral C(r) = fraction of pairs with distance < r
        # log C(r) vs log r → slope = correlation dimension
        r_values = np.logspace(
            np.log10(np.percentile(dists, 5)),
            np.log10(np.percentile(dists, 95)),
            20
        )

        correlations = []
        for r in r_values:
            c = np.sum(dists < r) / len(dists)
            if c > 0:
                correlations.append((np.log10(r), np.log10(c)))

        if len(correlations) < 5:
            return {"dimension": 0.0, "confidence": 0.0, "is_fractal": False}

        log_r = np.array([c[0] for c in correlations])
        log_c = np.array([c[1] for c in correlations])

        # Linear fit in scaling region (middle portion)
        mid_start = len(log_r) // 4
        mid_end = 3 * len(log_r) // 4
        if mid_end - mid_start < 3:
            mid_start = 0
            mid_end = len(log_r)

        slope, intercept, r_value, p_value, std_err = linregress(
            log_r[mid_start:mid_end], log_c[mid_start:mid_end]
        )

        dimension = float(slope)
        is_fractal = not (abs(dimension - round(dimension)) < 0.15
        )  # close the "not" parenthesis

        is_fractal = not (abs(dimension - round(dimension)) < 0.15)

        return {
            "dimension": dimension,
            "is_fractal": bool(is_fractal),
            "nearest_integer": int(round(dimension)),
            "fractional_part": float(abs(dimension - round(dimension))),
            "r_squared": float(r_value ** 2),
            "confidence": float(min(1.0, r_value ** 2) * min(1.0, n_points / 20)),
            "interpretation": (
                f"D≈{dimension:.2f} ({'FRACTAL/STRANGE' if is_fractal else f'regular (≈{int(round(dimension))}D)'})"
            ),
        }

    @staticmethod
    def _detect_clusters(final_points):
        """Detect multiple attractor basins via DBSCAN clustering."""
        n_points, n_dims = final_points.shape

        if n_points < 3:
            return {"n_clusters": 1, "confidence": 0, "labels": [0] * n_points}

        # Estimate eps from data
        dists = pdist(final_points)
        if len(dists) == 0:
            return {"n_clusters": 1, "confidence": 0, "labels": [0] * n_points}

        eps = np.percentile(dists, 15)  # 15th percentile of pairwise distances
        if eps < 1e-10:
            eps = np.median(dists) * 0.3

        clustering = DBSCAN(eps=eps, min_samples=max(2, n_points // 10)).fit(final_points)
        labels = clustering.labels_

        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()

        # Cluster centers
        cluster_centers = []
        cluster_sizes = []
        for c in range(n_clusters):
            mask = labels == c
            center = final_points[mask].mean(axis=0)
            cluster_centers.append(center.tolist())
            cluster_sizes.append(int(mask.sum()))

        return {
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "cluster_centers": cluster_centers,
            "n_noise_points": int(n_noise),
            "eps_used": float(eps),
            "labels": labels.tolist(),
            "confidence": float(min(1.0, n_clusters / 2) if n_clusters > 1 else 0.5),
        }

    @staticmethod
    def _analyze_recurrence(all_points):
        """
        Recurrence analysis for torus/quasi-periodic detection.
        Computes recurrence rate and determinism (RQA metrics).
        """
        n_traj, n_layers, n_dims = all_points.shape

        if n_layers < 5:
            return {"recurrence_rate": 0, "determinism": 0, "confidence": 0}

        # Use a sample of trajectories
        sample_size = min(n_traj, 15)
        sample_indices = np.linspace(0, n_traj - 1, sample_size, dtype=int)

        recurrence_rates = []
        determinisms = []

        for idx in sample_indices:
            traj = all_points[idx]  # (n_layers, n_dims)
            dists = squareform(pdist(traj))

            # Threshold for recurrence
            threshold = np.percentile(dists[dists > 0], 20) if np.any(dists > 0) else 1.0

            # Recurrence matrix
            R = (dists < threshold).astype(int)
            np.fill_diagonal(R, 0)  # exclude self-recurrence

            # Recurrence rate
            n_possible = n_layers * (n_layers - 1)
            rr = R.sum() / max(n_possible, 1)
            recurrence_rates.append(rr)

            # Determinism: fraction of recurrent points forming diagonal lines (length >= 2)
            det_points = 0
            total_recurrent = R.sum()
            for diag_offset in range(2, n_layers):
                diag = np.diag(R, k=diag_offset)
                # Count consecutive 1s of length >= 2
                in_line = False
                line_len = 0
                for val in diag:
                    if val == 1:
                        line_len += 1
                        in_line = True
                    else:
                        if in_line and line_len >= 2:
                            det_points += line_len
                        line_len = 0
                        in_line = False
                if in_line and line_len >= 2:
                    det_points += line_len

            determinism = det_points / max(total_recurrent, 1)
            determinisms.append(determinism)

        mean_rr = float(np.mean(recurrence_rates))
        mean_det = float(np.mean(determinisms))

        # High recurrence + high determinism = periodic/quasi-periodic
        # High recurrence + low determinism = chaotic
        if mean_rr > 0.1 and mean_det > 0.5:
            interpretation = "QUASI-PERIODIC (possible torus)"
        elif mean_rr > 0.1 and mean_det < 0.3:
            interpretation = "CHAOTIC recurrence"
        elif mean_rr > 0.05:
            interpretation = "WEAK recurrence"
        else:
            interpretation = "NO significant recurrence"

        return {
            "recurrence_rate": mean_rr,
            "determinism": mean_det,
            "interpretation": interpretation,
            "confidence": float(min(1.0, sample_size / 10)),
        }

    @staticmethod
    def _classify(results: dict) -> dict:
        """
        Final classification based on all analyses.
        Returns the most likely attractor type with confidence.
        """
        classifications = []

        # Fixed point
        fp = results["fixed_point"]
        if fp["detected"]:
            classifications.append({
                "type": "FIXED_POINT",
                "confidence": fp["confidence"],
                "evidence": f"convergence_ratio={fp['convergence_ratio']:.3f}, "
                           f"velocity_ratio={fp['velocity_ratio']:.3f}",
            })

        # Limit cycle
        lc = results["limit_cycle"]
        if lc["detected"]:
            classifications.append({
                "type": "LIMIT_CYCLE",
                "confidence": lc["confidence"],
                "evidence": f"period={lc['dominant_period']}, "
                           f"recurrence_strength={lc['mean_recurrence_strength']:.3f}",
            })

        # Torus (quasi-periodic)
        rec = results["recurrence"]
        if rec["recurrence_rate"] > 0.1 and rec["determinism"] > 0.5:
            # Additional check: not a simple cycle (multiple incommensurate frequencies)
            torus_confidence = min(rec["recurrence_rate"], rec["determinism"])
            if not lc["detected"] or lc["confidence"] < torus_confidence:
                classifications.append({
                    "type": "TORUS",
                    "confidence": float(torus_confidence),
                    "evidence": f"recurrence_rate={rec['recurrence_rate']:.3f}, "
                               f"determinism={rec['determinism']:.3f}",
                })

        # Strange attractor (chaotic)
        lyap = results["lyapunov"]
        frac = results["fractal_dimension"]
        if lyap["exponent"] > 0.1 and frac["is_fractal"]:
            strange_confidence = min(
                lyap["confidence"],
                frac["confidence"],
                min(1.0, lyap["exponent"])
            )
            classifications.append({
                "type": "STRANGE_ATTRACTOR",
                "confidence": float(strange_confidence),
                "evidence": f"lyapunov={lyap['exponent']:.3f}, "
                           f"fractal_dim={frac['dimension']:.3f}",
            })
        elif lyap["exponent"] > 0.05:
            classifications.append({
                "type": "STRANGE_ATTRACTOR",
                "confidence": float(lyap["confidence"] * 0.5),
                "evidence": f"lyapunov={lyap['exponent']:.3f} (weak chaos), "
                           f"fractal_dim={frac['dimension']:.3f}",
            })

        # Multiple attractors
        clusters = results["clusters"]
        if clusters["n_clusters"] > 1:
            classifications.append({
                "type": "MULTIPLE_ATTRACTORS",
                "confidence": clusters["confidence"],
                "evidence": f"n_clusters={clusters['n_clusters']}, "
                           f"sizes={clusters['cluster_sizes']}",
            })

        # No attractor detected
        if not classifications:
            classifications.append({
                "type": "NONE",
                "confidence": 0.5,
                "evidence": "No clear attractor structure detected",
            })

        # Sort by confidence
        classifications.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "primary": classifications[0],
            "all_candidates": classifications,
            "summary": " | ".join(
                f"{c['type']}({c['confidence']:.2f})" for c in classifications
            ),
        }

# =============================================================================
# Flask Web Server + API
# =============================================================================

app = Flask(__name__, static_folder=None)
data_loader: DataLoader = None
attractor_detector = AttractorDetector()

# HTML/JS/CSS served inline (single-file deployment)
FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Attractor Viewer — Token Trajectory Analysis</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0a0a0f;
    color: #e0e0e0;
    overflow: hidden;
    height: 100vh;
}
#app {
    display: grid;
    grid-template-columns: 320px 1fr;
    grid-template-rows: auto 1fr;
    height: 100vh;
}
header {
    grid-column: 1 / -1;
    background: #1a1a2e;
    padding: 8px 16px;
    border-bottom: 1px solid #333;
    display: flex;
    align-items: center;
    gap: 16px;
}
header h1 {
    font-size: 16px;
    font-weight: 600;
    color: #7fdbff;
}
header .info {
    font-size: 12px;
    color: #888;
}
#sidebar {
    background: #12121a;
    border-right: 1px solid #333;
    overflow-y: auto;
    padding: 12px;
}
#viewport {
    position: relative;
    background: #0a0a0f;
}
canvas {
    width: 100%;
    height: 100%;
    display: block;
}
.panel {
    margin-bottom: 12px;
    background: #1a1a2e;
    border-radius: 6px;
    border: 1px solid #2a2a3e;
    overflow: hidden;
}
.panel-header {
    padding: 8px 12px;
    background: #222238;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #7fdbff;
    cursor: pointer;
    user-select: none;
}
.panel-header:hover { background: #2a2a48; }
.panel-body {
    padding: 10px 12px;
}
.panel-body.collapsed { display: none; }
label {
    display: block;
    font-size: 11px;
    color: #aaa;
    margin-bottom: 4px;
}
select, input[type="number"], input[type="range"] {
    width: 100%;
    padding: 5px 8px;
    background: #0a0a14;
    border: 1px solid #333;
    border-radius: 4px;
    color: #e0e0e0;
    font-size: 12px;
    margin-bottom: 8px;
}
select:focus, input:focus {
    outline: none;
    border-color: #7fdbff;
}
.row {
    display: flex;
    gap: 8px;
    align-items: center;
}
.row label { flex: 1; margin-bottom: 0; }
.row input, .row select { flex: 2; margin-bottom: 0; }
button {
    padding: 6px 12px;
    background: #2a2a4e;
    border: 1px solid #444;
    border-radius: 4px;
    color: #e0e0e0;
    font-size: 11px;
    cursor: pointer;
    margin: 2px;
}
button:hover { background: #3a3a5e; border-color: #7fdbff; }
button.active { background: #7fdbff; color: #000; }
.attractor-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 600;
    margin: 2px;
}
.badge-fixed { background: #2ecc40; color: #000; }
.badge-cycle { background: #ff851b; color: #000; }
.badge-torus { background: #b10dc9; color: #fff; }
.badge-strange { background: #ff4136; color: #fff; }
.badge-multiple { background: #ffdc00; color: #000; }
.badge-none { background: #555; color: #fff; }
#attractor-results {
    font-size: 11px;
    line-height: 1.6;
}
#attractor-results .metric {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    border-bottom: 1px solid #1a1a2e;
}
#attractor-results .metric-label { color: #888; }
#attractor-results .metric-value { color: #e0e0e0; font-family: monospace; }
#tooltip {
    position: absolute;
    background: rgba(20, 20, 40, 0.95);
    border: 1px solid #7fdbff;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 11px;
    pointer-events: none;
    display: none;
    max-width: 350px;
    z-index: 1000;
    line-height: 1.5;
}
#tooltip .token-name { color: #7fdbff; font-weight: 600; font-size: 13px; }
#tooltip .detail { color: #aaa; }
#loading {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 14px;
    color: #7fdbff;
}
.color-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 6px;
}
.color-swatch {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
}
.color-swatch .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}
.checkbox-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
    font-size: 11px;
}
.checkbox-row input[type="checkbox"] {
    width: auto;
    margin: 0;
}
</style>
</head>
<body>
<div id="app">
<header>
    <h1>🌀 Attractor Viewer</h1>
    <span class="info" id="header-info">Loading...</span>
</header>
<div id="sidebar">
    <!-- Group Selection -->
    <div class="panel">
        <div class="panel-header" onclick="togglePanel(this)">📁 Data Source</div>
        <div class="panel-body">
            <label>Group</label>
            <select id="sel-group" onchange="onGroupChange()"></select>
            <label>Token Filter</label>
            <select id="sel-token-filter" onchange="onSettingsChange()">
                <option value="all">All Tokens</option>
                <option value="last">Last Token Only</option>
            </select>
        </div>
    </div>

    <!-- Layer Range -->
    <div class="panel">
        <div class="panel-header" onclick="togglePanel(this)">📊 Layer Range</div>
        <div class="panel-body">
            <div class="row">
                <label>From</label>
                <input type="number" id="layer-start" value="0" min="0" onchange="onSettingsChange()">
            </div>
            <div class="row">
                <label>To</label>
                <input type="number" id="layer-end" value="-1" min="-1" onchange="onSettingsChange()">
            </div>
            <div style="font-size:10px;color:#666;margin-top:4px;">-1 = last layer</div>
            <button onclick="setLayerRange(0, -1)">All Layers</button>
            <button onclick="setLayerRange(0, 10)">0–10</button>
            <button onclick="setLayerRange(0, 5)">0–5</button>
            <button onclick="setLayerRange(5, 15)">5–15</button>
            <button onclick="setLayerRange(10, -1)">10+</button>
        </div>
    </div>

    <!-- Visualization -->
    <div class="panel">
        <div class="panel-header" onclick="togglePanel(this)">🎨 Visualization</div>
        <div class="panel-body">
            <label>Projection</label>
            <select id="sel-projection" onchange="onProjectionChange()">
                <option value="raw3d">Raw Dimensions (3D slice)</option>
                <option value="pca3d">PCA 3D</option>
                <option value="pca2d">PCA 2D</option>
            </select>
            <div id="raw-dim-controls">
                <div class="row">
                    <label>X dim</label>
                    <input type="number" id="dim-x" value="0" min="0" onchange="onDimChange()">
                </div>
                <div class="row">
                    <label>Y dim</label>
                    <input type="number" id="dim-y" value="1" min="0" onchange="onDimChange()">
                </div>
                <div class="row">
                    <label>Z dim</label>
                    <input type="number" id="dim-z" value="2" min="0" onchange="onDimChange()">
                </div>
            </div>
            <label>Color By</label>
            <select id="sel-color" onchange="onColorChange()">
                <option value="order">Sequence Order (rainbow)</option>
                <option value="token">Token Text</option>
                <option value="layer_velocity">Layer Velocity</option>
                <option value="prompt">Prompt Index</option>
            </select>
            <div class="checkbox-row">
                <input type="checkbox" id="chk-lines" checked onchange="onDisplayChange()">
                <label for="chk-lines">Show trajectory lines</label>
            </div>
            <div class="checkbox-row">
                <input type="checkbox" id="chk-points" checked onchange="onDisplayChange()">
                <label for="chk-points">Show points</label>
            </div>
            <div class="checkbox-row">
                <input type="checkbox" id="chk-labels" onchange="onDisplayChange()">
                <label for="chk-labels">Show token labels</label>
            </div>
            <div class="checkbox-row">
                <input type="checkbox" id="chk-arrows" onchange="onDisplayChange()">
                <label for="chk-arrows">Show direction arrows</label>
            </div>
            <label>Point Size</label>
            <input type="range" id="point-size" min="1" max="15" value="4" onchange="onDisplayChange()">
            <label>Line Opacity</label>
            <input type="range" id="line-opacity" min="0" max="100" value="60" onchange="onDisplayChange()">
        </div>
    </div>

    <!-- Attractor Analysis -->
    <div class="panel">
        <div class="panel-header" onclick="togglePanel(this)">🔬 Attractor Analysis</div>
        <div class="panel-body">
            <button onclick="runAttractorAnalysis()" style="width:100%;padding:8px;background:#2a4a2a;border-color:#4a8a4a;">
                ▶ Analyze Attractors
            </button>
            <div id="attractor-results" style="margin-top:10px;">
                <div style="color:#666;font-style:italic;">Click analyze to detect attractor types</div>
            </div>
        </div>
    </div>

    <!-- Color Legend -->
    <div class="panel">
        <div class="panel-header" onclick="togglePanel(this)">🏷️ Legend</div>
        <div class="panel-body">
            <div id="color-legend" class="color-legend"></div>
        </div>
    </div>
</div>

<div id="viewport">
    <canvas id="canvas3d"></canvas>
    <div id="tooltip"></div>
    <div id="loading">Loading data...</div>
</div>
</div>

<script>
// =============================================================================
// State
// =============================================================================
let state = {
    groups: [],
    currentGroup: null,
    trajectories: [],
    totalLayers: 0,
    dModel: 0,
    layerStart: 0,
    layerEnd: -1,
    tokenFilter: 'all',
    projection: 'raw3d',
    dimX: 0, dimY: 1, dimZ: 2,
    colorBy: 'order',
    showLines: true,
    showPoints: true,
    showLabels: false,
    showArrows: false,
    pointSize: 4,
    lineOpacity: 0.6,
    // Camera
    rotX: -30,
    rotY: 45,
    zoom: 1.0,
    panX: 0, panY: 0,
    // Interaction
    dragging: false,
    lastMouseX: 0, lastMouseY: 0,
    hoveredIdx: -1,
    // Computed
    projectedPoints: [],  // per trajectory: array of [x,y,z] screen coords
    pcaMatrix: null,
};

// =============================================================================
// Init
// =============================================================================
async function init() {
    const resp = await fetch('/api/groups');
    const data = await resp.json();
    state.groups = data.groups;

    const sel = document.getElementById('sel-group');
    sel.innerHTML = '';
    for (const g of state.groups) {
        const opt = document.createElement('option');
        opt.value = g;
        opt.textContent = g;
        sel.appendChild(opt);
    }

    if (state.groups.length > 0) {
        state.currentGroup = state.groups[0];
        await loadGroupData();
    }

    setupCanvas();
    animate();
}

async function loadGroupData() {
    document.getElementById('loading').style.display = 'block';

    const params = new URLSearchParams({
        group: state.currentGroup,
        layer_start: state.layerStart,
        layer_end: state.layerEnd,
        token_filter: state.tokenFilter,
    });

    const resp = await fetch('/api/trajectories?' + params);
    const data = await resp.json();

    if (data.error) {
        document.getElementById('loading').textContent = 'Error: ' + data.error;
        return;
    }

    state.trajectories = data.trajectories;
    state.totalLayers = data.total_layers;
    state.dModel = data.d_model;
    state.nLayers = data.n_layers;

    // Update UI
    document.getElementById('header-info').textContent =
        `${data.n_tokens} tokens | ${data.n_layers} layers (${data.layer_range[0]}-${data.layer_range[1]}) | ${data.d_model}D`;

    document.getElementById('dim-x').max = data.d_model - 1;
    document.getElementById('dim-y').max = data.d_model - 1;
    document.getElementById('dim-z').max = data.d_model - 1;

    // Compute PCA if needed
    if (state.projection.startsWith('pca')) {
        computePCA();
    }

    updateLegend();
    document.getElementById('loading').style.display = 'none';
}

// =============================================================================
// PCA (computed client-side for responsiveness)
// =============================================================================
function computePCA() {
    // Collect all points into a matrix
    const allPts = [];
    for (const traj of state.trajectories) {
        for (const pt of traj.points) {
            allPts.push(pt);
        }
    }
    if (allPts.length === 0) return;

    const n = allPts.length;
    const d = allPts[0].length;

    // Mean center
    const mean = new Array(d).fill(0);
    for (const pt of allPts) {
        for (let i = 0; i < d; i++) mean[i] += pt[i];
    }
    for (let i = 0; i < d; i++) mean[i] /= n;

    // We'll use power iteration for top 3 components (fast enough for browser)
    // First compute covariance approximation using random projection if d is large
    const maxDims = Math.min(d, 50);  // Use top-50 variance dims for PCA base

    // Find top-variance dimensions
    const variances = new Array(d).fill(0);
    for (const pt of allPts) {
        for (let i = 0; i < d; i++) {
            const diff = pt[i] - mean[i];
            variances[i] += diff * diff;
        }
    }

    const dimOrder = Array.from({length: d}, (_, i) => i)
        .sort((a, b) => variances[b] - variances[a]);
    const topDims = dimOrder.slice(0, maxDims);

    // Build reduced matrix (n × maxDims)
    const reduced = [];
    for (const pt of allPts) {
        const row = topDims.map(di => pt[di] - mean[di]);
        reduced.push(row);
    }

    // Power iteration for top 3 eigenvectors of (reduced^T * reduced)
    const components = [];
    const residual = reduced.map(row => [...row]);

    for (let comp = 0; comp < 3; comp++) {
        let vec = new Array(maxDims).fill(0).map(() => Math.random() - 0.5);
        let norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
        vec = vec.map(v => v / norm);

        for (let iter = 0; iter < 50; iter++) {
            // Multiply: new_vec = (X^T * X) * vec
            const Xv = residual.map(row => row.reduce((s, v, i) => s + v * vec[i], 0));
            const newVec = new Array(maxDims).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < maxDims; j++) {
                    newVec[j] += residual[i][j] * Xv[i];
                }
            }
            norm = Math.sqrt(newVec.reduce((s, v) => s + v * v, 0));
            if (norm < 1e-10) break;
            vec = newVec.map(v => v / norm);
        }

        components.push({vec, topDims});

        // Deflate: remove this component from residual
        const projections = residual.map(row => row.reduce((s, v, i) => s + v * vec[i], 0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < maxDims; j++) {
                residual[i][j] -= projections[i] * vec[j];
            }
        }
    }

    state.pcaMatrix = {components, mean, topDims};
}

function projectPCA(point) {
    if (!state.pcaMatrix) return [0, 0, 0];
    const {components, mean, topDims} = state.pcaMatrix;
    const result = [];
    for (const comp of components) {
        let val = 0;
        for (let i = 0; i < comp.topDims.length; i++) {
            val += (point[comp.topDims[i]] - mean[comp.topDims[i]]) * comp.vec[i];
        }
        result.push(val);
    }
    return result;
}

// =============================================================================
// 3D Canvas Rendering
// =============================================================================
let canvas, ctx;

function setupCanvas() {
    canvas = document.getElementById('canvas3d');
    ctx = canvas.getContext('2d');

    const resize = () => {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    };
    resize();
    window.addEventListener('resize', resize);

    // Mouse interaction
    canvas.addEventListener('mousedown', e => {
        state.dragging = true;
        state.lastMouseX = e.clientX;
        state.lastMouseY = e.clientY;
    });
    canvas.addEventListener('mouseup', () => { state.dragging = false; });
    canvas.addEventListener('mouseleave', () => { state.dragging = false; });
    canvas.addEventListener('mousemove', e => {
        if (state.dragging) {
            const dx = e.clientX - state.lastMouseX;
            const dy = e.clientY - state.lastMouseY;
            state.rotY += dx * 0
.5;
            state.rotX += dy * 0.5;
            state.rotX = Math.max(-90, Math.min(90, state.rotX));
            state.lastMouseX = e.clientX;
            state.lastMouseY = e.clientY;
        } else {
            // Hover detection
            checkHover(e);
        }
    });
    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        const factor = e.deltaY > 0 ? 0.9 : 1.1;
        state.zoom *= factor;
        state.zoom = Math.max(0.1, Math.min(10, state.zoom));
    }, {passive: false});

    // Right-click pan
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('mousedown', e => {
        if (e.button === 2) {
            state.panning = true;
            state.lastMouseX = e.clientX;
            state.lastMouseY = e.clientY;
        }
    });
    canvas.addEventListener('mouseup', e => {
        if (e.button === 2) state.panning = false;
    });
    canvas.addEventListener('mousemove', e => {
        if (state.panning) {
            state.panX += (e.clientX - state.lastMouseX);
            state.panY += (e.clientY - state.lastMouseY);
            state.lastMouseX = e.clientX;
            state.lastMouseY = e.clientY;
        }
    });
}

function checkHover(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let minDist = 20;  // pixel threshold
    let nearest = -1;

    for (let i = 0; i < state.projectedPoints.length; i++) {
        const pts = state.projectedPoints[i];
        if (!pts || pts.length === 0) continue;
        // Check last point (endpoint)
        const last = pts[pts.length - 1];
        if (!last) continue;
        const dx = last.sx - mx;
        const dy = last.sy - my;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < minDist) {
            minDist = dist;
            nearest = i;
        }
    }

    state.hoveredIdx = nearest;
    const tooltip = document.getElementById('tooltip');
    if (nearest >= 0) {
        const traj = state.trajectories[nearest];
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX - canvas.parentElement.getBoundingClientRect().left + 15) + 'px';
        tooltip.style.top = (e.clientY - canvas.parentElement.getBoundingClientRect().top - 10) + 'px';
        tooltip.innerHTML = `
            <div class="token-name">"${escapeHtml(traj.token_text)}"</div>
            <div class="detail">Position: ${traj.token_pos} | Prompt: #${traj.prompt_idx}</div>
            <div class="detail">Sentence: ${escapeHtml((traj.sentence || '').substring(0, 80))}${(traj.sentence||'').length > 80 ? '...' : ''}</div>
            ${traj.prediction ? `<div class="detail">Prediction: "${escapeHtml(traj.prediction)}"</div>` : ''}
        `;
    } else {
        tooltip.style.display = 'none';
    }
}

function escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// =============================================================================
// 3D Projection & Rendering
// =============================================================================

function project3D(x, y, z) {
    // Apply rotation
    const radX = state.rotX * Math.PI / 180;
    const radY = state.rotY * Math.PI / 180;

    // Rotate around Y axis
    let x1 = x * Math.cos(radY) - z * Math.sin(radY);
    let z1 = x * Math.sin(radY) + z * Math.cos(radY);
    let y1 = y;

    // Rotate around X axis
    let y2 = y1 * Math.cos(radX) - z1 * Math.sin(radX);
    let z2 = y1 * Math.sin(radX) + z1 * Math.cos(radX);
    let x2 = x1;

    // Perspective projection
    const perspective = 800;
    const scale = perspective / (perspective + z2) * state.zoom;

    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;

    const sx = w / 2 + x2 * scale * (w / 4) + state.panX;
    const sy = h / 2 - y2 * scale * (h / 4) + state.panY;

    return {sx, sy, z: z2, scale};
}

function getTrajectoryPoints(traj) {
    const points = traj.points;
    if (!points || points.length === 0) return [];

    const result = [];
    for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        let x, y, z;

        if (state.projection === 'raw3d') {
            x = pt[state.dimX] || 0;
            y = pt[state.dimY] || 0;
            z = pt[state.dimZ] || 0;
        } else if (state.projection === 'pca3d' || state.projection === 'pca2d') {
            const pca = projectPCA(pt);
            x = pca[0] || 0;
            y = pca[1] || 0;
            z = state.projection === 'pca3d' ? (pca[2] || 0) : 0;
        } else {
            x = pt[0] || 0;
            y = pt[1] || 0;
            z = pt[2] || 0;
        }
        result.push({x, y, z});
    }
    return result;
}

function normalizePoints(allTrajectoryPoints) {
    // Find global bounds
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (const pts of allTrajectoryPoints) {
        for (const p of pts) {
            if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
            if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z;
        }
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const rangeZ = maxZ - minZ || 1;
    const maxRange = Math.max(rangeX, rangeY, rangeZ);

    // Normalize to [-1, 1]
    const normalized = [];
    for (const pts of allTrajectoryPoints) {
        const norm = [];
        for (const p of pts) {
            norm.push({
                x: (p.x - (minX + maxX) / 2) / (maxRange / 2),
                y: (p.y - (minY + maxY) / 2) / (maxRange / 2),
                z: (p.z - (minZ + maxZ) / 2) / (maxRange / 2),
            });
        }
        normalized.push(norm);
    }
    return normalized;
}

function getColor(trajIdx, pointIdx, nPoints) {
    const traj = state.trajectories[trajIdx];

    if (state.colorBy === 'order') {
        // Rainbow based on point index (layer) within trajectory
        const t = nPoints > 1 ? pointIdx / (nPoints - 1) : 0;
        return hslToRgb(t * 300, 80, 55);
    } else if (state.colorBy === 'token') {
        // Hash token text to color
        const hash = hashString(traj.token_text);
        return hslToRgb((hash % 360), 70, 50);
    } else if (state.colorBy === 'layer_velocity') {
        // Color by velocity (distance between consecutive points)
        if (pointIdx > 0 && traj.points[pointIdx] && traj.points[pointIdx - 1]) {
            let dist = 0;
            const curr = traj.points[pointIdx];
            const prev = traj.points[pointIdx - 1];
            for (let d = 0; d < curr.length; d++) {
                dist += (curr[d] - prev[d]) ** 2;
            }
            dist = Math.sqrt(dist);
            // Map to color (blue=slow, red=fast)
            const t = Math.min(1, dist / 10);
            return hslToRgb((1 - t) * 240, 80, 50);
        }
        return 'rgb(100,100,100)';
    } else if (state.colorBy === 'prompt') {
        const t = state.trajectories.length > 1 ? trajIdx / (state.trajectories.length - 1) : 0;
        return hslToRgb(t * 330, 75, 50);
    }
    return 'rgb(150,150,255)';
}

function hslToRgb(h, s, l) {
    s /= 100; l /= 100;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;
    let r, g, b;
    if (h < 60) { r = c; g = x; b = 0; }
    else if (h < 120) { r = x; g = c; b = 0; }
    else if (h < 180) { r = 0; g = c; b = x; }
    else if (h < 240) { r = 0; g = x; b = c; }
    else if (h < 300) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    return `rgb(${Math.round((r+m)*255)},${Math.round((g+m)*255)},${Math.round((b+m)*255)})`;
}

function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash);
}

// =============================================================================
// Main Render Loop
// =============================================================================

function animate() {
    render();
    requestAnimationFrame(animate);
}

function render() {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    ctx.clearRect(0, 0, w, h);

    if (state.trajectories.length === 0) return;

    // Get raw trajectory points
    const rawPoints = state.trajectories.map(t => getTrajectoryPoints(t));

    // Normalize to [-1, 1]
    const normPoints = normalizePoints(rawPoints);

    // Project to screen
    state.projectedPoints = [];
    const allProjected = [];

    for (let ti = 0; ti < normPoints.length; ti++) {
        const pts = normPoints[ti];
        const projected = [];
        for (let pi = 0; pi < pts.length; pi++) {
            const p = project3D(pts[pi].x, pts[pi].y, pts[pi].z);
            projected.push(p);
        }
        state.projectedPoints.push(projected);
        allProjected.push({trajIdx: ti, projected});
    }

    // Sort by average Z for painter's algorithm (back to front)
    allProjected.sort((a, b) => {
        const avgZa = a.projected.reduce((s, p) => s + p.z, 0) / (a.projected.length || 1);
        const avgZb = b.projected.reduce((s, p) => s + p.z, 0) / (b.projected.length || 1);
        return avgZb - avgZa;  // far first
    });

    const pointSize = parseInt(document.getElementById('point-size').value);
    const lineOpacity = parseInt(document.getElementById('line-opacity').value) / 100;
    const showLines = document.getElementById('chk-lines').checked;
    const showPoints = document.getElementById('chk-points').checked;
    const showLabels = document.getElementById('chk-labels').checked;
    const showArrows = document.getElementById('chk-arrows').checked;

    // Draw trajectories
    for (const {trajIdx, projected} of allProjected) {
        const nPts = projected.length;
        const isHovered = (trajIdx === state.hoveredIdx);
        const alpha = isHovered ? 1.0 : lineOpacity;

        // Lines
        if (showLines && nPts > 1) {
            ctx.beginPath();
            ctx.moveTo(projected[0].sx, projected[0].sy);
            for (let i = 1; i < nPts; i++) {
                ctx.lineTo(projected[i].sx, projected[i].sy);
            }
            ctx.strokeStyle = getColor(trajIdx, Math.floor(nPts / 2), nPts);
            ctx.globalAlpha = alpha * 0.7;
            ctx.lineWidth = isHovered ? 3 : 1;
            ctx.stroke();
        }

        // Points
        if (showPoints) {
            for (let i = 0; i < nPts; i++) {
                const p = projected[i];
                const color = getColor(trajIdx, i, nPts);
                const size = isHovered ? pointSize * 2 : pointSize * p.scale;

                ctx.beginPath();
                ctx.arc(p.sx, p.sy, Math.max(1, size), 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.globalAlpha = alpha;
                ctx.fill();

                // Outline for hovered
                if (isHovered) {
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }

        // Arrows
        if (showArrows && nPts > 1) {
            const step = Math.max(1, Math.floor(nPts / 4));
            for (let i = 0; i < nPts - 1; i += step) {
                const from = projected[i];
                const to = projected[i + 1];
                const dx = to.sx - from.sx;
                const dy = to.sy - from.sy;
                const len = Math.sqrt(dx * dx + dy * dy);
                if (len < 5) continue;

                const angle = Math.atan2(dy, dx);
                const arrowSize = 6;

                ctx.beginPath();
                ctx.moveTo(to.sx, to.sy);
                ctx.lineTo(
                    to.sx - arrowSize * Math.cos(angle - 0.4),
                    to.sy - arrowSize * Math.sin(angle - 0.4)
                );
                ctx.lineTo(
                    to.sx - arrowSize * Math.cos(angle + 0.4),
                    to.sy - arrowSize * Math.sin(angle + 0.4)
                );
                ctx.closePath();
                ctx.fillStyle = getColor(trajIdx, i, nPts);
                ctx.globalAlpha = alpha * 0.8;
                ctx.fill();
            }
        }

        // Labels
        if (showLabels && (nPts > 0)) {
            const last = projected[nPts - 1];
            ctx.globalAlpha = isHovered ? 1.0 : 0.7;
            ctx.font = isHovered ? 'bold 12px sans-serif' : '9px sans-serif';
            ctx.fillStyle = isHovered ? '#7fdbff' : '#ccc';
            ctx.fillText(state.trajectories[trajIdx].token_text, last.sx + 5, last.sy - 5);
        }
    }

    ctx.globalAlpha = 1.0;

    // Draw axes indicator (bottom-left)
    drawAxesIndicator();
}

function drawAxesIndicator() {
    const ox = 60, oy = canvas.height / window.devicePixelRatio - 60;
    const len = 40;

    const axes = [
        {x: 1, y: 0, z: 0, label: `X:d${state.dimX}`, color: '#ff4444'},
        {x: 0, y: 1, z: 0, label: `Y:d${state.dimY}`, color: '#44ff44'},
        {x: 0, y: 0, z: 1, label: `Z:d${state.dimZ}`, color: '#4444ff'},
    ];

    for (const axis of axes) {
        const p = project3D(axis.x * 0.3, axis.y * 0.3, axis.z * 0.3);
        const o = project3D(0, 0, 0);
        const dx = (p.sx - o.sx);
        const dy = (p.sy - o.sy);
        const norm = Math.sqrt(dx * dx + dy * dy) || 1;

        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(ox + dx / norm * len, oy + dy / norm * len);
        ctx.strokeStyle = axis.color;
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.font = '10px monospace';
        ctx.fillStyle = axis.color;
        ctx.fillText(axis.label, ox + dx / norm * (len + 5), oy + dy / norm * (len + 5));
    }
}

// =============================================================================
// UI Event Handlers
// =============================================================================

function togglePanel(header) {
    const body = header.nextElementSibling;
    body.classList.toggle('collapsed');
}

async function onGroupChange() {
    state.currentGroup = document.getElementById('sel-group').value;
    await loadGroupData();
}

async function onSettingsChange() {
    state.layerStart = parseInt(document.getElementById('layer-start').value);
    state.layerEnd = parseInt(document.getElementById('layer-end').value);
    state.tokenFilter = document.getElementById('sel-token-filter').value;
    await loadGroupData();
}

function setLayerRange(start, end) {
    document.getElementById('layer-start').value = start;
    document.getElementById('layer-end').value = end;
    onSettingsChange();
}

function onProjectionChange() {
    state.projection = document.getElementById('sel-projection').value;
    const rawControls = document.getElementById('raw-dim-controls');
    rawControls.style.display = state.projection === 'raw3d' ? 'block' : 'none';

    if (state.projection.startsWith('pca') && state.trajectories.length > 0) {
        computePCA();
    }
}

function onDimChange() {
    state.dimX = parseInt(document.getElementById('dim-x').value);
    state.dimY = parseInt(document.getElementById('dim-y').value);
    state.dimZ = parseInt(document.getElementById('dim-z').value);
}

function onColorChange() {
    state.colorBy = document.getElementById('sel-color').value;
    updateLegend();
}

function onDisplayChange() {
    state.showLines = document.getElementById('chk-lines').checked;
    state.showPoints = document.getElementById('chk-points').checked;
    state.showLabels = document.getElementById('chk-labels').checked;
    state.showArrows = document.getElementById('chk-arrows').checked;
}

function updateLegend() {
    const legend = document.getElementById('color-legend');
    legend.innerHTML = '';

    if (state.colorBy === 'order') {
        const steps = 8;
        for (let i = 0; i < steps; i++) {
            const t = i / (steps - 1);
            const color = hslToRgb(t * 300, 80, 55);
            const label = `Layer ${Math.round(t * (state.nLayers - 1))}`;
            legend.innerHTML += `<div class="color-swatch"><div class="dot" style="background:${color}"></div>${label}</div>`;
        }
    } else if (state.colorBy === 'token') {
        // Show first few unique tokens
        const seen = new Set();
        for (const traj of state.trajectories.slice(0, 20)) {
            if (seen.has(traj.token_text)) continue;
            seen.add(traj.token_text);
            const hash = hashString(traj.token_text);
            const color = hslToRgb(hash % 360, 70, 50);
            legend.innerHTML += `<div class="color-swatch"><div class="dot" style="background:${color}"></div>${escapeHtml(traj.token_text)}</div>`;
            if (seen.size >= 15) break;
        }
    } else if (state.colorBy === 'prompt') {
        const n = Math.min(state.trajectories.length, 12);
        const seen = new Set();
        for (let i = 0; i < state.trajectories.length && seen.size < n; i++) {
            const pidx = state.trajectories[i].prompt_idx;
            if (seen.has(pidx)) continue;
            seen.add(pidx);
            const t = state.trajectories.length > 1 ? i / (state.trajectories.length - 1) : 0;
            const color = hslToRgb(t * 330, 75, 50);
            legend.innerHTML += `<div class="color-swatch"><div class="dot" style="background:${color}"></div>Prompt #${pidx}</div>`;
        }
    } else if (state.colorBy === 'layer_velocity') {
        legend.innerHTML = `
            <div class="color-swatch"><div class="dot" style="background:rgb(0,0,255)"></div>Slow (small delta)</div>
            <div class="color-swatch"><div class="dot" style="background:rgb(255,0,0)"></div>Fast (large delta)</div>
        `;
    }
}

// =============================================================================
// Attractor Analysis (calls backend)
// =============================================================================

async function runAttractorAnalysis() {
    const resultsDiv = document.getElementById('attractor-results');
    resultsDiv.innerHTML = '<div style="color:#7fdbff;">⏳ Analyzing...</div>';

    const params = new URLSearchParams({
        group: state.currentGroup,
        layer_start: state.layerStart,
        layer_end: state.layerEnd,
        token_filter: state.tokenFilter,
    });

    try {
        const resp = await fetch('/api/analyze?' + params);
        const data = await resp.json();

        if (data.error) {
            resultsDiv.innerHTML = `<div style="color:#ff4136;">Error: ${data.error}</div>`;
            return;
        }

        let html = '';

        // Classification badge
        const cls = data.classification.primary;
        const badgeClass = {
            'FIXED_POINT': 'badge-fixed',
            'LIMIT_CYCLE': 'badge-cycle',
            'TORUS': 'badge-torus',
            'STRANGE_ATTRACTOR': 'badge-strange',
            'MULTIPLE_ATTRACTORS': 'badge-multiple',
            'NONE': 'badge-none',
        }[cls.type] || 'badge-none';

        html += `<div style="margin-bottom:8px;">
            <span class="attractor-badge ${badgeClass}">${cls.type}</span>
            <span style="font-size:10px;color:#888;"> conf: ${(cls.confidence * 100).toFixed(0)}%</span>
        </div>`;
        html += `<div style="font-size:10px;color:#aaa;margin-bottom:8px;">${cls.evidence}</div>`;

        // All candidates
        if (data.classification.all_candidates.length > 1) {
            html += `<div style="font-size:10px;color:#666;margin-bottom:6px;">Other candidates:</div>`;
            for (const c of data.classification.all_candidates.slice(1)) {
                const bc = {
                    'FIXED_POINT': 'badge-fixed', 'LIMIT_CYCLE': 'badge-cycle',
                    'TORUS': 'badge-torus', 'STRANGE_ATTRACTOR': 'badge-strange',
                    'MULTIPLE_ATTRACTORS': 'badge-multiple', 'NONE': 'badge-none',
                }[c.type] || 'badge-none';
                html += `<span class="attractor-badge ${bc}" style="font-size:9px;">${c.type} (${(c.confidence*100).toFixed(0)}%)</span> `;
            }
            html += '<br><br>';
        }

        // Metrics
        html += '<div style="border-top:1px solid #333;padding-top:6px;">';

        // Lyapunov
        const lyap = data.lyapunov;
        html += `<div class="metric"><span class="metric-label">Lyapunov exp.</span><span class="metric-value">${lyap.exponent.toFixed(4)}</span></div>`;
        html += `<div class="metric"><span class="metric-label">  → ${lyap.interpretation}</span><span class="metric-value"></span></div>`;

        // Fractal dimension
        const frac = data.fractal_dimension;
        html += `<div class="metric"><span class="metric-label">Fractal dim.</span><span class="metric-value">${frac.dimension.toFixed(3)}</span></div>`;
        html += `<div class="metric"><span class="metric-label">  → ${frac.interpretation}</span><span class="metric-value"></span></div>`;

        // Convergence
        const conv = data.convergence;
        html += `<div class="metric"><span class="metric-label">Convergence rate</span><span class="metric-value">${conv.convergence_rate.toFixed(4)}</span></div>`;
        html += `<div class="metric"><span class="metric-label">Spread ratio</span><span class="metric-value">${conv.spread_ratio.toFixed(4)}</span></div>`;

        // Recurrence
        const rec = data.recurrence;
        html += `<div class="metric"><span class="metric-label">Recurrence rate</span><span class="metric-value">${rec.recurrence_rate.toFixed(4)}</span></div>`;
        html += `<div class="metric"><span class="metric-label">Determinism</span><span class="metric-value">${rec.determinism.toFixed(4)}</span></div>`;
        html += `<div class="metric"><span class="metric-label">  → ${rec.interpretation}</span><span class="metric-value"></span></div>`;

        // Clusters
        const clust = data.clusters;
        html += `<div class="metric"><span class="metric-label">Clusters found</span><span class="metric-value">${clust.n_clusters}</span></div>`;
        if (clust.n_clusters > 1) {
            html += `<div class="metric"><span class="metric-label">Cluster sizes</span><span class="metric-value">${clust.cluster_sizes.join(', ')}</span></div>`;
        }

        // Fixed point
        const fp = data.fixed_point;
        html += `<div class="metric"><span class="metric-label">Fixed point?</span><span class="metric-value">${fp.detected ? 'YES' : 'no'} (conf: ${(fp.confidence*100).toFixed(0)}%)</span></div>`;

        // Limit cycle
        const lc = data.limit_cycle;
        html += `<div class="metric"><span class="metric-label">Limit cycle?</span><span class="metric-value">${lc.detected ? 'YES (period ' + lc.dominant_period + ')' : 'no'} (conf: ${(lc.confidence*100).toFixed(0)}%)</span></div>`;

        html += '</div>';
        resultsDiv.innerHTML = html;

    } catch (err) {
        resultsDiv.innerHTML = `<div style="color:#ff4136;">Error: ${err.message}</div>`;
    }
}

// =============================================================================
// Start
// =============================================================================
init();
</script>
</body>
</html>"""

# =============================================================================
# Flask Routes
# =============================================================================

@app.route('/')
def index():
    return FRONTEND_HTML

@app.route('/api/groups')
def api_groups():
    return jsonify({"groups": list(data_loader.groups.keys())})

@app.route('/api/group_info')
def api_group_info():
    group = request.args.get('group', '')
    if group not in data_loader.groups:
        return jsonify({"error": f"Unknown group: {group}"})
    info = data_loader.get_group_info(group)
    return jsonify(info)

@app.route('/api/trajectories')
def api_trajectories():
    group = request.args.get('group', '')
    layer_start = int(request.args.get('layer_start', 0))
    layer_end = int(request.args.get('layer_end', -1))
    token_filter = request.args.get('token_filter', 'all')

    if group not in data_loader.groups:
        return jsonify({"error": f"Unknown group: {group}"})

    result = data_loader.load_trajectories(group, layer_start, layer_end, token_filter)
    return jsonify(result)

@app.route('/api/analyze')
def api_analyze():
    group = request.args.get('group', '')
    layer_start = int(request.args.get('layer_start', 0))
    layer_end = int(request.args.get('layer_end', -1))
    token_filter = request.args.get('token_filter', 'all')

    if group not in data_loader.groups:
        return jsonify({"error": f"Unknown group: {group}"})

    # Load trajectories
    traj_data = data_loader.load_trajectories(group, layer_start, layer_end, token_filter)
    if "error" in traj_data:
        return jsonify(traj_data)

    # Run attractor analysis
    result = AttractorDetector.analyze(
        traj_data["trajectories"],
        traj_data["layer_range"],
    )

    return jsonify(result)

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Attractor Viewer — Browser-based Token Trajectory Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python viewer.py attractor_data/
    python viewer.py attractor_data/berlin_multilingual
    python viewer.py attractor_data/ --port 8899
    python viewer.py results/ --host 0.0.0.0
        """
    )
    parser.add_argument("data_dir", type=str,
                       help="Path to data directory (group or parent)")
    parser.add_argument("--port", type=int, default=8765,
                       help="Port for web server (default: 8765)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open browser")

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"ERROR: Path not found: {data_path}")
        sys.exit(1)

    global data_loader
    data_loader = DataLoader(data_path)

    if not data_loader.groups:
        print(f"ERROR: No valid data groups found in {data_path}")
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    print(f"\n{'=' * 60}")
    print(f"  🌀 Attractor Viewer")
    print(f"{'=' * 60}")
    print(f"  Data:    {data_path}")
    print(f"  Groups:  {list(data_loader.groups.keys())}")
    print(f"  Server:  {url}")
    print(f"{'=' * 60}")
    print(f"  Open in browser: {url}")
    print(f"  Press Ctrl+C to stop\n")

    # Auto-open browser
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    # Run Flask
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
