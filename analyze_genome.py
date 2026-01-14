#!/usr/bin/env python3
# ================================================================
# GENREG Genome Analysis Script
# ================================================================
# Comprehensive analysis of the best genome from a checkpoint.
# Generates multiple visualizations revealing network behavior.
#
# Usage:
#   python analyze_genome.py              # Interactive session selection
#   python analyze_genome.py --output-dir analysis_results
# ================================================================

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("[ERROR] PyTorch is required")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("[ERROR] Matplotlib is required")
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not available - some visualizations disabled")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[ERROR] Pygame is required for rendering")
    sys.exit(1)

# Import project modules
import config as cfg
from genreg_checkpoint import load_checkpoint
from genreg_visual_env import extract_vocabulary, load_corpus


# ================================================================
# SESSION DISCOVERY (same as eval.py)
# ================================================================
def find_sessions(checkpoint_base="checkpoints_visual_gpu"):
    """Find all training sessions with checkpoints."""
    if not os.path.exists(checkpoint_base):
        return []

    sessions = []
    for name in sorted(os.listdir(checkpoint_base)):
        session_path = os.path.join(checkpoint_base, name)
        if os.path.isdir(session_path) and name.startswith("session_"):
            checkpoints = glob.glob(os.path.join(session_path, "checkpoint_gen_*.pkl"))
            if checkpoints:
                latest = max(checkpoints, key=lambda p: int(os.path.basename(p).split("_")[2].split(".")[0]))
                gen = int(os.path.basename(latest).split("_")[2].split(".")[0])
                sessions.append({
                    "name": name,
                    "path": session_path,
                    "num_checkpoints": len(checkpoints),
                    "latest_gen": gen,
                    "latest_checkpoint": latest
                })
    return sessions


def select_session_interactive(sessions):
    """Interactively select a session."""
    if not sessions:
        print("[ERROR] No sessions found")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("AVAILABLE SESSIONS")
    print("=" * 60)

    for i, session in enumerate(sessions):
        print(f"  [{i + 1}] {session['name']}")
        print(f"      Checkpoints: {session['num_checkpoints']}, Latest: gen {session['latest_gen']:,}")

    if len(sessions) == 1:
        print(f"\n[AUTO] Selecting: {sessions[0]['name']}")
        return sessions[0]

    while True:
        try:
            choice = input(f"\nSelect session [1-{len(sessions)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)


# ================================================================
# IMAGE RENDERING
# ================================================================
class AnalysisRenderer:
    """Renders sentences and captures activations."""

    def __init__(self, corpus, width, height, font_size, device):
        self.device = device
        self.width = width
        self.height = height

        pygame.init()
        surface = pygame.Surface((width, height))
        font = pygame.font.Font(None, font_size)

        images_np = []
        self.answers = []
        self.sentences = []

        for entry in corpus:
            sentence = entry["text"]
            answers = [a.lower() for a in entry["answers"]]

            surface.fill((25, 25, 30))
            display_text = sentence.replace("____", "[____]")
            text_surface = font.render(display_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(width // 2, height // 2))
            surface.blit(text_surface, text_rect)

            pixels = pygame.surfarray.array3d(surface)
            grayscale = np.mean(pixels, axis=2)
            normalized = (grayscale / 255.0).flatten().astype(np.float32)
            images_np.append(normalized)

            self.answers.append(set(answers))
            self.sentences.append(sentence)

        images_np = np.stack(images_np)
        self.images = torch.from_numpy(images_np).to(device)

        pygame.quit()


# ================================================================
# ACTIVATION EXTRACTION
# ================================================================
def extract_activations(genome, images, vocabulary, device):
    """
    Run the genome on all images and extract detailed activations.

    Returns dict with:
        - hidden_activations: (num_sentences, hidden_size) raw hidden layer outputs
        - hidden_pre_activation: (num_sentences, hidden_size) before tanh
        - output_logits: (num_sentences, vocab_size) raw output
        - output_probs: (num_sentences, vocab_size) softmax probabilities
        - predictions: list of predicted word indices
        - weights: dict with w1, b1, w2, b2
    """
    controller = genome.controller
    vocab_size = len(vocabulary)

    # Get weights as tensors
    if hasattr(controller, '_use_torch') and controller._use_torch:
        w1 = controller.w1.to(device)
        b1 = controller.b1.to(device)
        w2 = controller.w2.to(device)
        b2 = controller.b2.to(device)
    else:
        w1 = torch.tensor(controller.w1, dtype=torch.float32, device=device)
        b1 = torch.tensor(controller.b1, dtype=torch.float32, device=device)
        w2 = torch.tensor(controller.w2, dtype=torch.float32, device=device)
        b2 = torch.tensor(controller.b2, dtype=torch.float32, device=device)

    # Forward pass with intermediate values
    hidden_pre = images @ w1.T + b1  # Before activation
    hidden = torch.tanh(hidden_pre)   # After tanh
    output_logits = hidden @ w2.T + b2

    vocab_logits = output_logits[:, :vocab_size]
    output_probs = F.softmax(vocab_logits, dim=1)
    predictions = torch.argmax(output_probs, dim=1)

    return {
        "hidden_pre_activation": hidden_pre.cpu().numpy(),
        "hidden_activations": hidden.cpu().numpy(),
        "output_logits": output_logits.cpu().numpy(),
        "output_probs": output_probs.cpu().numpy(),
        "predictions": predictions.cpu().numpy(),
        "weights": {
            "w1": w1.cpu().numpy(),
            "b1": b1.cpu().numpy(),
            "w2": w2.cpu().numpy(),
            "b2": b2.cpu().numpy(),
        }
    }


# ================================================================
# VISUALIZATION FUNCTIONS
# ================================================================

def plot_weight_heatmaps(weights, output_dir):
    """Plot heatmaps of weight matrices."""
    print("  - Weight matrix heatmaps...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Weight Matrix Analysis", fontsize=16, fontweight='bold')

    # W1: (hidden_size, input_size) - too large to show fully, show statistics
    w1 = weights["w1"]
    ax = axes[0, 0]
    # Show a subsample or aggregate view
    w1_reshaped = w1.reshape(w1.shape[0], 400, 100)  # hidden x width x height
    w1_spatial_mean = np.mean(np.abs(w1_reshaped), axis=0)  # Average across neurons
    im = ax.imshow(w1_spatial_mean.T, cmap='hot', aspect='auto')
    ax.set_title(f"W1 Spatial Sensitivity (avg across {w1.shape[0]} neurons)")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    plt.colorbar(im, ax=ax, label="Mean |weight|")

    # W1 per-neuron statistics
    ax = axes[0, 1]
    w1_neuron_stats = np.array([
        [np.mean(w1[i]), np.std(w1[i]), np.min(w1[i]), np.max(w1[i])]
        for i in range(w1.shape[0])
    ])
    x = np.arange(w1.shape[0])
    ax.bar(x - 0.2, w1_neuron_stats[:, 0], 0.4, label='Mean', alpha=0.7)
    ax.bar(x + 0.2, w1_neuron_stats[:, 1], 0.4, label='Std', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Weight Value")
    ax.set_title("W1 Per-Neuron Weight Statistics")
    ax.legend()
    ax.set_xticks(x)

    # W2: (output_size, hidden_size) - show full if small enough
    w2 = weights["w2"]
    ax = axes[1, 0]
    # Show top 50 outputs for clarity
    w2_subset = w2[:50, :]
    im = ax.imshow(w2_subset, cmap='RdBu_r', aspect='auto', vmin=-np.abs(w2_subset).max(), vmax=np.abs(w2_subset).max())
    ax.set_title(f"W2 Weight Matrix (first 50/{w2.shape[0]} outputs)")
    ax.set_xlabel("Hidden Neuron")
    ax.set_ylabel("Output Word Index")
    plt.colorbar(im, ax=ax, label="Weight")

    # Bias distributions
    ax = axes[1, 1]
    b1 = weights["b1"]
    b2 = weights["b2"][:50]  # First 50 for visibility
    ax.bar(np.arange(len(b1)) - 0.2, b1, 0.4, label='B1 (hidden)', alpha=0.7)
    ax.set_xlabel("Index")
    ax.set_ylabel("Bias Value")
    ax.set_title("Bias Values")
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_weight_heatmaps.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_neuron_saturation(activations, output_dir):
    """Analyze neuron saturation (tanh extremes)."""
    print("  - Neuron saturation analysis...")

    hidden = activations["hidden_activations"]
    hidden_pre = activations["hidden_pre_activation"]
    num_neurons = hidden.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Neuron Saturation Analysis", fontsize=16, fontweight='bold')

    # Saturation histogram per neuron
    ax = axes[0, 0]
    saturation_threshold = 0.95  # Consider saturated if |activation| > 0.95
    saturation_rates = np.mean(np.abs(hidden) > saturation_threshold, axis=0)
    colors = plt.cm.RdYlGn_r(saturation_rates)
    bars = ax.bar(range(num_neurons), saturation_rates * 100, color=colors)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Saturation Rate (%)")
    ax.set_title(f"Per-Neuron Saturation Rate (|activation| > {saturation_threshold})")
    ax.legend()

    # Activation distribution (all neurons combined)
    ax = axes[0, 1]
    ax.hist(hidden.flatten(), bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=-saturation_threshold, color='r', linestyle='--', label=f'Saturation zone')
    ax.axvline(x=saturation_threshold, color='r', linestyle='--')
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Density")
    ax.set_title("Hidden Activation Distribution (all neurons)")
    ax.legend()

    # Pre-activation distribution
    ax = axes[1, 0]
    ax.hist(hidden_pre.flatten(), bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(x=-2, color='gray', linestyle='--', alpha=0.5, label='tanh linear zone')
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Pre-Activation Value (before tanh)")
    ax.set_ylabel("Density")
    ax.set_title("Pre-Activation Distribution")
    ax.legend()

    # Per-neuron activation range
    ax = axes[1, 1]
    neuron_mins = np.min(hidden, axis=0)
    neuron_maxs = np.max(hidden, axis=0)
    neuron_means = np.mean(hidden, axis=0)
    x = np.arange(num_neurons)
    ax.fill_between(x, neuron_mins, neuron_maxs, alpha=0.3, label='Range')
    ax.plot(x, neuron_means, 'b-', linewidth=2, label='Mean')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Activation Value")
    ax.set_title("Per-Neuron Activation Range")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_neuron_saturation.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_hidden_activation_patterns(activations, sentences, output_dir):
    """Visualize hidden activation patterns across sentences."""
    print("  - Hidden activation patterns...")

    hidden = activations["hidden_activations"]
    num_sentences = len(sentences)
    num_neurons = hidden.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Hidden Layer Activation Patterns", fontsize=16, fontweight='bold')

    # Activation heatmap across all sentences
    ax = axes[0, 0]
    im = ax.imshow(hidden.T, cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1)
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Neuron Index")
    ax.set_title("Hidden Activations Across All Sentences")
    plt.colorbar(im, ax=ax, label="Activation")

    # Neuron correlation matrix
    ax = axes[0, 1]
    corr_matrix = np.corrcoef(hidden.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")
    ax.set_title("Neuron Activation Correlation Matrix")
    plt.colorbar(im, ax=ax, label="Correlation")

    # Top activated neurons per sentence (first 20 sentences)
    ax = axes[1, 0]
    sample_size = min(20, num_sentences)
    for i in range(sample_size):
        top_neurons = np.argsort(np.abs(hidden[i]))[-3:]  # Top 3 neurons
        ax.scatter([i] * 3, top_neurons, c=hidden[i, top_neurons],
                   cmap='RdBu_r', vmin=-1, vmax=1, s=50, alpha=0.7)
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Neuron Index")
    ax.set_title("Top 3 Most Active Neurons per Sentence (first 20)")

    # Neuron activity frequency
    ax = axes[1, 1]
    activity_threshold = 0.5
    activity_freq = np.mean(np.abs(hidden) > activity_threshold, axis=0)
    ax.bar(range(num_neurons), activity_freq * 100, color='teal', alpha=0.7)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Activity Frequency (%)")
    ax.set_title(f"Neuron Activity Frequency (|activation| > {activity_threshold})")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_hidden_patterns.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_output_analysis(activations, vocabulary, answers, output_dir):
    """Analyze output layer behavior."""
    print("  - Output layer analysis...")

    probs = activations["output_probs"]
    predictions = activations["predictions"]
    num_sentences = probs.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Output Layer Analysis", fontsize=16, fontweight='bold')

    # Confidence distribution
    ax = axes[0, 0]
    max_probs = np.max(probs, axis=1)
    ax.hist(max_probs, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=np.mean(max_probs), color='r', linestyle='--', label=f'Mean: {np.mean(max_probs):.2f}')
    ax.set_xlabel("Max Probability (Confidence)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend()

    # Entropy of predictions
    ax = axes[0, 1]
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    ax.hist(entropy, bins=30, density=True, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(x=np.mean(entropy), color='r', linestyle='--', label=f'Mean: {np.mean(entropy):.2f}')
    ax.set_xlabel("Prediction Entropy")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Entropy Distribution")
    ax.legend()

    # Most common predictions
    ax = axes[1, 0]
    pred_counts = defaultdict(int)
    for p in predictions:
        pred_counts[vocabulary[p]] += 1
    sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    words, counts = zip(*sorted_preds) if sorted_preds else ([], [])
    ax.barh(range(len(words)), counts, color='teal', alpha=0.7)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel("Count")
    ax.set_title("Top 15 Most Predicted Words")
    ax.invert_yaxis()

    # Correct vs incorrect confidence
    ax = axes[1, 1]
    correct_mask = np.array([vocabulary[predictions[i]] in answers[i] for i in range(num_sentences)])
    correct_conf = max_probs[correct_mask] if np.any(correct_mask) else []
    incorrect_conf = max_probs[~correct_mask] if np.any(~correct_mask) else []

    ax.hist(correct_conf, bins=20, alpha=0.6, label=f'Correct ({len(correct_conf)})', color='green', density=True)
    ax.hist(incorrect_conf, bins=20, alpha=0.6, label=f'Incorrect ({len(incorrect_conf)})', color='red', density=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence: Correct vs Incorrect Predictions")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_output_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_neuron_importance(activations, vocabulary, answers, output_dir):
    """Analyze which neurons are most important for correct predictions."""
    print("  - Neuron importance analysis...")

    hidden = activations["hidden_activations"]
    predictions = activations["predictions"]
    weights = activations["weights"]
    num_neurons = hidden.shape[1]
    num_sentences = hidden.shape[0]

    # Determine correctness
    correct_mask = np.array([vocabulary[predictions[i]] in answers[i] for i in range(num_sentences)])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Neuron Importance Analysis", fontsize=16, fontweight='bold')

    # Mean activation difference (correct vs incorrect)
    ax = axes[0, 0]
    if np.any(correct_mask) and np.any(~correct_mask):
        mean_correct = np.mean(hidden[correct_mask], axis=0)
        mean_incorrect = np.mean(hidden[~correct_mask], axis=0)
        diff = mean_correct - mean_incorrect
        colors = ['green' if d > 0 else 'red' for d in diff]
        ax.bar(range(num_neurons), diff, color=colors, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Mean Activation Difference")
        ax.set_title("Correct - Incorrect Mean Activation per Neuron")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')

    # Output weight magnitude per neuron (how much each neuron influences output)
    ax = axes[0, 1]
    w2 = weights["w2"]
    neuron_influence = np.sum(np.abs(w2), axis=0)  # Sum of absolute output weights
    ax.bar(range(num_neurons), neuron_influence, color='purple', alpha=0.7)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Sum of |Output Weights|")
    ax.set_title("Neuron Influence on Output Layer")

    # Activation variance per neuron
    ax = axes[1, 0]
    activation_var = np.var(hidden, axis=0)
    ax.bar(range(num_neurons), activation_var, color='orange', alpha=0.7)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Activation Variance")
    ax.set_title("Per-Neuron Activation Variance")

    # Dead neuron detection
    ax = axes[1, 1]
    dead_threshold = 0.1
    always_low = np.mean(np.abs(hidden) < dead_threshold, axis=0)
    always_high = np.mean(np.abs(hidden) > 0.9, axis=0)
    x = np.arange(num_neurons)
    width = 0.35
    ax.bar(x - width/2, always_low * 100, width, label=f'Near-zero (|a| < {dead_threshold})', color='gray', alpha=0.7)
    ax.bar(x + width/2, always_high * 100, width, label='Saturated (|a| > 0.9)', color='red', alpha=0.7)
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Percentage of Sentences")
    ax.set_title("Dead/Saturated Neuron Detection")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_neuron_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_representation_space(activations, vocabulary, answers, sentences, output_dir):
    """Visualize the learned representation space using dimensionality reduction."""
    print("  - Representation space visualization...")

    if not SKLEARN_AVAILABLE:
        print("    (skipped - scikit-learn not available)")
        return

    hidden = activations["hidden_activations"]
    predictions = activations["predictions"]
    num_sentences = hidden.shape[0]

    # Determine correctness
    correct_mask = np.array([vocabulary[predictions[i]] in answers[i] for i in range(num_sentences)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Hidden Representation Space", fontsize=16, fontweight='bold')

    # PCA
    ax = axes[0]
    pca = PCA(n_components=2)
    hidden_pca = pca.fit_transform(hidden)
    scatter = ax.scatter(hidden_pca[:, 0], hidden_pca[:, 1],
                        c=correct_mask, cmap='RdYlGn', alpha=0.7, s=50)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of Hidden Activations")
    plt.colorbar(scatter, ax=ax, label="Correct")

    # t-SNE
    ax = axes[1]
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_sentences - 1))
        hidden_tsne = tsne.fit_transform(hidden)
        scatter = ax.scatter(hidden_tsne[:, 0], hidden_tsne[:, 1],
                            c=correct_mask, cmap='RdYlGn', alpha=0.7, s=50)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("t-SNE of Hidden Activations")
        plt.colorbar(scatter, ax=ax, label="Correct")
    except Exception as e:
        ax.text(0.5, 0.5, f"t-SNE failed: {e}", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_representation_space.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_spatial_sensitivity(weights, output_dir):
    """Visualize spatial sensitivity of input layer weights."""
    print("  - Spatial sensitivity analysis...")

    w1 = weights["w1"]
    num_neurons = w1.shape[0]
    width, height = 400, 100

    # Reshape to spatial
    w1_spatial = w1.reshape(num_neurons, width, height)

    # Create figure with one heatmap per neuron (up to 24)
    n_cols = 6
    n_rows = (num_neurons + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
    fig.suptitle("Per-Neuron Spatial Sensitivity (Input Weights)", fontsize=16, fontweight='bold')

    axes = axes.flatten() if num_neurons > 1 else [axes]

    for i in range(num_neurons):
        ax = axes[i]
        spatial = w1_spatial[i].T  # Transpose for correct orientation
        vmax = np.abs(spatial).max()
        im = ax.imshow(spatial, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_title(f"Neuron {i}", fontsize=10)
        ax.axis('off')

    # Hide unused axes
    for i in range(num_neurons, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_spatial_sensitivity.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_analysis(activations, vocabulary, answers, sentences, output_dir):
    """Detailed prediction analysis."""
    print("  - Prediction analysis...")

    probs = activations["output_probs"]
    predictions = activations["predictions"]
    num_sentences = len(sentences)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Prediction Analysis", fontsize=16, fontweight='bold')

    # Accuracy by sentence position
    ax = axes[0, 0]
    correct_mask = np.array([vocabulary[predictions[i]] in answers[i] for i in range(num_sentences)])

    # Rolling accuracy
    window = 10
    if num_sentences >= window:
        rolling_acc = np.convolve(correct_mask.astype(float), np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, num_sentences), rolling_acc * 100, 'b-', linewidth=2)
        ax.axhline(y=np.mean(correct_mask) * 100, color='r', linestyle='--',
                  label=f'Overall: {np.mean(correct_mask)*100:.1f}%')
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Rolling Accuracy (window={window})")
    ax.legend()

    # Top-k accuracy
    ax = axes[0, 1]
    top_k_values = [1, 3, 5, 10]
    top_k_acc = []
    for k in top_k_values:
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct_in_top_k = np.array([
            any(vocabulary[top_k_preds[i, j]] in answers[i] for j in range(k))
            for i in range(num_sentences)
        ])
        top_k_acc.append(np.mean(correct_in_top_k) * 100)

    ax.bar(range(len(top_k_values)), top_k_acc, tick_label=[f'Top-{k}' for k in top_k_values],
           color='teal', alpha=0.7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Top-K Accuracy")
    for i, v in enumerate(top_k_acc):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center')

    # Confusion analysis: predicted vs expected categories
    ax = axes[1, 0]
    error_types = defaultdict(int)
    for i in range(num_sentences):
        pred_word = vocabulary[predictions[i]]
        if pred_word not in answers[i]:
            # Categorize error
            if any(pred_word in vocabulary[:50] for _ in [1]):  # Common words
                error_types["common_word"] += 1
            else:
                error_types["other"] += 1

    if error_types:
        labels, values = zip(*error_types.items())
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        ax.set_title("Error Categories")
    else:
        ax.text(0.5, 0.5, "No errors!", ha='center', va='center')

    # Second-choice analysis
    ax = axes[1, 1]
    second_choices = np.argsort(probs, axis=1)[:, -2]
    second_correct = np.array([vocabulary[second_choices[i]] in answers[i] for i in range(num_sentences)])
    first_wrong_second_right = (~correct_mask) & second_correct

    ax.bar(['1st choice correct', '2nd choice correct\n(when 1st wrong)', '2nd choice\nwould help'],
           [np.sum(correct_mask), np.sum(first_wrong_second_right), np.sum(first_wrong_second_right)],
           color=['green', 'blue', 'orange'], alpha=0.7)
    ax.set_ylabel("Count")
    ax.set_title("Choice Analysis")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_prediction_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_weight_distribution(weights, output_dir):
    """Analyze weight distributions."""
    print("  - Weight distribution analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Weight Distribution Analysis", fontsize=16, fontweight='bold')

    for idx, (name, w) in enumerate(weights.items()):
        ax = axes[idx // 2, idx % 2]
        ax.hist(w.flatten(), bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='-', linewidth=1)
        ax.axvline(x=np.mean(w), color='g', linestyle='--', label=f'Mean: {np.mean(w):.4f}')
        ax.axvline(x=np.std(w), color='orange', linestyle='--', label=f'Std: {np.std(w):.4f}')
        ax.axvline(x=-np.std(w), color='orange', linestyle='--')
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Density")
        ax.set_title(f"{name.upper()} Distribution (shape: {w.shape})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_weight_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(activations, vocabulary, answers, genome_info, output_dir):
    """Create a summary dashboard with key metrics."""
    print("  - Summary dashboard...")

    hidden = activations["hidden_activations"]
    probs = activations["output_probs"]
    predictions = activations["predictions"]
    weights = activations["weights"]
    num_sentences = len(answers)

    # Calculate metrics
    correct_mask = np.array([vocabulary[predictions[i]] in answers[i] for i in range(num_sentences)])
    accuracy = np.mean(correct_mask) * 100
    mean_confidence = np.mean(np.max(probs, axis=1))

    saturation_rate = np.mean(np.abs(hidden) > 0.95) * 100
    dead_neurons = np.sum(np.all(np.abs(hidden) < 0.1, axis=0))
    active_neurons = hidden.shape[1] - dead_neurons

    w1_sparsity = np.mean(np.abs(weights["w1"]) < 0.01) * 100
    w2_sparsity = np.mean(np.abs(weights["w2"]) < 0.01) * 100

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle(f"Genome Analysis Summary - ID: {genome_info['id']}, Gen: {genome_info['generation']:,}",
                 fontsize=16, fontweight='bold')

    # Key metrics
    ax = fig.add_subplot(gs[0, :2])
    ax.axis('off')
    metrics_text = f"""
    PERFORMANCE METRICS
    {'─' * 40}
    Accuracy:        {accuracy:.2f}%
    Mean Confidence: {mean_confidence:.3f}
    Trust Score:     {genome_info['trust']:.2f}

    NETWORK HEALTH
    {'─' * 40}
    Active Neurons:  {active_neurons}/{hidden.shape[1]}
    Saturation Rate: {saturation_rate:.1f}%
    W1 Sparsity:     {w1_sparsity:.1f}%
    W2 Sparsity:     {w2_sparsity:.1f}%
    """
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Architecture diagram
    ax = fig.add_subplot(gs[0, 2:])
    ax.axis('off')
    arch_text = f"""
    ARCHITECTURE
    {'─' * 40}
    Input:   {weights['w1'].shape[1]:,} (400x100 pixels)
    Hidden:  {weights['w1'].shape[0]} neurons (tanh)
    Output:  {weights['w2'].shape[0]} ({len(vocabulary)} vocab)

    Total Parameters: {sum(w.size for w in weights.values()):,}
    """
    ax.text(0.1, 0.9, arch_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Accuracy gauge
    ax = fig.add_subplot(gs[1, 0])
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    bounds = [0, 20, 40, 60, 80, 100]
    for i in range(5):
        ax.barh(0, 20, left=bounds[i], color=colors[i], alpha=0.3, height=0.5)
    ax.barh(0, accuracy, color='blue', height=0.3)
    ax.axvline(x=accuracy, color='blue', linewidth=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Accuracy %")
    ax.set_yticks([])
    ax.set_title("Accuracy Gauge")

    # Confidence distribution mini
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(np.max(probs, axis=1), bins=20, color='steelblue', alpha=0.7)
    ax.set_xlabel("Confidence")
    ax.set_title("Confidence Dist.")

    # Activation distribution mini
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(hidden.flatten(), bins=30, color='coral', alpha=0.7)
    ax.set_xlabel("Activation")
    ax.set_title("Hidden Activations")

    # Per-neuron activity
    ax = fig.add_subplot(gs[1, 3])
    activity = np.mean(np.abs(hidden), axis=0)
    ax.bar(range(len(activity)), activity, color='teal', alpha=0.7)
    ax.set_xlabel("Neuron")
    ax.set_title("Mean |Activation|")

    # Top predictions
    ax = fig.add_subplot(gs[2, :2])
    pred_counts = defaultdict(int)
    for p in predictions:
        pred_counts[vocabulary[p]] += 1
    sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if sorted_preds:
        words, counts = zip(*sorted_preds)
        ax.barh(range(len(words)), counts, color='purple', alpha=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Predictions")

    # Error analysis
    ax = fig.add_subplot(gs[2, 2:])
    error_words = defaultdict(int)
    for i in range(num_sentences):
        pred_word = vocabulary[predictions[i]]
        if pred_word not in answers[i]:
            error_words[pred_word] += 1
    sorted_errors = sorted(error_words.items(), key=lambda x: x[1], reverse=True)[:10]
    if sorted_errors:
        words, counts = zip(*sorted_errors)
        ax.barh(range(len(words)), counts, color='red', alpha=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title("Top 10 Wrong Predictions")
    else:
        ax.text(0.5, 0.5, "No errors!", ha='center', va='center', fontsize=14)
        ax.axis('off')

    plt.savefig(os.path.join(output_dir, "00_summary_dashboard.png"), dpi=150, bbox_inches='tight')
    plt.close()


def save_analysis_report(activations, vocabulary, answers, genome_info, output_dir):
    """Save a JSON report with numerical analysis results."""
    print("  - Saving analysis report...")

    hidden = activations["hidden_activations"]
    probs = activations["output_probs"]
    predictions = activations["predictions"]
    weights = activations["weights"]
    num_sentences = len(answers)

    correct_mask = np.array([vocabulary[predictions[i]] in answers[i] for i in range(num_sentences)])

    report = {
        "genome": {
            "id": genome_info["id"],
            "generation": genome_info["generation"],
            "trust": genome_info["trust"]
        },
        "architecture": {
            "input_size": int(weights["w1"].shape[1]),
            "hidden_size": int(weights["w1"].shape[0]),
            "output_size": int(weights["w2"].shape[0]),
            "vocab_size": len(vocabulary),
            "total_parameters": sum(w.size for w in weights.values())
        },
        "performance": {
            "accuracy": float(np.mean(correct_mask)),
            "correct_count": int(np.sum(correct_mask)),
            "total_count": num_sentences,
            "mean_confidence": float(np.mean(np.max(probs, axis=1))),
            "median_confidence": float(np.median(np.max(probs, axis=1)))
        },
        "hidden_layer": {
            "saturation_rate": float(np.mean(np.abs(hidden) > 0.95)),
            "dead_neurons": int(np.sum(np.all(np.abs(hidden) < 0.1, axis=0))),
            "mean_activation": float(np.mean(hidden)),
            "std_activation": float(np.std(hidden)),
            "per_neuron_mean": [float(x) for x in np.mean(hidden, axis=0)],
            "per_neuron_std": [float(x) for x in np.std(hidden, axis=0)]
        },
        "weights": {
            "w1": {
                "mean": float(np.mean(weights["w1"])),
                "std": float(np.std(weights["w1"])),
                "min": float(np.min(weights["w1"])),
                "max": float(np.max(weights["w1"])),
                "sparsity": float(np.mean(np.abs(weights["w1"]) < 0.01))
            },
            "b1": {
                "mean": float(np.mean(weights["b1"])),
                "std": float(np.std(weights["b1"]))
            },
            "w2": {
                "mean": float(np.mean(weights["w2"])),
                "std": float(np.std(weights["w2"])),
                "min": float(np.min(weights["w2"])),
                "max": float(np.max(weights["w2"])),
                "sparsity": float(np.mean(np.abs(weights["w2"]) < 0.01))
            },
            "b2": {
                "mean": float(np.mean(weights["b2"])),
                "std": float(np.std(weights["b2"]))
            }
        }
    }

    with open(os.path.join(output_dir, "analysis_report.json"), 'w') as f:
        json.dump(report, f, indent=2)


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of the best genome from a checkpoint"
    )
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to checkpoint file (.pkl)")
    parser.add_argument("--corpus", default="corpus/blanks_eval.json",
                        help="Path to evaluation corpus")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for analysis results")
    parser.add_argument("--device", default="auto",
                        help="Device: 'auto', 'cuda', or 'cpu'")

    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[ANALYSIS] Using device: {device}")

    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        sessions = find_sessions()
        session = select_session_interactive(sessions)
        checkpoint_path = session["latest_checkpoint"]
        print(f"\n[ANALYSIS] Selected: {session['name']}")

    # Load checkpoint
    print(f"[ANALYSIS] Loading checkpoint: {checkpoint_path}")
    population, generation, template_proteins, _ = load_checkpoint(checkpoint_path)

    # Find best genome
    best_genome = max(population.genomes, key=lambda g: g.trust)
    print(f"[ANALYSIS] Best genome: ID {best_genome.id}, Trust: {best_genome.trust:.1f}")

    genome_info = {
        "id": best_genome.id,
        "generation": generation,
        "trust": best_genome.trust
    }

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"analysis_gen{generation}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"[ANALYSIS] Output directory: {output_dir}")

    # Load vocabulary and corpus
    vocabulary = extract_vocabulary(load_corpus())
    print(f"[ANALYSIS] Vocabulary size: {len(vocabulary)}")

    # Load and render eval corpus
    with open(args.corpus, 'r') as f:
        eval_data = json.load(f)
    eval_corpus = eval_data["sentences"]
    print(f"[ANALYSIS] Eval corpus: {len(eval_corpus)} sentences")

    print("[ANALYSIS] Rendering sentences...")
    renderer = AnalysisRenderer(
        eval_corpus,
        width=cfg.FIELD_WIDTH,
        height=cfg.FIELD_HEIGHT,
        font_size=cfg.FONT_SIZE,
        device=device
    )

    # Extract activations
    print("[ANALYSIS] Extracting activations...")
    activations = extract_activations(best_genome, renderer.images, vocabulary, device)

    # Generate all visualizations
    print("[ANALYSIS] Generating visualizations...")

    plot_summary_dashboard(activations, vocabulary, renderer.answers, genome_info, output_dir)
    plot_weight_heatmaps(activations["weights"], output_dir)
    plot_neuron_saturation(activations, output_dir)
    plot_hidden_activation_patterns(activations, renderer.sentences, output_dir)
    plot_output_analysis(activations, vocabulary, renderer.answers, output_dir)
    plot_neuron_importance(activations, vocabulary, renderer.answers, output_dir)
    plot_representation_space(activations, vocabulary, renderer.answers, renderer.sentences, output_dir)
    plot_spatial_sensitivity(activations["weights"], output_dir)
    plot_prediction_analysis(activations, vocabulary, renderer.answers, renderer.sentences, output_dir)
    plot_weight_distribution(activations["weights"], output_dir)
    save_analysis_report(activations, vocabulary, renderer.answers, genome_info, output_dir)

    print(f"\n[ANALYSIS] Complete! Results saved to: {output_dir}/")
    print(f"[ANALYSIS] Generated files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"  - {f} ({size:.1f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
