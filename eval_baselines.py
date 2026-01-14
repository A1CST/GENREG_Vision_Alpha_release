# ================================================================
# GENREG Evaluation - Baseline Comparison
# ================================================================
# Evaluates a checkpoint against:
# - Random Guess Baseline: picks random vocabulary word
# - Frequency Baseline: picks most common answer in corpus
# - Model Performance: actual checkpoint predictions
#
# Outputs JSON with metrics and PNG bar chart comparison.
# ================================================================

import os
import sys
import json
import random
import argparse
from collections import Counter
from datetime import datetime

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("[ERROR] PyTorch required")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] Matplotlib not available - no PNG output")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[ERROR] Pygame required")
    sys.exit(1)

import config as cfg
from genreg_visual_env import load_corpus, get_vocabulary
from genreg_checkpoint import load_checkpoint, get_latest_checkpoint, list_checkpoints


# ================================================================
# GPU IMAGE CACHE (simplified from GRV_GPU.py)
# ================================================================
class GPUImageCache:
    """Pre-renders all sentences as GPU tensors."""

    def __init__(self, corpus, width, height, font_size, device, blank_marker="[____]"):
        self.device = device
        self.width = width
        self.height = height
        self.num_sentences = len(corpus)

        pygame.init()
        surface = pygame.Surface((width, height))
        font = pygame.font.Font(None, font_size)

        print(f"[CACHE] Rendering {len(corpus)} sentences...")

        images_np = []
        self.answers = []
        self.sentences = []

        for i, entry in enumerate(corpus):
            sentence = entry["text"]
            answers = [a.lower() for a in entry["answers"]]

            surface.fill((25, 25, 30))
            display_text = sentence.replace("____", blank_marker)
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
        print(f"[CACHE] Done - {len(corpus)} images on GPU")


# ================================================================
# BASELINE CALCULATORS
# ================================================================
def calculate_random_baseline(vocabulary, answer_sets, num_trials=10):
    """
    Calculate expected accuracy for random guessing.

    Args:
        vocabulary: list of vocab words
        answer_sets: list of sets of valid answers per sentence
        num_trials: number of trials for Monte Carlo estimation

    Returns:
        dict with accuracy stats
    """
    vocab_size = len(vocabulary)
    num_sentences = len(answer_sets)

    # For each sentence, probability of randomly picking a correct answer
    # is |valid_answers| / |vocabulary|

    # Monte Carlo simulation for variance estimate
    trial_accuracies = []

    for _ in range(num_trials):
        correct = 0
        for answers in answer_sets:
            random_word = random.choice(vocabulary)
            if random_word in answers:
                correct += 1
        trial_accuracies.append(correct / num_sentences)

    # Also compute theoretical expected value
    theoretical_accuracy = sum(
        len(answers) / vocab_size for answers in answer_sets
    ) / num_sentences

    return {
        "accuracy": np.mean(trial_accuracies),
        "std": np.std(trial_accuracies),
        "theoretical": theoretical_accuracy,
        "num_trials": num_trials,
    }


def calculate_frequency_baseline(corpus, vocabulary, answer_sets):
    """
    Calculate accuracy using most frequent answer strategy.

    The frequency baseline always picks the most common answer word
    in the entire corpus - a simple but strong baseline.

    Args:
        corpus: list of sentence entries
        vocabulary: list of vocab words
        answer_sets: list of sets of valid answers per sentence

    Returns:
        dict with accuracy stats and the top word
    """
    # Count all answer occurrences
    answer_counts = Counter()
    for entry in corpus:
        for answer in entry["answers"]:
            answer_counts[answer.lower()] += 1

    # Get most common answer
    most_common = answer_counts.most_common(10)
    top_word = most_common[0][0] if most_common else vocabulary[0]

    # Calculate accuracy if always guessing top_word
    correct = sum(1 for answers in answer_sets if top_word in answers)
    accuracy = correct / len(answer_sets)

    return {
        "accuracy": accuracy,
        "top_word": top_word,
        "top_word_count": answer_counts[top_word],
        "top_10": most_common,
        "correct_sentences": correct,
        "total_sentences": len(answer_sets),
    }


# ================================================================
# MODEL EVALUATION
# ================================================================
def evaluate_model(checkpoint_path, corpus, vocabulary, image_cache, device, num_runs=5):
    """
    Evaluate model accuracy on the corpus.

    Args:
        checkpoint_path: path to checkpoint file
        corpus: list of sentence entries
        vocabulary: list of vocab words
        image_cache: GPUImageCache instance
        device: torch device
        num_runs: number of evaluation runs (model is stochastic)

    Returns:
        dict with accuracy stats
    """
    population, generation, template, _ = load_checkpoint(checkpoint_path)

    # Get best genome
    best_genome = max(population.genomes, key=lambda g: g.trust)
    controller = best_genome.controller

    # Build weight tensors
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

    vocab_size = len(vocabulary)
    num_sentences = len(corpus)

    run_accuracies = []
    all_predictions = []

    for run in range(num_runs):
        with torch.no_grad():
            # Forward pass: (sentences, input) @ (input, hidden) + bias
            hidden = torch.tanh(image_cache.images @ w1.T + b1)
            outputs = hidden @ w2.T + b2

            # Get vocabulary logits and sample
            vocab_logits = outputs[:, :vocab_size]
            vocab_probs = F.softmax(vocab_logits, dim=1)
            word_indices = torch.multinomial(vocab_probs, 1).squeeze(1)

            # Check correctness
            correct = 0
            predictions = []
            for i, idx in enumerate(word_indices.cpu().tolist()):
                word = vocabulary[idx]
                is_correct = word in image_cache.answers[i]
                correct += int(is_correct)
                predictions.append({
                    "sentence": image_cache.sentences[i],
                    "predicted": word,
                    "expected": list(image_cache.answers[i]),
                    "correct": is_correct,
                })

            run_accuracies.append(correct / num_sentences)
            if run == 0:
                all_predictions = predictions

    return {
        "accuracy": np.mean(run_accuracies),
        "std": np.std(run_accuracies),
        "num_runs": num_runs,
        "generation": generation,
        "best_trust": best_genome.trust,
        "predictions": all_predictions,
    }


# ================================================================
# VISUALIZATION
# ================================================================
def create_comparison_chart(results, output_path):
    """Create bar chart comparing baselines to model."""
    if not MATPLOTLIB_AVAILABLE:
        print("[SKIP] Matplotlib not available")
        return

    labels = ["Random\nGuess", "Frequency\nBaseline", "Model"]
    accuracies = [
        results["random_baseline"]["accuracy"] * 100,
        results["frequency_baseline"]["accuracy"] * 100,
        results["model"]["accuracy"] * 100,
    ]
    errors = [
        results["random_baseline"]["std"] * 100,
        0,  # Frequency baseline has no variance
        results["model"]["std"] * 100,
    ]

    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, accuracies, yerr=errors, capsize=5, color=colors,
                  edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, acc, err in zip(bars, accuracies, errors):
        height = bar.get_height()
        label = f"{acc:.1f}%"
        if err > 0:
            label += f"\n(+/-{err:.1f})"
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    # Add frequency baseline info
    freq_word = results["frequency_baseline"]["top_word"]
    ax.annotate(f'(always "{freq_word}")',
                xy=(1, accuracies[1]),
                xytext=(0, -25),
                textcoords="offset points",
                ha='center', va='top',
                fontsize=10, fontstyle='italic', color='#666')

    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title(f"Model vs Baselines (Gen {results['model']['generation']:,})", fontsize=16)
    ax.set_ylim(0, max(accuracies) * 1.2)
    ax.axhline(y=100 / len(results["vocabulary"]), color='gray', linestyle='--',
               alpha=0.5, label=f'Chance (1/{len(results["vocabulary"])})')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Chart: {output_path}")
    plt.close()


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint against baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_baselines.py                              # Use latest checkpoint
  python eval_baselines.py -c path/to/checkpoint.pkl   # Specific checkpoint
  python eval_baselines.py --list                      # List available checkpoints
        """
    )
    parser.add_argument("--checkpoint", "-c", metavar="PATH",
                        help="Path to checkpoint file")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available checkpoints")
    parser.add_argument("--output", "-o", metavar="DIR", default="eval_results",
                        help="Output directory for results (default: eval_results)")
    parser.add_argument("--runs", "-r", type=int, default=5,
                        help="Number of evaluation runs for model (default: 5)")

    args = parser.parse_args()

    # List checkpoints if requested
    if args.list:
        print("\nAvailable checkpoints:")
        print("=" * 60)
        base_dir = "checkpoints_visual_gpu"
        if os.path.exists(base_dir):
            for session in sorted(os.listdir(base_dir), reverse=True):
                session_path = os.path.join(base_dir, session)
                if os.path.isdir(session_path):
                    checkpoints = list_checkpoints(session_path)
                    if checkpoints:
                        latest = checkpoints[-1]
                        gen = int(os.path.basename(latest).split("_")[2].split(".")[0])
                        print(f"  {session}: {len(checkpoints)} checkpoints (latest: gen {gen:,})")
                        print(f"    -> {latest}")
        return

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Find latest from any session
        base_dir = "checkpoints_visual_gpu"
        checkpoint_path = None
        if os.path.exists(base_dir):
            for session in sorted(os.listdir(base_dir), reverse=True):
                session_path = os.path.join(base_dir, session)
                if os.path.isdir(session_path):
                    latest = get_latest_checkpoint(session_path)
                    if latest:
                        checkpoint_path = latest
                        break

        if not checkpoint_path:
            print("[ERROR] No checkpoints found. Use --checkpoint to specify one.")
            sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    vocabulary = get_vocabulary()
    corpus = load_corpus()
    print(f"Vocabulary: {len(vocabulary)} words")
    print(f"Corpus: {len(corpus)} sentences")

    # Create image cache
    image_cache = GPUImageCache(
        corpus=corpus,
        width=cfg.FIELD_WIDTH,
        height=cfg.FIELD_HEIGHT,
        font_size=cfg.FONT_SIZE,
        device=device,
        blank_marker=cfg.BLANK_MARKER
    )

    # Calculate baselines
    print("\n" + "-" * 40)
    print("Calculating Random Baseline...")
    random_baseline = calculate_random_baseline(vocabulary, image_cache.answers)
    print(f"  Accuracy: {random_baseline['accuracy']*100:.2f}% (+/- {random_baseline['std']*100:.2f}%)")
    print(f"  Theoretical: {random_baseline['theoretical']*100:.2f}%")

    print("\nCalculating Frequency Baseline...")
    freq_baseline = calculate_frequency_baseline(corpus, vocabulary, image_cache.answers)
    print(f"  Top word: '{freq_baseline['top_word']}' (appears in {freq_baseline['top_word_count']} answers)")
    print(f"  Accuracy: {freq_baseline['accuracy']*100:.2f}%")
    print(f"  Top 10 words: {[w for w, c in freq_baseline['top_10']]}")

    print("\nEvaluating Model...")
    model_results = evaluate_model(
        checkpoint_path, corpus, vocabulary, image_cache, device,
        num_runs=args.runs
    )
    print(f"  Generation: {model_results['generation']:,}")
    print(f"  Best Trust: {model_results['best_trust']:.1f}")
    print(f"  Accuracy: {model_results['accuracy']*100:.2f}% (+/- {model_results['std']*100:.2f}%)")

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "vocabulary_size": len(vocabulary),
        "corpus_size": len(corpus),
        "vocabulary": vocabulary,
        "random_baseline": random_baseline,
        "frequency_baseline": freq_baseline,
        "model": model_results,
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Random Guess:      {random_baseline['accuracy']*100:6.2f}%")
    print(f"  Frequency Baseline:{freq_baseline['accuracy']*100:6.2f}%")
    print(f"  Model:             {model_results['accuracy']*100:6.2f}%")
    print()

    improvement_over_random = (model_results['accuracy'] - random_baseline['accuracy']) / random_baseline['accuracy'] * 100
    improvement_over_freq = (model_results['accuracy'] - freq_baseline['accuracy']) / max(0.001, freq_baseline['accuracy']) * 100

    print(f"  Model vs Random:   {improvement_over_random:+.1f}% improvement")
    print(f"  Model vs Frequency:{improvement_over_freq:+.1f}% improvement")

    # Save outputs
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = os.path.join(args.output, f"baselines_{timestamp}.json")
    # Remove predictions from JSON to keep it small
    results_json = {k: v for k, v in results.items()}
    results_json["model"] = {k: v for k, v in model_results.items() if k != "predictions"}
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[SAVED] JSON: {json_path}")

    # Save PNG
    png_path = os.path.join(args.output, f"baselines_{timestamp}.png")
    create_comparison_chart(results, png_path)

    pygame.quit()
    print("\nDone!")


if __name__ == "__main__":
    main()
