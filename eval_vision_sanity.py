# ================================================================
# GENREG Evaluation - Vision Sanity Check
# ================================================================
# Tests whether the model is actually using visual information
# by comparing performance on:
# - Normal images: original rendered text
# - Shuffled pixels: same pixels but randomly permuted
# - Blank images: uniform gray (no information)
#
# If the model relies on vision, shuffled/blank should perform
# at random chance level while normal should be much better.
#
# Outputs JSON with metrics and PNG comparison chart.
# ================================================================

import os
import sys
import json
import random
import argparse
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
# IMAGE CACHES FOR DIFFERENT CONDITIONS
# ================================================================
class NormalImageCache:
    """Normal rendered images."""

    def __init__(self, corpus, width, height, font_size, device, blank_marker="[____]"):
        self.device = device
        self.width = width
        self.height = height
        self.num_sentences = len(corpus)

        pygame.init()
        surface = pygame.Surface((width, height))
        font = pygame.font.Font(None, font_size)

        print(f"[NORMAL] Rendering {len(corpus)} sentences...")

        images_np = []
        self.answers = []
        self.sentences = []

        for entry in corpus:
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
        print(f"[NORMAL] Done")


class ShuffledImageCache:
    """Images with pixels randomly shuffled (destroys spatial structure)."""

    def __init__(self, normal_cache, shuffle_mode="per_image"):
        """
        Create shuffled version of normal cache.

        Args:
            normal_cache: NormalImageCache instance
            shuffle_mode:
                "per_image" - each image gets different random shuffle
                "global" - same shuffle applied to all images (tests if
                          model learns the shuffle pattern)
        """
        self.device = normal_cache.device
        self.width = normal_cache.width
        self.height = normal_cache.height
        self.num_sentences = normal_cache.num_sentences
        self.answers = normal_cache.answers
        self.sentences = normal_cache.sentences
        self.shuffle_mode = shuffle_mode

        print(f"[SHUFFLED] Creating shuffled images (mode={shuffle_mode})...")

        images_np = normal_cache.images.cpu().numpy()
        num_pixels = images_np.shape[1]

        if shuffle_mode == "global":
            # Same permutation for all images
            perm = np.random.permutation(num_pixels)
            shuffled = images_np[:, perm]
        else:
            # Different permutation per image
            shuffled = np.zeros_like(images_np)
            for i in range(len(images_np)):
                perm = np.random.permutation(num_pixels)
                shuffled[i] = images_np[i, perm]

        self.images = torch.from_numpy(shuffled).to(self.device)
        print(f"[SHUFFLED] Done")


class BlankImageCache:
    """Blank uniform images (no visual information)."""

    def __init__(self, normal_cache, fill_value=0.5):
        """
        Create blank version of normal cache.

        Args:
            normal_cache: NormalImageCache instance
            fill_value: pixel value to fill with (0.5 = mid-gray)
        """
        self.device = normal_cache.device
        self.width = normal_cache.width
        self.height = normal_cache.height
        self.num_sentences = normal_cache.num_sentences
        self.answers = normal_cache.answers
        self.sentences = normal_cache.sentences

        print(f"[BLANK] Creating blank images (fill={fill_value})...")

        num_pixels = normal_cache.images.shape[1]
        blank = np.full((self.num_sentences, num_pixels), fill_value, dtype=np.float32)

        self.images = torch.from_numpy(blank).to(self.device)
        print(f"[BLANK] Done")


class NoiseImageCache:
    """Random noise images (tests if model ignores input entirely)."""

    def __init__(self, normal_cache, noise_type="uniform"):
        """
        Create random noise images.

        Args:
            normal_cache: NormalImageCache instance
            noise_type: "uniform" (0-1) or "gaussian" (mean=0.5, std=0.2)
        """
        self.device = normal_cache.device
        self.width = normal_cache.width
        self.height = normal_cache.height
        self.num_sentences = normal_cache.num_sentences
        self.answers = normal_cache.answers
        self.sentences = normal_cache.sentences

        print(f"[NOISE] Creating noise images (type={noise_type})...")

        num_pixels = normal_cache.images.shape[1]

        if noise_type == "gaussian":
            noise = np.random.normal(0.5, 0.2, (self.num_sentences, num_pixels))
            noise = np.clip(noise, 0, 1).astype(np.float32)
        else:
            noise = np.random.uniform(0, 1, (self.num_sentences, num_pixels)).astype(np.float32)

        self.images = torch.from_numpy(noise).to(self.device)
        print(f"[NOISE] Done")


# ================================================================
# MODEL EVALUATION
# ================================================================
def evaluate_on_cache(checkpoint_path, image_cache, vocabulary, device, num_runs=5):
    """
    Evaluate model on a given image cache.

    Args:
        checkpoint_path: path to checkpoint
        image_cache: any cache with .images and .answers
        vocabulary: list of vocab words
        device: torch device
        num_runs: number of evaluation runs

    Returns:
        dict with accuracy stats
    """
    population, generation, template, _ = load_checkpoint(checkpoint_path)

    best_genome = max(population.genomes, key=lambda g: g.trust)
    controller = best_genome.controller

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
    num_sentences = image_cache.num_sentences

    run_accuracies = []

    for run in range(num_runs):
        with torch.no_grad():
            hidden = torch.tanh(image_cache.images @ w1.T + b1)
            outputs = hidden @ w2.T + b2

            vocab_logits = outputs[:, :vocab_size]
            vocab_probs = F.softmax(vocab_logits, dim=1)
            word_indices = torch.multinomial(vocab_probs, 1).squeeze(1)

            correct = 0
            for i, idx in enumerate(word_indices.cpu().tolist()):
                word = vocabulary[idx]
                if word in image_cache.answers[i]:
                    correct += 1

            run_accuracies.append(correct / num_sentences)

    return {
        "accuracy": np.mean(run_accuracies),
        "std": np.std(run_accuracies),
        "num_runs": num_runs,
        "generation": generation,
    }


def calculate_random_baseline(vocabulary, answer_sets):
    """Calculate expected random guess accuracy."""
    vocab_size = len(vocabulary)
    theoretical = sum(len(a) / vocab_size for a in answer_sets) / len(answer_sets)
    return theoretical


# ================================================================
# VISUALIZATION
# ================================================================
def create_sanity_chart(results, output_path):
    """Create bar chart showing vision sanity check results."""
    if not MATPLOTLIB_AVAILABLE:
        print("[SKIP] Matplotlib not available")
        return

    conditions = ["Normal\nImages", "Shuffled\nPixels", "Blank\nImages", "Random\nNoise"]
    accuracies = [
        results["normal"]["accuracy"] * 100,
        results["shuffled"]["accuracy"] * 100,
        results["blank"]["accuracy"] * 100,
        results["noise"]["accuracy"] * 100,
    ]
    errors = [
        results["normal"]["std"] * 100,
        results["shuffled"]["std"] * 100,
        results["blank"]["std"] * 100,
        results["noise"]["std"] * 100,
    ]

    # Color code: green for normal (should be high), red for others (should be low)
    colors = ["#27ae60", "#e74c3c", "#e74c3c", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(conditions, accuracies, yerr=errors, capsize=5, color=colors,
                  edgecolor="black", linewidth=1.5)

    # Add value labels
    for bar, acc, err in zip(bars, accuracies, errors):
        height = bar.get_height()
        label = f"{acc:.1f}%"
        if err > 0.1:
            label += f"\n(+/-{err:.1f})"
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    # Add random chance line
    random_chance = results["random_baseline"] * 100
    ax.axhline(y=random_chance, color='gray', linestyle='--', linewidth=2,
               label=f'Random Chance ({random_chance:.1f}%)')

    # Add interpretation zones
    ax.axhspan(0, random_chance * 1.5, alpha=0.1, color='red',
               label='Expected for non-visual')

    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_title(f"Vision Sanity Check (Gen {results['normal']['generation']:,})\n"
                 f"Does the model actually use visual information?", fontsize=14)
    ax.set_ylim(0, max(accuracies) * 1.3)

    ax.legend(loc='upper right')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add interpretation text
    normal_acc = results["normal"]["accuracy"]
    shuffled_acc = results["shuffled"]["accuracy"]
    random_base = results["random_baseline"]

    if normal_acc > shuffled_acc * 2 and shuffled_acc < random_base * 1.5:
        verdict = "PASS: Model relies on visual structure"
        verdict_color = "green"
    elif normal_acc > shuffled_acc * 1.3:
        verdict = "PARTIAL: Model uses some visual info"
        verdict_color = "orange"
    else:
        verdict = "FAIL: Model may not use vision"
        verdict_color = "red"

    ax.text(0.5, 0.02, verdict, transform=ax.transAxes, fontsize=14,
            fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor=verdict_color, alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Chart: {output_path}")
    plt.close()


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Vision sanity check - verify model uses visual data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_vision_sanity.py                              # Use latest checkpoint
  python eval_vision_sanity.py -c path/to/checkpoint.pkl   # Specific checkpoint
  python eval_vision_sanity.py --list                      # List available checkpoints

This test verifies the model actually uses visual information by comparing:
- Normal images: should have HIGH accuracy
- Shuffled pixels: should drop to ~random (spatial structure destroyed)
- Blank images: should be ~random (no information)
- Noise images: should be ~random (no signal)

If shuffled/blank/noise perform similar to normal, the model is NOT
using visual information (might be relying on biases or language stats).
        """
    )
    parser.add_argument("--checkpoint", "-c", metavar="PATH",
                        help="Path to checkpoint file")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available checkpoints")
    parser.add_argument("--output", "-o", metavar="DIR", default="eval_results",
                        help="Output directory for results (default: eval_results)")
    parser.add_argument("--runs", "-r", type=int, default=5,
                        help="Number of evaluation runs (default: 5)")
    parser.add_argument("--shuffle-mode", choices=["per_image", "global"],
                        default="per_image",
                        help="Shuffle mode: per_image (default) or global")

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
    print("VISION SANITY CHECK")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Shuffle mode: {args.shuffle_mode}")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    vocabulary = get_vocabulary()
    corpus = load_corpus()
    print(f"Vocabulary: {len(vocabulary)} words")
    print(f"Corpus: {len(corpus)} sentences")

    # Create image caches
    print("\n" + "-" * 40)
    normal_cache = NormalImageCache(
        corpus=corpus,
        width=cfg.FIELD_WIDTH,
        height=cfg.FIELD_HEIGHT,
        font_size=cfg.FONT_SIZE,
        device=device,
        blank_marker=cfg.BLANK_MARKER
    )

    shuffled_cache = ShuffledImageCache(normal_cache, shuffle_mode=args.shuffle_mode)
    blank_cache = BlankImageCache(normal_cache, fill_value=0.5)
    noise_cache = NoiseImageCache(normal_cache, noise_type="uniform")

    # Calculate random baseline
    random_baseline = calculate_random_baseline(vocabulary, normal_cache.answers)
    print(f"\nRandom baseline (theoretical): {random_baseline*100:.2f}%")

    # Evaluate on each condition
    print("\n" + "-" * 40)
    print("Evaluating on Normal images...")
    normal_results = evaluate_on_cache(checkpoint_path, normal_cache, vocabulary, device, args.runs)
    print(f"  Accuracy: {normal_results['accuracy']*100:.2f}% (+/- {normal_results['std']*100:.2f}%)")

    print("\nEvaluating on Shuffled pixels...")
    shuffled_results = evaluate_on_cache(checkpoint_path, shuffled_cache, vocabulary, device, args.runs)
    print(f"  Accuracy: {shuffled_results['accuracy']*100:.2f}% (+/- {shuffled_results['std']*100:.2f}%)")

    print("\nEvaluating on Blank images...")
    blank_results = evaluate_on_cache(checkpoint_path, blank_cache, vocabulary, device, args.runs)
    print(f"  Accuracy: {blank_results['accuracy']*100:.2f}% (+/- {blank_results['std']*100:.2f}%)")

    print("\nEvaluating on Noise images...")
    noise_results = evaluate_on_cache(checkpoint_path, noise_cache, vocabulary, device, args.runs)
    print(f"  Accuracy: {noise_results['accuracy']*100:.2f}% (+/- {noise_results['std']*100:.2f}%)")

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_path,
        "shuffle_mode": args.shuffle_mode,
        "vocabulary_size": len(vocabulary),
        "corpus_size": len(corpus),
        "random_baseline": random_baseline,
        "normal": normal_results,
        "shuffled": shuffled_results,
        "blank": blank_results,
        "noise": noise_results,
    }

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"  Normal Images:     {normal_results['accuracy']*100:6.2f}%")
    print(f"  Shuffled Pixels:   {shuffled_results['accuracy']*100:6.2f}%")
    print(f"  Blank Images:      {blank_results['accuracy']*100:6.2f}%")
    print(f"  Noise Images:      {noise_results['accuracy']*100:6.2f}%")
    print(f"  Random Chance:     {random_baseline*100:6.2f}%")
    print()

    # Compute drops
    normal_acc = normal_results['accuracy']
    shuffled_acc = shuffled_results['accuracy']
    blank_acc = blank_results['accuracy']
    noise_acc = noise_results['accuracy']

    drop_shuffled = (1 - shuffled_acc / max(0.001, normal_acc)) * 100
    drop_blank = (1 - blank_acc / max(0.001, normal_acc)) * 100
    drop_noise = (1 - noise_acc / max(0.001, normal_acc)) * 100

    print(f"  Drop with shuffled pixels: {drop_shuffled:+.1f}%")
    print(f"  Drop with blank images:    {drop_blank:+.1f}%")
    print(f"  Drop with noise images:    {drop_noise:+.1f}%")
    print()

    # Verdict
    if normal_acc > shuffled_acc * 2 and shuffled_acc < random_baseline * 1.5:
        verdict = "PASS"
        explanation = "Model performance drops significantly when visual structure is destroyed."
        print(f"  Verdict: {verdict}")
        print(f"  {explanation}")
    elif normal_acc > shuffled_acc * 1.3:
        verdict = "PARTIAL"
        explanation = "Model shows some reliance on visual information, but not strong."
        print(f"  Verdict: {verdict}")
        print(f"  {explanation}")
    else:
        verdict = "FAIL"
        explanation = "Model performance similar with shuffled pixels - may not use vision."
        print(f"  Verdict: {verdict}")
        print(f"  {explanation}")

    results["verdict"] = verdict
    results["explanation"] = explanation
    results["drop_shuffled"] = drop_shuffled
    results["drop_blank"] = drop_blank
    results["drop_noise"] = drop_noise

    # Save outputs
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = os.path.join(args.output, f"vision_sanity_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] JSON: {json_path}")

    # Save PNG
    png_path = os.path.join(args.output, f"vision_sanity_{timestamp}.png")
    create_sanity_chart(results, png_path)

    pygame.quit()
    print("\nDone!")


if __name__ == "__main__":
    main()
