#!/usr/bin/env python3
# ================================================================
# GENREG Evaluation Script
# ================================================================
# Loads a checkpoint and evaluates model performance on a held-out
# evaluation corpus.
#
# Usage:
#   python eval.py                    # Interactive session selection
#   python eval.py --verbose          # With detailed output
#
# Examples:
#   python eval.py
#   python eval.py --corpus corpus/blanks_eval.json --verbose
# ================================================================

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

# Check for required dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[ERROR] NumPy is required for evaluation")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ERROR] PyTorch is required for evaluation")
    sys.exit(1)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] Pygame not available - using cached rendering if possible")

# Import project modules
import config as cfg
from genreg_checkpoint import load_checkpoint
from genreg_visual_env import extract_vocabulary, load_corpus


# ================================================================
# SESSION AND CHECKPOINT DISCOVERY
# ================================================================
def find_sessions(checkpoint_base="checkpoints_visual_gpu"):
    """Find all training sessions with checkpoints."""
    if not os.path.exists(checkpoint_base):
        return []

    sessions = []
    for name in sorted(os.listdir(checkpoint_base)):
        session_path = os.path.join(checkpoint_base, name)
        if os.path.isdir(session_path) and name.startswith("session_"):
            # Count checkpoints
            checkpoints = glob.glob(os.path.join(session_path, "checkpoint_gen_*.pkl"))
            if checkpoints:
                # Get latest generation
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
    """Interactively select a session from the list."""
    if not sessions:
        print("[ERROR] No sessions found in checkpoints_visual_gpu/")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("AVAILABLE SESSIONS")
    print("=" * 60)

    for i, session in enumerate(sessions):
        print(f"  [{i + 1}] {session['name']}")
        print(f"      Checkpoints: {session['num_checkpoints']}, Latest: gen {session['latest_gen']:,}")

    print()

    if len(sessions) == 1:
        print(f"[AUTO] Only one session found, selecting: {sessions[0]['name']}")
        return sessions[0]

    while True:
        try:
            choice = input(f"Select session [1-{len(sessions)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]
            print(f"Please enter a number between 1 and {len(sessions)}")
        except ValueError:
            print("Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)


def get_latest_checkpoint(session_path):
    """Get the latest checkpoint from a session directory."""
    checkpoints = glob.glob(os.path.join(session_path, "checkpoint_gen_*.pkl"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(os.path.basename(p).split("_")[2].split(".")[0]))


# ================================================================
# EVAL CORPUS LOADING
# ================================================================
def load_eval_corpus(path=None):
    """Load evaluation corpus from JSON file."""
    if path is None:
        path = "corpus/blanks_eval.json"

    # Handle relative paths
    if not os.path.isabs(path):
        if not os.path.exists(path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, path)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            sentences = data.get("sentences", [])
            print(f"[EVAL] Loaded {len(sentences)} eval sentences from {path}")
            return sentences
    except FileNotFoundError:
        print(f"[ERROR] Eval corpus not found at {path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in {path}")
        sys.exit(1)


# ================================================================
# IMAGE RENDERING
# ================================================================
class EvalImageRenderer:
    """Renders sentences as images for evaluation."""

    def __init__(self, corpus, width, height, font_size, device, blank_marker="[____]"):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is required for rendering")

        self.device = device
        self.width = width
        self.height = height
        self.num_sentences = len(corpus)

        # Initialize pygame for rendering
        pygame.init()
        surface = pygame.Surface((width, height))
        font = pygame.font.Font(None, font_size)

        print(f"[EVAL] Rendering {len(corpus)} sentences...")
        start_time = time.time()

        images_np = []
        self.answers = []
        self.sentences = []

        for i, entry in enumerate(corpus):
            sentence = entry["text"]
            answers = [a.lower() for a in entry["answers"]]

            # Render
            surface.fill((25, 25, 30))
            display_text = sentence.replace("____", blank_marker)
            text_surface = font.render(display_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(width // 2, height // 2))
            surface.blit(text_surface, text_rect)

            # Capture
            pixels = pygame.surfarray.array3d(surface)
            grayscale = np.mean(pixels, axis=2)
            normalized = (grayscale / 255.0).flatten().astype(np.float32)
            images_np.append(normalized)

            self.answers.append(set(answers))
            self.sentences.append(sentence)

        # Convert to GPU tensor
        images_np = np.stack(images_np)
        self.images = torch.from_numpy(images_np).to(device)

        elapsed = time.time() - start_time
        print(f"[EVAL] Rendered {len(corpus)} images in {elapsed:.2f}s")

        pygame.quit()


# ================================================================
# EVALUATION LOGIC
# ================================================================
def evaluate_genome(genome, images, vocabulary, answers, device, use_sampling=False):
    """
    Evaluate a single genome on all sentences.

    Args:
        genome: GENREGGenome instance
        images: (num_sentences, input_size) tensor
        vocabulary: list of vocabulary words
        answers: list of sets of valid answers for each sentence
        device: torch device
        use_sampling: if True, sample from distribution; if False, use argmax

    Returns:
        dict with evaluation metrics
    """
    controller = genome.controller
    num_sentences = images.shape[0]
    vocab_size = len(vocabulary)

    # Get weights
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

    # Forward pass: (num_sentences, input_size) -> (num_sentences, output_size)
    hidden = torch.tanh(images @ w1.T + b1)
    outputs = hidden @ w2.T + b2

    # Get vocabulary probabilities
    vocab_logits = outputs[:, :vocab_size]
    vocab_probs = F.softmax(vocab_logits, dim=1)

    # Get predictions
    if use_sampling:
        word_indices = torch.multinomial(vocab_probs, 1).squeeze(1)
    else:
        word_indices = torch.argmax(vocab_probs, dim=1)

    # Evaluate correctness
    correct = 0
    predictions = []

    for i in range(num_sentences):
        pred_idx = word_indices[i].item()
        pred_word = vocabulary[pred_idx] if pred_idx < len(vocabulary) else "<unk>"
        is_correct = pred_word in answers[i]

        if is_correct:
            correct += 1

        predictions.append({
            "sentence": i,
            "predicted": pred_word,
            "correct": is_correct,
            "valid_answers": list(answers[i]),
            "confidence": vocab_probs[i, pred_idx].item()
        })

    accuracy = correct / num_sentences if num_sentences > 0 else 0.0

    return {
        "correct": correct,
        "total": num_sentences,
        "accuracy": accuracy,
        "predictions": predictions
    }


def evaluate_population(population, images, vocabulary, answers, device, top_k=5, verbose=False):
    """
    Evaluate all genomes in population.

    Args:
        population: GENREGPopulation instance
        images: (num_sentences, input_size) tensor
        vocabulary: list of vocabulary words
        answers: list of sets of valid answers
        device: torch device
        top_k: number of top genomes to report in detail
        verbose: whether to print per-sentence results

    Returns:
        dict with population-level metrics
    """
    results = []

    print(f"\n[EVAL] Evaluating {len(population.genomes)} genomes on {images.shape[0]} sentences...")
    print("=" * 60)

    for i, genome in enumerate(population.genomes):
        result = evaluate_genome(genome, images, vocabulary, answers, device, use_sampling=False)
        result["genome_id"] = genome.id
        result["trust"] = genome.trust
        results.append(result)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Genome {i+1}/{len(population.genomes)}: {result['accuracy']*100:.1f}% ({result['correct']}/{result['total']})")

    # Sort by accuracy (then by trust as tiebreaker)
    results.sort(key=lambda x: (x["accuracy"], x["trust"]), reverse=True)

    # Population statistics
    accuracies = [r["accuracy"] for r in results]
    best_accuracy = max(accuracies)
    mean_accuracy = sum(accuracies) / len(accuracies)
    median_accuracy = sorted(accuracies)[len(accuracies) // 2]

    print("\n" + "=" * 60)
    print("POPULATION SUMMARY")
    print("=" * 60)
    print(f"  Best accuracy:   {best_accuracy * 100:.2f}%")
    print(f"  Mean accuracy:   {mean_accuracy * 100:.2f}%")
    print(f"  Median accuracy: {median_accuracy * 100:.2f}%")

    # Top-K genomes
    print(f"\n  Top {top_k} Genomes:")
    print(f"  {'Rank':<6} {'Genome ID':<12} {'Accuracy':<12} {'Trust':<12}")
    print(f"  {'-'*42}")
    for i, r in enumerate(results[:top_k]):
        print(f"  {i+1:<6} {r['genome_id']:<12} {r['accuracy']*100:.1f}%{'':<6} {r['trust']:.1f}")

    # Per-sentence analysis for best genome
    best_result = results[0]

    if verbose:
        print(f"\n" + "=" * 60)
        print(f"DETAILED RESULTS (Best Genome: {best_result['genome_id']})")
        print("=" * 60)

        for pred in best_result["predictions"]:
            status = "CORRECT" if pred["correct"] else "WRONG"
            print(f"  [{status}] Predicted: '{pred['predicted']}' ({pred['confidence']*100:.1f}%)")
            print(f"           Valid: {pred['valid_answers']}")

    # Error analysis
    errors_by_predicted = defaultdict(int)
    for pred in best_result["predictions"]:
        if not pred["correct"]:
            errors_by_predicted[pred["predicted"]] += 1

    if errors_by_predicted:
        print(f"\n  Most Common Wrong Predictions:")
        sorted_errors = sorted(errors_by_predicted.items(), key=lambda x: x[1], reverse=True)[:10]
        for word, count in sorted_errors:
            print(f"    '{word}': {count} times")

    return {
        "population_size": len(population.genomes),
        "num_sentences": images.shape[0],
        "best_accuracy": best_accuracy,
        "mean_accuracy": mean_accuracy,
        "median_accuracy": median_accuracy,
        "results": results,
        "best_genome": best_result
    }


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a GENREG checkpoint on a held-out corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py                    # Interactive session selection
  python eval.py --verbose          # Show per-sentence predictions
  python eval.py --all-genomes      # Evaluate all genomes, not just best
        """
    )
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to checkpoint file (.pkl) - if omitted, select interactively")
    parser.add_argument("--corpus", default="corpus/blanks_eval.json",
                        help="Path to evaluation corpus (default: corpus/blanks_eval.json)")
    parser.add_argument("--all-genomes", action="store_true",
                        help="Evaluate all genomes instead of just the best")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-sentence predictions for best genome")
    parser.add_argument("--device", default="auto",
                        help="Device to use: 'auto', 'cuda', or 'cpu' (default: auto)")
    parser.add_argument("--json-output", metavar="FILE",
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Select device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[EVAL] Using device: {device}")

    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Interactive session selection
        sessions = find_sessions()
        session = select_session_interactive(sessions)
        checkpoint_path = session["latest_checkpoint"]
        print(f"\n[EVAL] Selected: {session['name']}")

    # Load checkpoint
    print(f"[EVAL] Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    population, generation, template_proteins, phase_state = load_checkpoint(checkpoint_path)
    print(f"[EVAL] Loaded generation {generation:,} with {len(population.genomes)} genomes")

    # Find best genome by trust
    best_genome = max(population.genomes, key=lambda g: g.trust)
    print(f"[EVAL] Best genome: ID {best_genome.id}, Trust: {best_genome.trust:.1f}")

    # Load vocabulary (from training corpus to ensure consistency)
    vocabulary = extract_vocabulary(load_corpus())
    print(f"[EVAL] Vocabulary size: {len(vocabulary)}")

    # Load eval corpus
    eval_corpus = load_eval_corpus(args.corpus)

    # Render eval images
    renderer = EvalImageRenderer(
        eval_corpus,
        width=cfg.FIELD_WIDTH,
        height=cfg.FIELD_HEIGHT,
        font_size=cfg.FONT_SIZE,
        device=device
    )

    # Run evaluation
    start_time = time.time()

    if args.all_genomes:
        # Evaluate all genomes
        results = evaluate_population(
            population,
            renderer.images,
            vocabulary,
            renderer.answers,
            device,
            top_k=5,
            verbose=args.verbose
        )
        best_result = results["best_genome"]
    else:
        # Evaluate only the best genome
        print(f"\n[EVAL] Evaluating best genome on {renderer.images.shape[0]} sentences...")
        print("=" * 60)

        best_result = evaluate_genome(
            best_genome,
            renderer.images,
            vocabulary,
            renderer.answers,
            device,
            use_sampling=False
        )
        best_result["genome_id"] = best_genome.id
        best_result["trust"] = best_genome.trust

        print(f"\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Genome ID:  {best_genome.id}")
        print(f"  Trust:      {best_genome.trust:.1f}")
        print(f"  Accuracy:   {best_result['accuracy'] * 100:.2f}% ({best_result['correct']}/{best_result['total']})")

        # Show predictions if verbose
        if args.verbose:
            print(f"\n  Per-Sentence Results:")
            print(f"  {'-' * 56}")
            for pred in best_result["predictions"]:
                status = "OK" if pred["correct"] else "X "
                sentence_preview = renderer.sentences[pred["sentence"]][:40]
                print(f"  [{status}] '{pred['predicted']}' ({pred['confidence']*100:.0f}%) <- {sentence_preview}...")
                if not pred["correct"]:
                    print(f"       Valid: {pred['valid_answers']}")

        # Error analysis
        errors_by_predicted = defaultdict(int)
        for pred in best_result["predictions"]:
            if not pred["correct"]:
                errors_by_predicted[pred["predicted"]] += 1

        if errors_by_predicted:
            print(f"\n  Most Common Errors:")
            sorted_errors = sorted(errors_by_predicted.items(), key=lambda x: x[1], reverse=True)[:5]
            for word, count in sorted_errors:
                print(f"    '{word}': {count} times")

        results = {
            "best_accuracy": best_result["accuracy"],
            "num_sentences": best_result["total"]
        }

    elapsed = time.time() - start_time

    print(f"\n[EVAL] Completed in {elapsed:.2f}s")

    # Save JSON output if requested
    if args.json_output:
        output_data = {
            "checkpoint": checkpoint_path,
            "generation": generation,
            "corpus": args.corpus,
            "num_sentences": best_result["total"],
            "best_genome_id": best_result["genome_id"],
            "best_genome_trust": best_result["trust"],
            "accuracy": best_result["accuracy"],
            "correct": best_result["correct"],
            "predictions": best_result["predictions"]
        }

        with open(args.json_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"[EVAL] Results saved to {args.json_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
