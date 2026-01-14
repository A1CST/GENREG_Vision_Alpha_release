# ================================================================
# GENREG - Extract Best Genome
# ================================================================
# Extracts the best genome from a checkpoint into a minimal file
# that only contains the weights needed for inference.
#
# Output format is compatible with text_predictor.py
# ================================================================

import os
import sys
import pickle
import argparse
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("[ERROR] NumPy required")
    sys.exit(1)

from genreg_visual_env import get_vocabulary
from genreg_checkpoint import load_checkpoint, get_latest_checkpoint, list_checkpoints


def extract_best_genome(checkpoint_path, output_dir="best_genomes/grv"):
    """
    Extract the best genome from a checkpoint file.

    Args:
        checkpoint_path: path to checkpoint .pkl file
        output_dir: directory to save extracted genome

    Returns:
        output_path: path to saved genome file
    """
    print(f"[EXTRACT] Loading checkpoint: {checkpoint_path}")

    # Load full checkpoint
    population, generation, template, phase_state = load_checkpoint(checkpoint_path)

    # Find best genome by trust
    best_genome = max(population.genomes, key=lambda g: g.trust)
    controller = best_genome.controller

    print(f"[EXTRACT] Found {len(population.genomes)} genomes")
    print(f"[EXTRACT] Best genome trust: {best_genome.trust:.2f}")
    print(f"[EXTRACT] Generation: {generation}")

    # Extract controller weights
    # Handle both torch tensors and numpy/list formats
    if hasattr(controller, '_use_torch') and controller._use_torch:
        w1 = controller.w1.cpu().numpy().tolist()
        b1 = controller.b1.cpu().numpy().tolist()
        w2 = controller.w2.cpu().numpy().tolist()
        b2 = controller.b2.cpu().numpy().tolist()
    elif hasattr(controller.w1, 'tolist'):
        # Already numpy arrays
        w1 = controller.w1.tolist()
        b1 = controller.b1.tolist()
        w2 = controller.w2.tolist()
        b2 = controller.b2.tolist()
    else:
        # Already lists
        w1 = controller.w1
        b1 = controller.b1
        w2 = controller.w2
        b2 = controller.b2

    # Load vocabulary
    vocabulary = get_vocabulary()
    print(f"[EXTRACT] Vocabulary: {len(vocabulary)} words")

    # Build minimal genome package
    genome_data = {
        "controller": {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "input_size": controller.input_size,
            "hidden_size": controller.hidden_size,
            "output_size": controller.output_size,
        },
        "vocabulary": vocabulary,
        "genome": {
            "id": best_genome.id,
            "trust": best_genome.trust,
        },
        "extraction_info": {
            "source_checkpoint": checkpoint_path,
            "generation": generation,
            "extraction_date": datetime.now().isoformat(),
            "population_size": len(population.genomes),
            "num_genomes_in_checkpoint": len(population.genomes),
        }
    }

    # Calculate approximate size
    w1_size = len(w1) * len(w1[0]) if w1 else 0
    w2_size = len(w2) * len(w2[0]) if w2 else 0
    total_weights = w1_size + len(b1) + w2_size + len(b2)

    print(f"[EXTRACT] Network architecture: {controller.input_size} -> {controller.hidden_size} -> {controller.output_size}")
    print(f"[EXTRACT] Total weights: {total_weights:,}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with generation and trust
    trust_str = f"{best_genome.trust:.0f}".replace("-", "neg")
    filename = f"best_genome_grv-blankfill_gen{generation}_trust{trust_str}.pkl"
    output_path = os.path.join(output_dir, filename)

    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(genome_data, f)

    # Get file size
    file_size = os.path.getsize(output_path)

    print(f"\n[SAVED] {output_path}")
    print(f"[SAVED] File size: {file_size / 1024:.1f} KB")

    return output_path, genome_data


def print_genome_info(genome_data):
    """Print detailed info about extracted genome."""
    print("\n" + "=" * 60)
    print("EXTRACTED GENOME INFO")
    print("=" * 60)

    controller = genome_data["controller"]
    genome = genome_data["genome"]
    info = genome_data.get("extraction_info", {})
    vocab = genome_data.get("vocabulary", [])

    print(f"\n  Architecture:")
    print(f"    Input size:  {controller['input_size']:,} (pixels)")
    print(f"    Hidden size: {controller['hidden_size']}")
    print(f"    Output size: {controller['output_size']}")

    print(f"\n  Performance:")
    print(f"    Trust: {genome['trust']:.2f}")
    print(f"    Generation: {info.get('generation', 'unknown')}")

    print(f"\n  Vocabulary:")
    print(f"    Size: {len(vocab)} words")
    if vocab:
        print(f"    First 10: {vocab[:10]}")
        print(f"    Last 10: {vocab[-10:]}")

    print(f"\n  Source:")
    print(f"    Checkpoint: {info.get('source_checkpoint', 'unknown')}")
    print(f"    Extracted: {info.get('extraction_date', 'unknown')}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract best genome from checkpoint for inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_best_genome.py                              # Use latest checkpoint
  python extract_best_genome.py -c path/to/checkpoint.pkl   # Specific checkpoint
  python extract_best_genome.py --list                      # List available checkpoints
  python extract_best_genome.py -o my_genomes/              # Custom output directory

The extracted genome can be used with text_predictor.py:
  python text_predictor.py
        """
    )
    parser.add_argument("--checkpoint", "-c", metavar="PATH",
                        help="Path to checkpoint file")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available checkpoints")
    parser.add_argument("--output", "-o", metavar="DIR", default="best_genomes/grv",
                        help="Output directory (default: best_genomes/grv)")
    parser.add_argument("--info", "-i", action="store_true",
                        help="Print detailed info about extracted genome")

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
                        print(f"\n  {session}:")
                        print(f"    Checkpoints: {len(checkpoints)}")
                        print(f"    Latest: gen {gen:,}")
                        print(f"    Path: {latest}")
        else:
            print(f"  No checkpoint directory found at: {base_dir}")
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
    print("EXTRACT BEST GENOME")
    print("=" * 60)

    # Extract
    output_path, genome_data = extract_best_genome(checkpoint_path, args.output)

    # Print info if requested
    if args.info:
        print_genome_info(genome_data)

    print("\n[DONE] Genome extracted successfully!")
    print(f"[DONE] Use with: python text_predictor.py")


if __name__ == "__main__":
    main()
