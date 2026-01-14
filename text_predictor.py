"""
Visual Text Predictor - A visually-grounded language model

This model "sees" text as pixels, not tokens. It was trained through evolution
(no backpropagation) to fill in blanks in sentences by viewing them as rendered images.

Usage:
    python text_predictor.py

Type a partial sentence, and the model will predict the next word.
The pygame window shows exactly what the model "sees".
"""

import os
import json
import pickle
import glob
import numpy as np

# Check for optional torch (falls back to numpy if not available)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[INFO] PyTorch not available, using NumPy for inference")

import pygame

# ================================================================
# CONFIGURATION (must match training)
# ================================================================
FIELD_WIDTH = 400
FIELD_HEIGHT = 100
FONT_SIZE = 16
BLANK_MARKER = "[____]"
BACKGROUND_COLOR = (25, 25, 30)
TEXT_COLOR = (255, 255, 255)

# Paths
GENOME_DIR = "best_genomes/grv"
CHECKPOINT_DIR = "checkpoints_visual_gpu"
CORPUS_PATH = "corpus/blanks.json"


def load_vocabulary_from_corpus():
    """Extract vocabulary from corpus file."""
    if not os.path.exists(CORPUS_PATH):
        print(f"[WARN] Corpus not found at {CORPUS_PATH}")
        return []

    with open(CORPUS_PATH, 'r') as f:
        data = json.load(f)

    vocab = set()
    for sentence in data.get("sentences", []):
        for answer in sentence.get("answers", []):
            vocab.add(answer.lower().strip())

    return sorted(vocab)


def find_extracted_genomes():
    """Find pre-extracted genome files."""
    pattern = os.path.join(GENOME_DIR, "best_genome_grv-blankfill_*.pkl")
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)
    return files


def find_checkpoint_sessions():
    """Find all checkpoint sessions with their latest checkpoint."""
    sessions = []
    if not os.path.exists(CHECKPOINT_DIR):
        return sessions

    for session_name in os.listdir(CHECKPOINT_DIR):
        session_path = os.path.join(CHECKPOINT_DIR, session_name)
        if not os.path.isdir(session_path):
            continue

        # Find all checkpoints in this session
        checkpoints = glob.glob(os.path.join(session_path, "checkpoint_gen_*.pkl"))
        if not checkpoints:
            continue

        # Sort by generation number (extracted from filename)
        def get_gen(path):
            name = os.path.basename(path)
            try:
                return int(name.replace("checkpoint_gen_", "").replace(".pkl", ""))
            except:
                return 0

        checkpoints.sort(key=get_gen, reverse=True)
        latest = checkpoints[0]
        latest_gen = get_gen(latest)

        sessions.append({
            "name": session_name,
            "path": latest,
            "generation": latest_gen,
        })

    # Sort by session name (most recent first)
    sessions.sort(key=lambda x: x["name"], reverse=True)
    return sessions


def extract_best_from_checkpoint(checkpoint_path):
    """
    Extract the best genome from a full checkpoint file.

    Returns:
        weights: dict with w1, b1, w2, b2
        info: metadata
    """
    print(f"[EXTRACT] Loading checkpoint: {checkpoint_path}")

    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    genomes = data.get("genomes", [])
    if not genomes:
        raise ValueError("No genomes found in checkpoint")

    # Find best genome by trust
    best = max(genomes, key=lambda g: g.get("trust", 0))

    print(f"[EXTRACT] Found {len(genomes)} genomes, best trust: {best.get('trust', 'unknown')}")

    controller = best["controller"]
    weights = {
        "w1": np.array(controller["w1"], dtype=np.float32),
        "b1": np.array(controller["b1"], dtype=np.float32),
        "w2": np.array(controller["w2"], dtype=np.float32),
        "b2": np.array(controller["b2"], dtype=np.float32),
    }

    info = {
        "input_size": controller.get("input_size", data.get("input_size")),
        "hidden_size": controller.get("hidden_size", data.get("hidden_size")),
        "output_size": controller.get("output_size", data.get("output_size")),
        "trust": best.get("trust", "unknown"),
        "generation": data.get("generation", "unknown"),
        "source": "checkpoint",
    }

    return weights, info


def select_genome_source():
    """
    Interactive menu to select genome source.

    Returns:
        weights: dict with w1, b1, w2, b2
        vocabulary: list of words
        info: metadata
    """
    print("=" * 60)
    print("GENOME SOURCE SELECTION")
    print("=" * 60)

    # Find available sources
    extracted = find_extracted_genomes()
    sessions = find_checkpoint_sessions()

    options = []

    # Add extracted genomes
    for i, path in enumerate(extracted):
        name = os.path.basename(path)
        options.append({
            "type": "extracted",
            "path": path,
            "label": f"[Extracted] {name}",
        })

    # Add checkpoint sessions
    for session in sessions:
        options.append({
            "type": "checkpoint",
            "path": session["path"],
            "label": f"[Checkpoint] {session['name']} (gen {session['generation']})",
        })

    if not options:
        print("[ERROR] No genome sources found!")
        print(f"  - No extracted genomes in {GENOME_DIR}")
        print(f"  - No checkpoints in {CHECKPOINT_DIR}")
        return None, None, None

    # Display menu
    print("\nAvailable sources:")
    for i, opt in enumerate(options):
        print(f"  {i + 1}. {opt['label']}")

    print(f"\n  0. Exit")
    print()

    # Get selection
    while True:
        try:
            choice = input("Select source [1]: ").strip()
            if choice == "":
                choice = 1
            elif choice == "0":
                return None, None, None
            else:
                choice = int(choice)

            if 1 <= choice <= len(options):
                break
            print(f"Please enter 1-{len(options)} or 0 to exit")
        except ValueError:
            print("Please enter a number")

    selected = options[choice - 1]
    print()

    # Load based on type
    if selected["type"] == "extracted":
        return load_genome(selected["path"])
    else:
        weights, info = extract_best_from_checkpoint(selected["path"])
        # Load vocabulary from corpus since checkpoints don't include it
        vocabulary = load_vocabulary_from_corpus()
        print(f"[VOCAB] Loaded {len(vocabulary)} words from corpus")
        return weights, vocabulary, info


def load_genome(path):
    """
    Load a trained genome from pickle file.

    Returns:
        weights: dict with w1, b1, w2, b2
        vocabulary: list of words the model can predict
        info: metadata about the genome
    """
    print(f"[LOAD] Loading genome from: {path}")

    with open(path, 'rb') as f:
        data = pickle.load(f)

    controller = data["controller"]
    vocabulary = data.get("vocabulary", [])

    weights = {
        "w1": np.array(controller["w1"], dtype=np.float32),
        "b1": np.array(controller["b1"], dtype=np.float32),
        "w2": np.array(controller["w2"], dtype=np.float32),
        "b2": np.array(controller["b2"], dtype=np.float32),
    }

    info = {
        "input_size": controller["input_size"],
        "hidden_size": controller["hidden_size"],
        "output_size": controller["output_size"],
        "trust": data.get("genome", {}).get("trust", "unknown"),
        "extraction_info": data.get("extraction_info", {}),
    }

    print(f"[LOAD] Architecture: {info['input_size']} -> {info['hidden_size']} -> {info['output_size']}")
    print(f"[LOAD] Vocabulary: {len(vocabulary)} words")
    print(f"[LOAD] Trust: {info['trust']}")

    return weights, vocabulary, info


class VisualTextPredictor:
    """
    A text predictor that "sees" text as pixels.

    The neural network was evolved to recognize visual patterns in rendered text
    and predict missing words. No tokenization - pure visual perception.
    """

    def __init__(self, weights, vocabulary, use_torch=True):
        """
        Initialize the predictor.

        Args:
            weights: dict with w1, b1, w2, b2 numpy arrays
            vocabulary: list of words the model can output
            use_torch: use PyTorch if available (faster on GPU)
        """
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((FIELD_WIDTH, FIELD_HEIGHT))
        pygame.display.set_caption("Visual Text Predictor - What the model sees")
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.render_surface = pygame.Surface((FIELD_WIDTH, FIELD_HEIGHT))

        # Setup weights
        self.use_torch = use_torch and TORCH_AVAILABLE
        if self.use_torch:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.w1 = torch.tensor(weights["w1"], dtype=torch.float32, device=device)
            self.b1 = torch.tensor(weights["b1"], dtype=torch.float32, device=device)
            self.w2 = torch.tensor(weights["w2"], dtype=torch.float32, device=device)
            self.b2 = torch.tensor(weights["b2"], dtype=torch.float32, device=device)
            print(f"[INIT] Using PyTorch on {device}")
        else:
            self.w1 = weights["w1"]
            self.b1 = weights["b1"]
            self.w2 = weights["w2"]
            self.b2 = weights["b2"]
            print("[INIT] Using NumPy")

        print(f"[INIT] Ready! Type a partial sentence to predict the next word.")
        print(f"[INIT] Type 'quit', 'exit', or 'q' to exit.")
        print()

    def render_text(self, text):
        """
        Render text to the pygame surface and return pixel observation.

        Args:
            text: string to render (will have blank marker appended)

        Returns:
            numpy array of shape (40000,) - normalized grayscale pixels
        """
        # Fill background
        self.render_surface.fill(BACKGROUND_COLOR)

        # Render text centered
        text_surface = self.font.render(text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(FIELD_WIDTH // 2, FIELD_HEIGHT // 2))
        self.render_surface.blit(text_surface, text_rect)

        # Update display window
        self.screen.blit(self.render_surface, (0, 0))
        pygame.display.flip()

        # Convert to pixel observation
        pixels = pygame.surfarray.array3d(self.render_surface)  # (400, 100, 3)
        grayscale = np.mean(pixels, axis=2)  # (400, 100)
        normalized = (grayscale / 255.0).flatten().astype(np.float32)  # (40000,)

        return normalized

    def forward(self, observation):
        """
        Neural network forward pass.

        Args:
            observation: numpy array of shape (40000,)

        Returns:
            probabilities: array of shape (vocab_size,)
        """
        if self.use_torch:
            x = torch.tensor(observation, dtype=torch.float32, device=self.device)

            # Hidden layer: tanh(w1 @ x + b1)
            hidden = torch.tanh(self.w1 @ x + self.b1)

            # Output layer: w2 @ hidden + b2
            logits = self.w2 @ hidden + self.b2

            # Slice to vocabulary size if needed
            logits = logits[:self.vocab_size]

            # Softmax
            probs = torch.softmax(logits, dim=0)

            return probs.cpu().numpy()
        else:
            # NumPy implementation
            # Hidden layer
            hidden = np.tanh(self.w1 @ observation + self.b1)

            # Output layer
            logits = self.w2 @ hidden + self.b2

            # Slice to vocabulary size
            logits = logits[:self.vocab_size]

            # Softmax (with numerical stability)
            max_logit = np.max(logits)
            exp_logits = np.exp(logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)

            return probs

    def predict(self, user_input, temperature=1.0, use_argmax=False):
        """
        Predict the next word given user input.

        Args:
            user_input: partial sentence from user
            temperature: sampling temperature (higher = more random)
            use_argmax: if True, always pick highest probability word

        Returns:
            predicted_word: string
            confidence: probability of the predicted word
        """
        # Prepare display text with blank marker
        display_text = f"{user_input} {BLANK_MARKER}"

        # Render and get observation
        observation = self.render_text(display_text)

        # Forward pass
        probs = self.forward(observation)

        if use_argmax:
            # Deterministic: pick highest probability
            word_idx = np.argmax(probs)
        else:
            # Stochastic: sample from distribution
            if temperature != 1.0:
                # Apply temperature
                log_probs = np.log(probs + 1e-10)
                scaled_probs = np.exp(log_probs / temperature)
                probs = scaled_probs / np.sum(scaled_probs)

            word_idx = np.random.choice(len(probs), p=probs)

        predicted_word = self.vocabulary[word_idx]
        confidence = probs[word_idx]

        return predicted_word, confidence

    def run(self):
        """Main interaction loop."""
        print("=" * 60)
        print("VISUAL TEXT PREDICTOR")
        print("=" * 60)
        print("Type a partial sentence. The model will predict the next word.")
        print("The pygame window shows what the model 'sees'.")
        print()
        print("Commands:")
        print("  quit/exit/q  - Exit the program")
        print("  !argmax      - Toggle deterministic mode (highest prob)")
        print("  !temp X      - Set temperature (default 1.0)")
        print("=" * 60)
        print()

        use_argmax = False
        temperature = 1.0

        running = True
        while running:
            # Process pygame events (needed to keep window responsive)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input == '!argmax':
                use_argmax = not use_argmax
                mode = "deterministic (argmax)" if use_argmax else "stochastic (sampling)"
                print(f"[MODE] Switched to {mode}")
                continue

            if user_input.startswith('!temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"[TEMP] Temperature set to {temperature}")
                except (ValueError, IndexError):
                    print("[ERROR] Usage: !temp 0.5")
                continue

            if not user_input:
                continue

            # Predict
            predicted_word, confidence = self.predict(
                user_input,
                temperature=temperature,
                use_argmax=use_argmax
            )

            # Show result
            completed = f"{user_input} {predicted_word}"
            print(f"    {completed}")
            print(f"    (confidence: {confidence:.1%})")
            print()

        pygame.quit()


def main():
    """Entry point."""
    # Select and load genome
    weights, vocabulary, info = select_genome_source()

    if weights is None:
        print("Exiting.")
        return

    if not vocabulary:
        print("[ERROR] No vocabulary found! Cannot predict words.")
        return

    # Create predictor and run
    predictor = VisualTextPredictor(weights, vocabulary)
    predictor.run()


if __name__ == "__main__":
    main()
