# GENREG Vision-Language Model - Evaluation & Analysis

This repository contains evaluation scripts and the best checkpoint from a GENREG (Genetic Neural Regulation) vision-language grounding model trained entirely through evolution (no backpropagation).

## 🎯 What This Is

A neural network that:
- **Reads text as pixels** (400×100 grayscale images)
- **Predicts the next word** in a sentence
- Was **trained by evolution**, not gradient descent
- Achieves **72.16% accuracy** with **100% neuron saturation**
- Uses only **1 hidden layer** with 24 neurons

The model naturally converged to a fully saturated (binary) regime - something gradient descent actively avoids but evolution discovered on its own.

## 📊 Key Results

- **Accuracy:** 72.16% (beats 10.18% frequency baseline by 608.8%)
- **Vision-dependent:** Drops to 5.57% with shuffled pixels (92.3% drop)
- **Saturation:** 100% of neurons operate at extreme values (±1)
- **Architecture:** 40,000 inputs → 24 hidden → 439 outputs
- **Parameters:** ~970k total

## 📁 Repository Contents

```
├── checkpoints/
│   └── checkpoint_gen_1272976.pkl    # Best trained model
├── corpus/
│   └── blanks_1k.json                # Evaluation corpus (1,528 vocab)
├── eval.py                           # Standard evaluation
├── eval_baselines.py                 # Compare vs random/frequency baselines
├── eval_vision_sanity.py             # Test if model uses visual info
├── analyze_genome.py                 # Comprehensive network analysis
└── text_predictor.py                 # Interactive demo
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/genreg-vision-language
cd genreg-vision-language

# Install dependencies
pip install torch numpy pygame matplotlib scikit-learn
```

### Run Interactive Demo

```bash
python text_predictor.py
```

Type a partial sentence and watch the model predict the next word by "seeing" the text as an image.

## 📝 Evaluation Scripts

### 1. Standard Evaluation (`eval.py`)

Evaluates model accuracy on a held-out corpus.

```bash
# Interactive session selection
python eval.py

# With verbose output
python eval.py --verbose

# Custom corpus
python eval.py --corpus path/to/corpus.json
```

**Output:**
- Accuracy percentage
- Per-sentence predictions
- Confidence scores
- Timing information

---

### 2. Baseline Comparison (`eval_baselines.py`)

Compares model performance against statistical baselines.

```bash
python eval_baselines.py
```

**Baselines tested:**
- **Random Guess:** Picks random vocabulary word (~1% expected)
- **Frequency Baseline:** Always predicts most common word in corpus
- **Model:** Actual checkpoint predictions

**Output:**
- `eval_results/baselines_YYYYMMDD_HHMMSS.json` - Metrics
- `eval_results/baselines_YYYYMMDD_HHMMSS.png` - Bar chart

**Example results:**
```
Random Guess:        0.99%
Frequency Baseline: 10.18%
Model:              72.16%
```

---

### 3. Vision Sanity Check (`eval_vision_sanity.py`)

Tests whether the model actually uses visual information or just memorizes language statistics.

```bash
python eval_vision_sanity.py
```

**Test conditions:**
- **Normal:** Original rendered text images
- **Shuffled Pixels:** Same pixels but randomly permuted
- **Blank:** Uniform gray (no information)
- **Noise:** Random pixel values

**Output:**
- `eval_results/vision_sanity_YYYYMMDD_HHMMSS.json` - Metrics
- `eval_results/vision_sanity_YYYYMMDD_HHMMSS.png` - Comparison chart

**Example results:**
```
Normal Images:      72.16%
Shuffled Pixels:     5.57%  (92.3% drop)
Blank Images:        9.28%  (87.1% drop)
Noise Images:        4.61%  (93.6% drop)
```

**Verdict:** Model demonstrates strong reliance on visual information.

---

### 4. Genome Analysis (`analyze_genome.py`)

Comprehensive analysis generating 10 visualization pages revealing network behavior.

```bash
python analyze_genome.py
```

**Generates:**
- `analysis_results/00_summary_dashboard.png` - Overall metrics
- `analysis_results/01_weight_heatmaps.png` - Weight distributions
- `analysis_results/02_neuron_saturation.png` - Activation ranges
- `analysis_results/03_hidden_patterns.png` - Neuron correlations
- `analysis_results/04_output_analysis.png` - Prediction behavior
- `analysis_results/05_neuron_importance.png` - Feature significance
- `analysis_results/07_spatial_sensitivity.png` - Input weight patterns
- `analysis_results/08_prediction_analysis.png` - Accuracy trends
- `analysis_results/09_weight_distributions.png` - Parameter statistics
- `analysis_results/analysis_report.json` - Raw metrics

**Key findings revealed:**
- 100% neuron saturation rate
- 4/24 neurons locked (act as learned biases)
- Extreme weight values (range: -679 to +634)
- 99.5% mean prediction confidence

---

### 5. Interactive Predictor (`text_predictor.py`)

Live demo showing what the model "sees" and predicts.

```bash
python text_predictor.py
```

**Features:**
- Type partial sentences
- See rendered image (what model sees)
- Get top-5 predictions with confidence
- Real-time inference

**Example:**
```
Input:  the cat sat on the ____
Sees:   [400×100 pixel image displayed]
Predicts:
  1. mat    (99.8%)
  2. floor  (98.2%)
  3. rug    (95.1%)
```

## 🔬 Understanding the Results

### Why 100% Saturation Matters

In traditional neural networks trained with gradient descent, saturation (neurons stuck at extreme values like ±1) is considered a failure state because:
- Gradients vanish (derivative of tanh at ±∞ is ~0)
- Learning stops
- Networks try to avoid it with careful initialization and normalization

**But this model thrives on it.**

Evolution doesn't need gradients, so it can exploit saturated regimes that gradient descent literally cannot see. The fully saturated neurons act as **binary feature detectors** - each asking a yes/no question about the visual input.

### The Natural Convergence Finding

**What's novel:** The model wasn't forced to be binary. There were:
- No weight clipping
- No quantization tricks
- No Straight-Through Estimators (used in Binarized Neural Networks)

Evolution **chose** full saturation on its own, suggesting this is actually the optimal state for the task - something gradient descent can't reach because it requires maintaining gradient flow.

### Single Layer Success

The model achieves 72% accuracy with just **one hidden layer**. Modern vision-language models like CLIP use hundreds of layers. 

**Hypothesis:** Gradient descent may need depth partly to work around saturation - by distributing computation across many layers with moderate activations, gradients can flow. Evolution doesn't have this constraint.

## 📦 Checkpoint Details

**File:** `checkpoints/checkpoint_gen_1272976.pkl`

**Training:**
- Generations: 1,272,976
- Method: Trust-based evolutionary selection
- No backpropagation or gradient calculations
- Direct pixel-to-word mapping (no pre-trained encoders)

**Architecture:**
```python
{
    "input_size": 40000,      # 400×100 pixels flattened
    "hidden_size": 24,        # Single hidden layer
    "output_size": 439,       # Vocabulary size
    "activation": "tanh",
    "total_parameters": 970999
}
```

## 🧪 Reproducibility

All evaluation scripts use the same checkpoint and configurations. To reproduce results:

1. Run all evaluations:
```bash
python eval_baselines.py      # Get baseline comparison
python eval_vision_sanity.py  # Verify vision dependency
python analyze_genome.py      # Generate full analysis
```

2. Check `eval_results/` for outputs

3. Compare against reported metrics:
   - Accuracy: 72.16%
   - Vision drop: 92.3%
   - Saturation: 100%

## 🔧 Configuration

Key parameters (defined in `config.py` or script headers):

```python
FIELD_WIDTH = 400          # Image width
FIELD_HEIGHT = 100         # Image height  
FONT_SIZE = 16             # Text rendering size
BLANK_MARKER = "[____]"    # Fill-in-the-blank indicator
```

## 📖 Citation

If you use this code or findings, please cite:

```bibtex
@misc{genreg2026,
  title={Natural Convergence to Binary Neural Networks via Evolution},
  author={[Your Name]},
  year={2026},
  note={Vision-language grounding without backpropagation}
}
```

## 🤝 Contributing

Found interesting patterns in the analysis? Have ideas for extensions? Open an issue or PR!

## 📜 License

[Your chosen license - MIT suggested for research code]

## 🔗 Related Work

- **Binarized Neural Networks (BNNs):** Gradient-trained binary networks requiring Straight-Through Estimators
- **NEAT/HyperNEAT:** Evolutionary neural architecture search
- **CLIP:** Vision-language models (uses massive transformers + gradient descent)

**Key difference:** This model naturally discovered the binary regime through evolution, without coercion.

## ❓ FAQ

**Q: Why is 100% saturation good if it breaks gradient descent?**  
A: It's only "bad" because gradient descent can't handle it. Evolution proved that fully saturated networks can achieve strong performance - we've been avoiding them due to optimizer constraints, not task requirements.

**Q: Does this scale to larger vocabularies/tasks?**  
A: Unknown. This is proof-of-concept at 439 vocab. Scaling is an open question.

**Q: Why use evolution instead of gradient descent?**  
A: Not claiming evolution is "better" - it's 1000x slower. But it can explore solution spaces that gradient descent is blind to, revealing what might be possible if we could overcome the saturation barrier.

**Q: Can gradient descent reach this regime?**  
A: Probably not naturally. You'd need tricks like Straight-Through Estimators (fake gradients) or quantization-aware training. The point is evolution gets there without any hacks.

## 📧 Contact

[Your contact information]

---

**Note:** This is research code released for reproducibility and exploration. The trained checkpoint and evaluation scripts allow verification of the reported findings.
