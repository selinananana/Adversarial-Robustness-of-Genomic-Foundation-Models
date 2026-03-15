# Genomic Adversarial Robustness 

> A reinforcement-learning pipeline for adversarial robustness evaluation of DNA sequence classifiers using biologically constrained synonymous codon substitutions.

---

## Overview

This project investigates whether genomic foundation models can be fooled by **synonymous nucleotide mutations** — edits that change the DNA sequence but leave the encoded protein intact. A PPO agent navigates the synonymous-substitution space to find minimal edit paths that defeat a classifier.

**Key question:** How many silent codon swaps does it take to make a promoter look like a non-promoter to a deep learning model?


## Models

| Model | Source | Architecture | Tokenisation |
|-------|--------|--------------|--------------|
| **DNABERT** | [jerryji1993/DNABERT](https://github.com/jerryji1993/DNABERT) · [zhihan1996/DNA_bert_6](https://huggingface.co/zhihan1996/DNA_bert_6) | BERT, 12-layer | 6-mer k-mer |
| **Enformer** | EleutherAI/enformer-official-rough | Conv + Attention | Raw nucleotides |

DNABERT v1 is used (rather than DNABERT-2) for its compatibility with standard HuggingFace infrastructure — it loads via `AutoTokenizer`/`AutoModel` with no custom remote code, making it stable across all recent `transformers` versions.

---

## Project Structure

```
.
├── main.py              # Entry point — runs the full evaluation pipeline
├── dna_env.py           # Gymnasium MDP: synonymous-substitution action space
├── target_models.py     # GenomicClassifier wrapper (DNABERT + Enformer)
├── benchmark.py         # Accuracy/F1 performance + robustness analysis
└── README.md
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install torch transformers stable-baselines3 gymnasium \
            genomic-benchmarks scikit-learn numpy

# 3. Pre-download model weights
huggingface-cli download zhihan1996/DNA_bert_6
huggingface-cli download EleutherAI/enformer-official-rough
```

---

## Usage

```bash
# First run (downloads weights if not cached)
python main.py

# Subsequent runs (offline)
TRANSFORMERS_OFFLINE=1 python main.py
```

---

## How It Works

### K-mer Preprocessing (`target_models.py`)

DNABERT v1 requires sequences to be converted to space-separated 6-mers before tokenisation:
```
ACGTACGT  →  ACGTAC CGTACG GTACGT
```
This is handled automatically inside `GenomicClassifier.get_score()`.

### Adversarial Environment (`dna_env.py`)

`DNAAdversarialEnv` wraps any classifier with a `get_score(sequence: str) -> float` interface:

- **State:** One-hot encoded nucleotide sequence
- **Action:** Codon index to mutate (action space = sequence_length / 3)
- **Transition:** Replace selected codon with a synonymous alternative (full 18-amino-acid codon table)
- **Reward:** `1.0 - classifier_confidence`
- **Termination:** Confidence drops below 0.1 (success) or step budget exhausted

### Attack Loop (`main.py`)

A fresh PPO agent is trained for 2,000 timesteps per target sequence. Result (success, edit count, final probability) is logged for each sequence.

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | 50 | Sequences for performance benchmarking |
| `samples_to_attack` | 5 | Positive sequences targeted by RL agent |
| `total_timesteps` | 2000 | PPO steps per sequence |
| `n_steps` | 512 | PPO rollout buffer size |
| `threshold` | 0.1 | Success threshold (classifier confidence) |

---

## Extending to New Models

Any model with a `get_score(sequence: str) -> float` method can be plugged in:

```python
class MyClassifier:
    def get_score(self, sequence: str) -> float:
        ...  # return probability in [0, 1]

env = DNAAdversarialEnv(sequence, MyClassifier())
```

---

## Citation

```bibtex
@techreport{genomic_adversarial_2025,
  title  = {Adversarial Robustness of Genomic Foundation Models:
             A Synonymous-Substitution Attack Framework Using Reinforcement Learning},
  year   = {2025}
}
```

---

## References

- Ji et al. (2021). DNABERT. *Bioinformatics*, 37(15), 2112-2120.
- Avsec et al. (2021). Enformer. *Nature Methods.*
- Grevsova et al. (2023). Genomic Benchmarks. *BMC Genomic Data.*
- Schulman et al. (2017). PPO. *arXiv:1707.06347.*

---

## License

MIT
