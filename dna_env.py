"""
dna_env.py — Improved adversarial environment for genomic sequences.

Key improvements over v1:
1. Single-nucleotide substitution action space (not codon-level) — works on
   any sequence type including promoters, not just CDS.
2. Delta-reward: reward is the *change* in score per step, not the raw score.
   This gives the agent a useful gradient signal even when scores are near 0.5.
3. Score is cached between steps — only recomputed when sequence actually changed.
4. Step limit enforced via truncation signal (not just timestep budget in PPO).
5. Best-sequence tracking: env keeps track of the lowest score seen so far
   so benchmark.py can report the best achieved reduction even on failures.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

BASES = ['A', 'C', 'G', 'T']
BASE_IDX = {b: i for i, b in enumerate(BASES)}

# Full synonymous codon map — retained for optional codon-mode use
SYNON_MAP = {
    'GCT': ['GCC', 'GCA', 'GCG'], 'GCC': ['GCT', 'GCA', 'GCG'],
    'GCA': ['GCT', 'GCC', 'GCG'], 'GCG': ['GCT', 'GCC', 'GCA'],
    'TGT': ['TGC'], 'TGC': ['TGT'],
    'GAT': ['GAC'], 'GAC': ['GAT'],
    'GAA': ['GAG'], 'GAG': ['GAA'],
    'TTT': ['TTC'], 'TTC': ['TTT'],
    'GGT': ['GGC', 'GGA', 'GGG'], 'GGC': ['GGT', 'GGA', 'GGG'],
    'GGA': ['GGT', 'GGC', 'GGG'], 'GGG': ['GGT', 'GGC', 'GGA'],
    'CAT': ['CAC'], 'CAC': ['CAT'],
    'ATT': ['ATC', 'ATA'], 'ATC': ['ATT', 'ATA'], 'ATA': ['ATT', 'ATC'],
    'AAA': ['AAG'], 'AAG': ['AAA'],
    'TTA': ['TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 'TTG': ['TTA', 'CTT', 'CTC', 'CTA', 'CTG'],
    'CTT': ['TTA', 'TTG', 'CTC', 'CTA', 'CTG'], 'CTC': ['TTA', 'TTG', 'CTT', 'CTA', 'CTG'],
    'CTA': ['TTA', 'TTG', 'CTT', 'CTC', 'CTG'], 'CTG': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA'],
    'AAT': ['AAC'], 'AAC': ['AAT'],
    'CCT': ['CCC', 'CCA', 'CCG'], 'CCC': ['CCT', 'CCA', 'CCG'],
    'CCA': ['CCT', 'CCC', 'CCG'], 'CCG': ['CCT', 'CCC', 'CCA'],
    'CAA': ['CAG'], 'CAG': ['CAA'],
    'CGT': ['CGC', 'CGA', 'CGG', 'AGA', 'AGG'], 'CGC': ['CGT', 'CGA', 'CGG', 'AGA', 'AGG'],
    'CGA': ['CGT', 'CGC', 'CGG', 'AGA', 'AGG'], 'CGG': ['CGT', 'CGC', 'CGA', 'AGA', 'AGG'],
    'AGA': ['CGT', 'CGC', 'CGA', 'CGG', 'AGG'], 'AGG': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA'],
    'TCT': ['TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 'TCC': ['TCT', 'TCA', 'TCG', 'AGT', 'AGC'],
    'TCA': ['TCT', 'TCC', 'TCG', 'AGT', 'AGC'], 'TCG': ['TCT', 'TCC', 'TCA', 'AGT', 'AGC'],
    'AGT': ['TCT', 'TCC', 'TCA', 'TCG', 'AGC'], 'AGC': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT'],
    'ACT': ['ACC', 'ACA', 'ACG'], 'ACC': ['ACT', 'ACA', 'ACG'],
    'ACA': ['ACT', 'ACC', 'ACG'], 'ACG': ['ACT', 'ACC', 'ACA'],
    'GTT': ['GTC', 'GTA', 'GTG'], 'GTC': ['GTT', 'GTA', 'GTG'],
    'GTA': ['GTT', 'GTC', 'GTG'], 'GTG': ['GTT', 'GTC', 'GTA'],
    'TAT': ['TAC'], 'TAC': ['TAT'],
}


class DNAAdversarialEnv(gym.Env):
    """
    Single-nucleotide substitution MDP.

    Action space: MultiDiscrete([L, 4]) — choose a position and a new base.
    Observation:  Box(L,) int32 — base index per position.
    Reward:       delta = prev_score - new_score (positive when score drops).
    Terminated:   score drops below `threshold`.
    Truncated:    after `max_steps` steps without success.
    """

    def __init__(self, original_sequence, classifier, threshold=0.5, max_steps=500):
        super().__init__()
        self.classifier = classifier
        self.threshold = threshold
        self.max_steps = max_steps

        seq = original_sequence.upper()
        # Replace unknown bases with random valid base
        seq = ''.join(b if b in BASE_IDX else np.random.choice(BASES) for b in seq)
        self.original_seq = list(seq)
        self.current_seq = list(seq)
        self.edits_made = 0
        self.steps = 0

        L = len(self.original_seq)
        # Action: [position (0..L-1), new_base (0..3)]
        self.action_space = spaces.MultiDiscrete([L, 4])
        self.observation_space = spaces.Box(low=0, high=3, shape=(L,), dtype=np.int32)

        # Cache initial score so first step can compute a delta
        self._current_score = self.classifier.get_score("".join(self.current_seq))
        self.best_score = self._current_score
        self.best_seq = list(self.current_seq)

    def step(self, action):
        pos, new_base_idx = int(action[0]), int(action[1])
        new_base = BASES[new_base_idx]
        prev_score = self._current_score

        if self.current_seq[pos] != new_base:
            self.current_seq[pos] = new_base
            self.edits_made += 1
            self._current_score = self.classifier.get_score("".join(self.current_seq))

            if self._current_score < self.best_score:
                self.best_score = self._current_score
                self.best_seq = list(self.current_seq)

        # Delta reward: how much did score drop this step?
        reward = prev_score - self._current_score

        self.steps += 1
        terminated = self._current_score < self.threshold
        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "score": self._current_score,
            "edits": self.edits_made,
            "best_score": self.best_score,
        }

    def _get_obs(self):
        return np.array([BASE_IDX.get(b, 0) for b in self.current_seq], dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_seq = list(self.original_seq)
        self.edits_made = 0
        self.steps = 0
        self._current_score = self.classifier.get_score("".join(self.current_seq))
        self.best_score = self._current_score
        self.best_seq = list(self.current_seq)
        return self._get_obs(), {}


class DNACodonEnv(DNAAdversarialEnv):
    """
    Synonymous codon substitution MDP — use this for CDS sequences.
    Inherits delta-reward and best-sequence tracking from DNAAdversarialEnv.
    Action: codon index (0 .. L//3 - 1); substitution is drawn from SYNON_MAP.
    """

    def __init__(self, original_sequence, classifier, threshold=0.5, max_steps=500):
        # Truncate to codon boundary before calling super()
        seq = original_sequence.upper()
        remainder = len(seq) % 3
        if remainder:
            seq = seq[: len(seq) - remainder]
        super().__init__(seq, classifier, threshold, max_steps)

        n_codons = len(self.original_seq) // 3
        self.action_space = spaces.Discrete(max(n_codons, 1))

    def step(self, action):
        idx = int(action) * 3
        codon = "".join(self.current_seq[idx: idx + 3])
        prev_score = self._current_score

        if codon in SYNON_MAP:
            new_codon = np.random.choice(SYNON_MAP[codon])
            if new_codon != codon:
                self.current_seq[idx: idx + 3] = list(new_codon)
                self.edits_made += 1
                self._current_score = self.classifier.get_score("".join(self.current_seq))
                if self._current_score < self.best_score:
                    self.best_score = self._current_score
                    self.best_seq = list(self.current_seq)

        reward = prev_score - self._current_score
        self.steps += 1
        terminated = self._current_score < self.threshold
        truncated = self.steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {
            "score": self._current_score,
            "edits": self.edits_made,
            "best_score": self.best_score,
        }
