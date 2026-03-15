"""
target_models.py

Wraps DNABERT and Enformer for use in the adversarial pipeline.
If a fine-tuned checkpoint exists in checkpoints/, the classification head
is loaded automatically and get_score() returns calibrated probabilities.
Without a checkpoint, the model falls back to the raw pre-trained score
(useful for sanity-checking but not suitable for attack evaluation).
"""

import os
import glob
import torch
import torch.nn as nn
import numpy as np

# Patch BEFORE importing anything from transformers.
import transformers.utils.import_utils as _import_utils
_import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils as _modeling_utils
_modeling_utils.check_torch_load_is_safe = lambda: None

from transformers import AutoTokenizer, AutoModel, AutoConfig

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


# ── Encoding helpers ───────────────────────────────────────────────────────────

_BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def seq_to_onehot(seq):
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        if b in _BASE_IDX:
            arr[i, _BASE_IDX[b]] = 1.0
    return arr


def seq_to_kmer(seq, k=6):
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


# ── Classification heads (must match finetune.py definitions) ─────────────────

def _make_dnabert_head(hidden_size):
    return nn.Sequential(
        nn.Linear(hidden_size, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    )


def _make_enformer_head(n_tracks=5313):
    return nn.Sequential(
        nn.Linear(n_tracks, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    )


# ── Main wrapper ───────────────────────────────────────────────────────────────

class GenomicClassifier:
    def __init__(self, model_path, name):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_head = None

        print(f"[*] Initializing {self.name}...")
        print(f"    - Loading weights from {model_path}...")

        local_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"

        try:
            if "enformer" in name.lower():
                self._load_enformer(model_path, local_only)
            else:
                self._load_bert(model_path, local_only)
        except OSError as e:
            raise OSError(f"Failed to load '{name}' from '{model_path}'.\nOriginal error: {e}")

        self.model.eval()
        self._try_load_head()
        status = "fine-tuned" if self.classifier_head is not None else "pre-trained (no checkpoint)"
        print(f"[+] {self.name} loaded on {self.device} [{status}].")

    def _load_bert(self, model_path, local_only):
        config = AutoConfig.from_pretrained(model_path, local_files_only=local_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_only)
        if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id or 0
        self.model = AutoModel.from_pretrained(
            model_path, config=config, local_files_only=local_only,
        ).to(self.device)

    def _load_enformer(self, model_path, local_only):
        try:
            from enformer_pytorch import Enformer, EnformerConfig
        except ImportError:
            raise ImportError("pip install enformer-pytorch")

        bin_path = self._find_cached_file(model_path, "pytorch_model.bin", local_only)
        if bin_path is None:
            raise OSError(f"pytorch_model.bin not found for {model_path}.")

        print(f"    - Cache: {bin_path}")

        from transformers.modeling_utils import PreTrainedModel
        from transformers import PretrainedConfig

        enformer_cfg = EnformerConfig()
        pt_cfg = PretrainedConfig()
        for k, v in vars(enformer_cfg).items():
            setattr(pt_cfg, k, v)

        _orig = PreTrainedModel.__init__
        PreTrainedModel.__init__ = lambda s, c=None: torch.nn.Module.__init__(s)
        try:
            self.model = Enformer.__new__(Enformer)
            Enformer.__init__(self.model, pt_cfg)
        finally:
            PreTrainedModel.__init__ = _orig

        sd = torch.load(bin_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd, strict=True)
        self.model = self.model.to(self.device)
        self.tokenizer = None

    @staticmethod
    def _find_cached_file(model_path, filename, local_only):
        repo_id = model_path.replace("/", "--")
        roots = [
            f"/root/.cache/huggingface/hub/models--{repo_id}",
            os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo_id}"),
        ]
        for root in roots:
            hits = glob.glob(os.path.join(root, "**", filename), recursive=True)
            if hits:
                return hits[0]
        if not local_only:
            try:
                from huggingface_hub import hf_hub_download
                return hf_hub_download(repo_id=model_path, filename=filename)
            except Exception:
                pass
        return None

    def _try_load_head(self):
        key = "enformer" if "enformer" in self.name.lower() else "dnabert"
        ckpt = os.path.join(CHECKPOINT_DIR, f"{key}_classifier.pt")
        if not os.path.exists(ckpt):
            print(f"    [!] No checkpoint at {ckpt} — using raw pre-trained scores.")
            print(f"        Run: python finetune.py --model {key}")
            return

        head = _make_enformer_head() if key == "enformer" else _make_dnabert_head(self.model.config.hidden_size)
        full_sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        head_sd = {k.replace("classifier.", ""): v for k, v in full_sd.items() if k.startswith("classifier.")}
        head.load_state_dict(head_sd)
        self.classifier_head = head.to(self.device).eval()
        print(f"    [+] Loaded fine-tuned head from {ckpt}")

    def get_score(self, sequence):
        if "enformer" in self.name.lower():
            return self._score_enformer(sequence)
        return self._score_bert(sequence)

    def _score_bert(self, sequence):
        inputs = self.tokenizer(
            seq_to_kmer(sequence, k=6),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            cls = out.last_hidden_state[:, 0, :]
            if self.classifier_head is not None:
                return torch.sigmoid(self.classifier_head(cls)).item()
            return torch.sigmoid(cls.mean()).item()

    def _score_enformer(self, sequence):
        target_len = 196608
        seq = sequence.upper()
        if len(seq) >= target_len:
            start = (len(seq) - target_len) // 2
            seq = seq[start: start + target_len]
        else:
            seq = seq.center(target_len, 'N')
        onehot = torch.tensor(seq_to_onehot(seq)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(onehot)
            pooled = out['human'].mean(dim=1)
            if self.classifier_head is not None:
                return torch.sigmoid(self.classifier_head(pooled)).item()
            return torch.sigmoid(pooled.mean()).item()
