"""
main.py — Adversarial robustness pipeline (v2).

Workflow:
  1. Fine-tune first:  python finetune.py --model both --epochs 5
  2. Then run attack:  python main.py

The attack uses calibrated model scores (post fine-tuning), a delta-reward
signal, and single-nucleotide substitutions — all of which produce meaningful
results on promoter sequences.
"""

import os
import json
import time
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanNontataPromoters

from target_models import GenomicClassifier
from dna_env import DNAAdversarialEnv
from benchmark import run_performance_test, analyze_robustness


def attack_sequence(seq, model, seq_idx, cfg):
    """Run a single PPO attack. Returns an attack log dict."""
    env = DNAAdversarialEnv(
        seq, model,
        threshold=cfg["threshold"],
        max_steps=cfg["max_steps"],
    )

    agent = PPO(
        "MlpPolicy", env,
        verbose=0,
        device="cpu",
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        ent_coef=cfg["ent_coef"],   # entropy bonus keeps exploration alive
        n_epochs=10,
    )
    agent.learn(total_timesteps=cfg["total_timesteps"])

    # Report the *best* score seen during the run, not just the final state
    final_score = env.best_score
    success = final_score < cfg["threshold"]

    print(f"    [!] #{seq_idx}: success={success}  "
          f"best_score={final_score:.4f}  edits={env.edits_made}  "
          f"steps={env.steps}")

    return {
        "seq_index": seq_idx,
        "success": success,
        "edits": env.edits_made,
        "best_score": round(final_score, 4),
        "final_score": round(env._current_score, 4),
        "steps": env.steps,
    }


def main(cfg):
    start = time.time()

    print("[1/4] Loading dataset...")
    dataset = HumanNontataPromoters(split="test", version=0)
    print(f"[+] {len(dataset)} sequences loaded.")

    print("\n[2/4] Initializing models...")
    models = [
        GenomicClassifier("zhihan1996/DNA_bert_6", "DNABERT"),
        GenomicClassifier("EleutherAI/enformer-official-rough", "Enformer"),
    ]

    # Warn if no checkpoints found — results will be meaningless
    for m in models:
        if m.classifier_head is None:
            print(f"\n  [WARN] {m.name} has no fine-tuned head. "
                  f"Run: python finetune.py --model "
                  f"{'dnabert' if 'enformer' not in m.name.lower() else 'enformer'} --epochs 5")

    all_results = {}

    label1_indices = [i for i in range(len(dataset)) if dataset[i][1] == 1]
    print(f"\n[3/4] Found {len(label1_indices)} positive sequences. "
          f"Attacking first {cfg['n_attack']}.")

    for model in models:
        print(f"\n{'='*55}")
        print(f"EVALUATING: {model.name}")
        print("="*55)

        # Performance
        print("[*] Performance benchmark...")
        perf = run_performance_test(model, dataset, limit=cfg["perf_limit"])
        print(f"[+] Accuracy={perf['accuracy']:.3f}  F1={perf['f1']:.3f}")

        # Attack
        print(f"[*] Attacking {cfg['n_attack']} sequences "
              f"(threshold={cfg['threshold']}, max_steps={cfg['max_steps']})...")
        attack_logs = []
        for i in label1_indices[: cfg["n_attack"]]:
            seq, _ = dataset[i]
            print(f"    -> Attacking #{i}  (len={len(seq)})")
            log = attack_sequence(seq, model, i, cfg)
            attack_logs.append(log)

        robustness = analyze_robustness(attack_logs)

        all_results[model.name] = {
            "performance": perf,
            "robustness": robustness,
            "attack_logs": attack_logs,
        }

        # Save after each model so a crash doesn't lose everything
        with open("results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"[+] Saved results.json")

    elapsed = (time.time() - start) / 60
    print(f"\n[+] Done in {elapsed:.1f} min. Results -> results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_attack",        type=int,   default=10,
                        help="Number of positive sequences to attack")
    parser.add_argument("--perf_limit",      type=int,   default=200,
                        help="Sequences for accuracy/F1 benchmark")
    parser.add_argument("--threshold",       type=float, default=0.3,
                        help="Score threshold for attack success")
    parser.add_argument("--max_steps",       type=int,   default=1000,
                        help="Max env steps per attack episode")
    parser.add_argument("--total_timesteps", type=int,   default=10000,
                        help="PPO training steps per sequence")
    parser.add_argument("--n_steps",         type=int,   default=256,
                        help="PPO rollout length")
    parser.add_argument("--batch_size",      type=int,   default=64)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--ent_coef",        type=float, default=0.01,
                        help="PPO entropy coefficient (exploration bonus)")
    args = parser.parse_args()
    main(vars(args))
