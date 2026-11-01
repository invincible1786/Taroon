"""Aggregate results across folds from results.jsonl.

Computes pooled test accuracy, 95% Clopper-Pearson confidence interval,
and exact binomial p-value vs chance (0.5).
"""
import json
import os
import sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest


def clopper_pearson(k, n, confidence=0.95):
    res = binomtest(k, n, p=0.5)
    ci = res.proportion_ci(confidence_level=confidence, method="exact")
    return ci.low, ci.high, res.pvalue


def main(filepath="results.jsonl"):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    runs = defaultdict(list)
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (entry.get("channels", "all"), entry.get("seed", 0), entry.get("permute", False))
            runs[key].append(entry)

    print(f"{'Channels':<15} {'Seed':<5} {'Permuted':<9} {'Folds':<6} {'Correct/N':<12} {'Accuracy':<10} {'95% CI':<18} {'p-value':<10}")
    print("-" * 88)

    for (channels, seed, permute), entries in runs.items():
        total_correct = 0
        total_n = 0
        for e in entries:
            acc = e["test_acc"]
            n = e["n_test"]
            total_correct += int(round(acc * n))
            total_n += n

        pooled_acc = total_correct / total_n if total_n > 0 else 0
        low, high, pval = clopper_pearson(total_correct, total_n)
        ci_str = f"[{low*100:.1f}%, {high*100:.1f}%]"
        p_str = f"{pval:.4f}" if pval >= 0.0001 else f"{pval:.2e}"
        print(f"{channels:<15} {seed:<5} {str(permute):<9} {len(entries):<6} {f'{total_correct}/{total_n}':<12} {f'{pooled_acc*100:.2f}%':<10} {ci_str:<18} {p_str:<10}")


if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"
    main(fp)
