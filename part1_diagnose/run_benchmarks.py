"""Run all 5 folds for all key channel configurations and permutation test."""
import subprocess
import sys
import os

configs = [
    ("all", False),
    ("frontal", False),
    ("central", False),
    ("no_frontal", False),
    ("all", True),
]

# Clean or initialize results.jsonl
results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
if os.path.exists(results_path):
    os.remove(results_path)

print("Starting benchmark runs on GPU...")
for ch, perm in configs:
    for fold in range(5):
        cmd = [sys.executable, "train.py", "--channels", ch, "--fold", str(fold), "--seed", "0"]
        if perm:
            cmd.append("--permute")
        print(f"Running channels={ch}, fold={fold}, permute={perm}...")
        subprocess.run(cmd, check=True)

print("\n--- ALL BENCHMARKS COMPLETE ---")
subprocess.run([sys.executable, "aggregate.py"])

