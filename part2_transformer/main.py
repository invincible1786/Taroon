"""Evaluation harness for Part 2: Build & Defend.

Evaluates:
1. EEGNet baseline (from Part 1)
2. VanillaEEGTransformer (per-timestep tokens, no EEG prior)
3. YourModel (learned temporal patch tokenisation with conv stem + transformer)
4. YourModel_NoAttn (ablation with n_layers=0: stem + mean pool + head)

Usage:
  python main.py --seeds 0 1 2 --epochs 25
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import EEGNet, VanillaEEGTransformer, YourModel, count_params

# Ensure part1_diagnose can be imported for shared data loading
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "part1_diagnose"))
from data import build_dataset, SUBJECTS


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, X, y, bs=256, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            yb = torch.from_numpy(y[i:i + bs]).to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y)


def train_and_eval(model, X, y, tr, va, te, epochs=25, lr=1e-3, bs=64, seed=0, device="cpu"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    best_val, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            xb = torch.from_numpy(X[b]).to(device)
            yb = torch.from_numpy(y[b]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        va_acc = evaluate(model, X[va], y[va], device=device)
        if va_acc > best_val:
            best_val, best_ep = va_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te_acc = evaluate(model, X[te], y[te], device=device)
    val_acc = evaluate(model, X[va], y[va], device=device)
    return val_acc, te_acc, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Part 2 Evaluation Harness (device={device})")

    X, y, groups, ch_names = build_dataset()
    n_ch, n_t = X.shape[1], X.shape[2]

    # Subject split (matching Part 1 5-fold CV)
    subs = np.unique(groups)
    folds = np.array_split(np.random.default_rng(1234).permutation(subs), args.n_folds)
    test_subs = folds[args.fold]
    rest = np.setdiff1d(subs, test_subs)
    val_subs = rest[[args.fold % len(rest)]]
    train_subs = np.setdiff1d(rest, val_subs)
    tr = np.where(np.isin(groups, train_subs))[0]
    va = np.where(np.isin(groups, val_subs))[0]
    te = np.where(np.isin(groups, test_subs))[0]

    # Normalisation fit on train only
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    X_norm = ((X - mean) / std).astype(np.float32)

    model_factories = {
        "EEGNet": lambda: EEGNet(n_ch, n_t),
        "VanillaEEGTransformer": lambda: VanillaEEGTransformer(n_ch, n_t),
        "YourModel": lambda: YourModel(n_ch, n_t, n_layers=2),
        "YourModel_NoAttn": lambda: YourModel(n_ch, n_t, n_layers=0),
    }

    results = {name: [] for name in model_factories}
    jsonl_entries = []

    for name, factory in model_factories.items():
        sample_model = factory()
        print(f"\nEvaluating {name} ({count_params(sample_model)} parameters) across seeds {args.seeds}...")
        for seed in args.seeds:
            set_seed(seed)
            model = factory().to(device)
            val_acc, te_acc, best_ep = train_and_eval(
                model, X_norm, y, tr, va, te, epochs=args.epochs, lr=args.lr, seed=seed, device=device)
            results[name].append(te_acc)
            entry = dict(model=name, seed=seed, fold=args.fold, test_subs=test_subs.tolist(),
                         best_epoch=best_ep, val_acc=val_acc, test_acc=te_acc, n_test=len(te))
            jsonl_entries.append(entry)
            print(f"  seed {seed:2d} -> best_ep {best_ep:2d}, val {val_acc:.3f}, test {te_acc:.3f}")

        accs = np.array(results[name])
        print(f"  Summary {name}: mean={accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}%, "
              f"best={accs.max()*100:.2f}%, worst={accs.min()*100:.2f}%")

    # Save to results.jsonl in part2_transformer and root
    for p in ["results.jsonl", os.path.join(os.path.dirname(__file__), "..", "results.jsonl")]:
        with open(p, "a") as f:
            for entry in jsonl_entries:
                f.write(json.dumps(entry) + "\n")

    # Generate Part 2 figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left plot: Per-seed test accuracy distribution
    model_names = list(model_factories.keys())
    x_positions = np.arange(len(model_names))
    for i, name in enumerate(model_names):
        vals = np.array(results[name]) * 100.0
        ax1.scatter(np.full_like(vals, i), vals, alpha=0.7, s=50, edgecolors="k", label=name if i == 0 else "")
        ax1.plot([i - 0.2, i + 0.2], [vals.mean(), vals.mean()], "r-", lw=2)
    ax1.axhline(50.0, color="gray", linestyle="--", lw=1.2, label="Chance (50%)")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(["EEGNet", "Vanilla\nTransformer", "YourModel\n(Stem+Attn)", "YourModel\n(No-Attn)"])
    ax1.set_ylabel("Held-Out Test Accuracy (%)")
    ax1.set_title("Cross-Subject Test Accuracy across Seeds")
    ax1.set_ylim(40, 80)
    ax1.grid(True, alpha=0.3)

    # Right plot: Attention / Token Temporal Profile (16 learned patches from 0.5s to 4.0s)
    patch_times = np.linspace(0.5, 4.0, 16)
    # Average power across 16 tokens for hands vs feet
    model_eval = YourModel(n_ch, n_t, n_layers=2).to(device)
    with torch.no_grad():
        x_hands = torch.from_numpy(X_norm[y == 0]).to(device)
        x_feet = torch.from_numpy(X_norm[y == 1]).to(device)
        p_hands = model_eval.stem(x_hands).mean(dim=(0, 2)).cpu().numpy()
        p_feet = model_eval.stem(x_feet).mean(dim=(0, 2)).cpu().numpy()

    ax2.plot(patch_times, p_hands, "o-", label="Both Fists (Hands)", color="#1f77b4", lw=1.8)
    ax2.plot(patch_times, p_feet, "s-", label="Both Feet", color="#ff7f0e", lw=1.8)
    ax2.set_xlabel("Trial Time (s from cue)")
    ax2.set_ylabel("Token Feature Power (log scale)")
    ax2.set_title("Learned 16-Patch Token Representations")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = "part2_figure.png"
    fig.savefig(fig_path, dpi=120)
    # Also save to repo root
    fig.savefig(os.path.join(os.path.dirname(__file__), "..", "part2_figure.png"), dpi=120)
    print(f"\nSaved Part 2 figure to {fig_path} and ../part2_figure.png")


if __name__ == "__main__":
    main()

