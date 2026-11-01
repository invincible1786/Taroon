"""Train and evaluate a hands-vs-feet decoder on unseen subjects.

Usage:
  python train.py --channels all --seed 0
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn

from data import build_dataset, SUBJECTS
from model import EEGNet, count_params

FRONTAL = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"]  # closest to the eyes
CENTRAL = ["FC3", "FC1", "FCz", "FC2", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CP1", "CPz", "CP2", "CP4"]
OCCIPITAL = ["O1", "Oz", "O2", "PO3", "POz", "PO4", "PO7", "PO8"]  # same size as FRONTAL

CHANNEL_GROUPS = {
    "all": None,
    "frontal": FRONTAL,
    "central": CENTRAL,
    "occipital": OCCIPITAL,
    "no_frontal": "NOT_FRONTAL",
}


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, X, y, bs=256, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            yb = torch.from_numpy(y[i:i + bs]).to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=list(CHANNEL_GROUPS), default="all")
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--select", choices=["val", "last"], default="val")
    ap.add_argument("--permute", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    X, y, groups, ch_names = build_dataset(args.subjects)
    picks = CHANNEL_GROUPS[args.channels]
    if picks == "NOT_FRONTAL":
        picks = [c for c in ch_names if c not in FRONTAL]
    if picks is not None:
        if not picks:
            raise ValueError(f"Fill in CHANNEL_GROUPS['{args.channels}'] first.")
        missing = [c for c in picks if c not in ch_names]
        assert not missing, f"Missing channels: {missing}"
        X = X[:, [ch_names.index(c) for c in picks]]
    print(f"data: {X.shape}, subjects={len(np.unique(groups))}, "
          f"class balance={np.bincount(y)}, device={device}")

    # Subject-level split: 5-fold cross-validation by subject.
    subs = np.unique(groups)
    folds = np.array_split(np.random.default_rng(1234).permutation(subs), args.n_folds)
    test_subs = folds[args.fold]
    rest = np.setdiff1d(subs, test_subs)
    val_subs = rest[[args.fold % len(rest)]]
    train_subs = np.setdiff1d(rest, val_subs)
    tr = np.where(np.isin(groups, train_subs))[0]
    va = np.where(np.isin(groups, val_subs))[0]
    te = np.where(np.isin(groups, test_subs))[0]
    assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[va]) & set(groups[te]))

    # Per-channel z-scoring: fit normalisation on training subjects only.
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    # Permutation test: shuffle train labels only; val and test keep true labels.
    y_fit = y.copy()
    if args.permute:
        y_fit[tr] = np.random.default_rng(args.seed).permutation(y[tr])

    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_ch, n_t).to(device)
    print(f"EEGNet: {count_params(model)} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    best_val, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]
            xb = torch.from_numpy(X[b]).to(device)
            yb = torch.from_numpy(y_fit[b]).to(device)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        va_acc = evaluate(model, X[va], y[va], device=device)
        te_acc = evaluate(model, X[te], y[te], device=device)
        if args.select == "val" and va_acc > best_val:
            best_val, best_ep = va_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val {va_acc:.3f}  test {te_acc:.3f}")

    if args.select == "val" and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_ep = args.epochs - 1

    final_test_acc = evaluate(model, X[te], y[te], device=device)
    final_val_acc = evaluate(model, X[va], y[va], device=device)
    result = dict(channels=args.channels, seed=args.seed, fold=args.fold,
                  test_subs=test_subs.tolist(), best_epoch=best_ep,
                  val_acc=final_val_acc, test_acc=final_test_acc, n_test=len(te),
                  permute=args.permute)
    print(json.dumps(result))
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
    with open(results_path, "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
