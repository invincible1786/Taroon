# Part 2: Build & Defend — Transformer Decoder with Learned Temporal-Patch Prior

## Overview
This package explores self-attention for bilateral motor imagery EEG decoding (both fists vs both feet) on PhysioNet EEGBCI.

### Models
1. **EEGNet**: Validated baseline from Part 1 with convolutional temporal and depthwise spatial priors.
2. **VanillaEEGTransformer**: Per-timestep tokens without inductive priors (sequence length $T=561$, attention surface $561^2 = 314,721$).
3. **YourModel**: Injects an inductive bias via a learned temporal-patch conv stem (`PatchStem`). Shinks the sequence length to 16 tokens ($16^2 = 256$ attention matrix, $>1200\times$ smaller attention surface).
4. **YourModel_NoAttn**: Ablation control (`n_layers=0`) isolating the contribution of the convolutional stem from self-attention.

### Usage
```bash
pip install -r requirements.txt
python main.py --seeds 0 1 2 3 4
```
Outputs per-seed held-out accuracy, saves summary to `results.jsonl`, and produces `part2_figure.png`.

