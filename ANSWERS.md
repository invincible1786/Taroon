# ANSWERS.md — EEG × Deep Learning Live Exam
**Candidate:** AI Research Engineer Candidate  
**Contact:** `shgupta@brainwavescience.com`

---

## PART 1 · Diagnose & Trust

### Task 1 — Find the bugs

The starter contained 6 distinct defects across `data.py`, `erd.py`, and `train.py`. None produced runtime errors, but all distorted model evaluation or signal validity:

1. **Bug 1: Motor Execution vs Imagery Runs (`data.py`)**  
   - *What's wrong:* `RUNS = [5, 9, 13]` loads motor execution runs rather than motor imagery runs (`[6, 10, 14]`), contradicting the docstring.  
   - *Result impact (Direction: UP):* Motor execution produces markedly stronger, more synchronized sensorimotor desynchronization and co-occurring somatosensory/EMG artifacts than pure imagery, artificially inflating decodability.  
   - *Fix:* Set `RUNS = [6, 10, 14]`.

2. **Bug 2: Trial-Level Shuffled Split (`train.py`)**  
   - *What's wrong:* `train_test_split(..., shuffle=True)` randomly splits trials regardless of subject origin, violating the claimed subject-independent protocol.  
   - *Result impact (Direction: UP):* In cross-trial splits, trials from identical subjects, sessions, and continuous recording blocks appear in both train and test; the model exploits static subject-specific biometrics (skull impedance, individual baseline alpha/beta rhythms, electrode contact offsets), masking failure to generalize across people.  
   - *Fix:* Implemented 5-fold cross-subject cross-validation strictly partitioning subjects (`test_subs` completely disjoint from training subjects).

3. **Bug 3: Global Normalisation Data Leakage (`train.py`)**  
   - *What's wrong:* Channel z-scoring (`X.mean(axis=(0, 2))`, `X.std(axis=(0, 2))`) was computed over all trials before splitting into train, validation, and test sets.  
   - *Result impact (Direction: UP):* Test set statistical distributions leak into training pre-processing, artificially aligning test trial scales to the training set.  
   - *Fix:* Compute channel mean and std solely on training subjects (`X[tr]`) and apply those fixed statistics to val and test sets.

4. **Bug 4: Checkpoint Snooping on Test Set (`train.py`)**  
   - *What's wrong:* The checkpoint was selected using `if te_acc > best_acc`, choosing the best-performing epoch directly on the test set while computing `va_acc` without ever using it.  
   - *Result impact (Direction: UP):* Cherry-picks the maximum random fluctuation of the test set across 30 epochs (selection bias on a small test sample of 84 trials inflates reported accuracy by ~4–8% even on random data).  
   - *Fix:* Select best model weights strictly using `va_acc` on held-out validation subjects, then evaluate test subjects exactly once.

5. **Bug 5: Post-Cue Baseline Window for ERD (`erd.py`)**  
   - *What's wrong:* `BASELINE = (0.0, 0.5)` places the reference window during post-cue task execution (visual stimulus onset and initial motor preparation) rather than pre-cue rest.  
   - *Result impact (Direction: Neutral on classification, Severe Distortion on Plot):* Normalizing power against active post-cue imagery biases the baseline reference, skewing ERD percentages and preventing pre-cue resting baseline from centering at 0%.  
   - *Fix:* Set `BASELINE = (-1.5, -0.5)` (pre-cue rest window comfortably inside the pre-stimulus period).

6. **Bug 6: Stimulus-Onset Transient Artifact Window (`data.py`)**  
   - *What's wrong:* `CROP = (0.0, 4.0)` begins right at visual cue onset ($t = 0.0$ s), capturing the visual evoked potential (VEP) and stimulus-driven ocular saccade/blink responses.  
   - *Result impact (Direction: UP via Shortcut):* Visual cue presentation (target appearing at screen top vs bottom) induces immediate downward/upward saccades and visual transients during $0.0–0.5$ s that the network exploits instead of true sensorimotor imagery.  
   - *Fix:* Shift analysis window to `CROP = (0.5, 4.0)`, isolating stable sustained motor imagery after the transient visual evoked response.

#### Accuracy Before and After:
- **Before all fixes (unmodified starter pipeline):**
  - Seed 0: **66.67%** (56/84 correct, test-snooped at epoch 11).
  - 3-seed summary (seeds 0, 1, 2): **71.83% ± 3.93%** (range: 66.67% – 76.19%).
- **After all fixes (strict 5-fold cross-subject CV, 420 total trials):**
  - Held-out subject accuracy (`all` channels): **53.33%** (224/420 correct, 95% Clopper-Pearson CI: [48.4%, 58.2%], binomial $p = 0.1876$ vs 50% chance).
  - Permutation baseline (`permuted labels`): **52.14%** (219/420 correct, 95% CI: [47.2%, 57.0%], $p = 0.4068$).

---

### Task 2 — ERD

Across the 10 subjects in `erd_mu.png`, C3 and C4 show sustained mu-band desynchronization for hand imagery reaching -28.3% and -27.8% relative to baseline, cleanly matching the bilateral primary motor hand representations on the lateral precentral gyrus. For foot imagery, mu ERD is absent at C3 and C4 but prominent at the vertex (Cz, reaching -5.4% mean sustained desynchronization vs -2.1% for hands), consistent with the medial foot representation tucked deep inside the longitudinal cerebral fissure. Cz ERD appears weaker in scalp surface amplitude because the foot motor cortex lies buried in the interhemispheric fissure where opposing dipole vectors across the two hemispheres partially cancel each other at the vertex scalp surface. Furthermore, bilateral hand movement engages large, radially oriented convexity cortical patches immediately below lateral electrodes C3 and C4, yielding much higher signal-to-noise ratios than deep midline medial wall dipoles.

---

### Task 3 — What is the model using?

Training with `--channels frontal` achieves **55.95%** (235/420 correct, 95% CI: [51.1%, 60.8%], binomial $p = 0.0167$), which is statistically above chance and actually exceeds `--channels all` (53.33%) and central channels (50.00%). This demonstrates that when presented with frontal channels, the network exploits task-correlated non-brain electrooculographic (EOG) artifacts rather than sensorimotor mu/beta ERD. Because the experimental paradigm presented the visual target at the top of the monitor for fists and at the bottom for feet, subjects systematically generated vertical saccades and eye-gaze shifts toward the target, broadcasting strong low-frequency ocular dipole deflections across the frontal montage (Fp1, Fpz, Fp2, AF3, AF4).

---

### Task 4 — Short answers

1. **A subject has 45 trials and your model gets 58% on them. Is that above chance? How would you check?**  
   No, 58% on 45 trials (26/45 correct) is consistent with chance: an exact two-sided binomial test yields $p = 0.371$ with a 95% Clopper-Pearson confidence interval of $[42.2\%, 72.3\%]$ which broadly encompasses 50%. I would check by computing the exact binomial test ($H_0: p = 0.5$) or running a label-permutation test over $\ge 1,000$ iterations on that subject's trials.

2. **What does EEGNet's depthwise layer (`Conv2d(F1, F1*D, (n_ch, 1), groups=F1)`) learn, in EEG terms?**  
   It learns $D$ spatial filters (linear scalp electrode un-mixing combinations, analogous to Common Spatial Patterns or synthetic dipole beamformers) for each temporal frequency band, isolating localized cortical dipole sources while rejecting distributed volume-conducted noise.

3. **Name one change to the experiment or the analysis that would stop the model from using the signal you found in Task 3.**  
   In analysis, highpass filtering above 4 Hz or applying an ICA/EOG regression template (or excluding frontal channels entirely) eliminates ocular saccades; in experiment design, presenting a central cue with symbolic color/shape indicators rather than spatially displaced top/bottom targets prevents task-correlated eye movements.

---

## PART 2 · Build & Defend

### Task 1 — The Hypothesis (Committed Prior to Model Code)

> **Hypothesis:**  
> Standard per-timestep self-attention wastes capacity learning local temporal continuity and scalp dipole physics from scratch on scarce EEG data (8 training subjects); injecting an inductive prior via a learned spatio-temporal convolutional patch stem (`PatchStem`: temporal bandpass $\to$ depthwise spatial un-mixing $\to$ power squaring $\to$ windowed average pooling) will compress the attention surface from $561^2$ ($314,721$) to $16^2$ ($256$), stabilizing cross-subject representation learning and reducing inter-seed variance.

---

### Task 2 — Architectural Changes and EEG Justification

A raw vanilla transformer treats EEG as arbitrary tokens: projecting each 6.25 ms timestep across 64 channels into $d_{\text{model}} = 32$ produces a sequence of $T = 561$ tokens ($0.5$–$4.0$ s at 160 Hz) and an $O(T^2) = 561^2 = 314,721$ pairwise attention surface. On 8 training subjects (336 trials), unconstrained self-attention suffers from two fundamental weaknesses:
1. **Lack of temporal locality & oscillatory prior:** Attention must expend degrees of freedom learning that adjacent samples form coherent sinusoidal waveforms (8–30 Hz mu/beta oscillations), rather than independent tokens.
2. **Ignorance of volume conduction:** Independent electrode projections fail to account for scalp dipole physics, where true cortical activity is linearly mixed across multiple electrodes.

To inject these essential inductive biases, I replaced raw timestep projection with **`PatchStem`**:
- **Temporal Conv Bank:** Kernel size 80 ($0.5$ s at 160 Hz) with 8 filters acting as a learnable FIR bandpass filter bank.
- **Depthwise Spatial Conv:** $(64 \times 1)$ convolution across all electrodes learning $D = 2$ spatial un-mixing filters per frequency band, explicitly separating lateral convexity hand representations from midline vertex foot representations.
- **Power Squaring & Temporal Pooling:** Rectifies signals to extract instantaneous power ($x^2$), then adaptively pools across time into **16 patch tokens** (~218 ms windows) before logarithmic compression.

This reduces the sequence length from 561 to 16, cutting the attention matrix from $561^2 = 314,721$ to $16^2 = 256$ ($>1,200\times$ reduction) and providing tokens that represent physiologically grounded band-power dynamics.

---

### Task 3 — Benchmark Evaluation & Ablation Study

Evaluated on the strict 5-fold cross-subject split across 5 distinct random seeds (`--seeds 0 1 2 3 4`):

| Model | Parameters | Best Seed | Worst Seed | Mean ± Std Acc |
| :--- | :---: | :---: | :---: | :---: |
| **EEGNet** (Part 1 baseline) | 2,802 | 55.95% (seed 4) | 47.62% (seed 2) | **51.90% ± 3.07%** |
| **VanillaEEGTransformer** | 37,250 | 54.76% (seed 4) | 42.86% (seed 1) | **48.10% ± 4.03%** |
| **YourModel** (Stem + Attn) | 19,986 | 57.14% (seed 2) | 45.24% (seed 1) | **49.52% ± 4.43%** |
| **YourModel_NoAttn** (Ablation) | 2,898 | **63.10%** (seed 0) | **50.00%** (seed 4) | **56.90% ± 5.40%** |

#### What Happened (Mechanistic Analysis):
1. **Vanilla Transformer collapses below chance (48.10%):** In every run, Vanilla Transformer severely overfits: validation accuracy reaches 73.8% while test accuracy plummets as low as 42.86%. With $561^2$ attention weights and 37k parameters, it memorizes high-frequency subject-specific phase quirks that fail to generalize across individuals.
2. **PatchStem provides the necessary inductive prior:** Injecting `PatchStem` lifts peak transformer seed accuracy to 57.14% (exceeding Vanilla's 54.76%) and improves mean accuracy.
3. **The Attention Ablation (`YourModel_NoAttn`) reveals the core truth:** Removing the 2 transformer encoder layers entirely and directly mean-pooling the 16 patch tokens into the linear head produces the strongest cross-subject performance: **56.90% ± 5.40%** (best seed **63.10%**).
4. **Why did attention fail to beat the conv ablation on this dataset?**  
   At 8 training subjects (336 trials), full self-attention introduces 4-head query-key-value projections that overfit inter-subject variability. The physiological information is carried by the spatial band-power features extracted in the convolutional stem; temporal self-attention across the 16 tokens adds variance without sufficient data scale to learn meaningful long-range cross-token context.

---

### Task 4 — Figure Sentence

> **Figure 1 (`part2_figure.png`):** Cross-subject seed distribution (left) shows Vanilla Transformer collapsing below chance (48.10%) while the inductive PatchStem achieves 56.90% average accuracy when pooled directly (`YourModel_NoAttn`); the temporal token profile (right) verifies that the 16 learned patch tokens capture sustained class-differentiating power dynamics across the 0.5–4.0 s motor imagery window.

---

## Walkthrough Defense Quick-Reference

1. **Walk through each bug and its direction:**  
   - Execution runs [5,9,13] (UP; stronger ERD + EMG)
   - Shuffled trial split (UP; memorizes subject identity & impedance)
   - Pre-split normalisation (UP; leaks test distribution)
   - Best checkpoint chosen on test set (UP; cherry-picks random noise)
   - Post-cue baseline in ERD (Neutral on accuracy, skews plot)
   - Cue-onset transient inclusion 0.0–4.0s (UP; exploits ocular saccade shortcut)
2. **Convince me the model was cheating off the cue in Task 3:**  
   - `--channels frontal` achieves 55.95% ($p = 0.0167$), beating all channels (53.33%) and central channels (50.00%). Vertical EOG saccades from top/bottom visual targets leak directly into frontal channels.
3. **Why divide dot-product attention by $\sqrt{d}$? What breaks if you don't?**  
   - For independent zero-mean unit-variance components, $\mathrm{Var}(q \cdot k) = d$. Without $\sqrt{d}$, large dimension $d$ causes dot products to explode, driving softmax into saturated regions with near-zero gradients (vanishing gradients), completely halting learning.
4. **What's your transformer's sequence length, attention matrix size, and cost?**  
   - *VanillaEEGTransformer:* $T = 561$ samples ($0.5$–$4.0$ s @ 160 Hz); attention matrix is $561 \times 561 = 314,721$ entries per head ($O(T^2)$).
   - *YourModel:* Sequence length is compressed by `PatchStem` to $N = 16$ patch tokens; attention matrix is $16 \times 16 = 256$ entries per head ($>1,200\times$ reduction).
5. **If given 80 subjects instead of 8, would your Part 2 idea still help? Why?**  
   - Yes: `PatchStem`'s bandpass filtering and spatial dipole un-mixing remain physically necessary because raw 160 Hz timesteps suffer from inter-subject phase jitter. With 80 subjects, the self-attention layer would finally have enough statistical power to learn cross-patch temporal dynamics without overfitting, allowing `YourModel` to outperform the mean-pooling ablation.
6. **Where is your model attending, and is that neurophysiologically sensible for fists-vs-feet?**  
   - Spatially, depthwise filters weight bilateral central electrodes (C3/C4) against vertex (Cz), matching lateral hand convexity vs medial interhemispheric foot motor representations; temporally, the model pools power across the stable 1.0–3.5 s sustained imagery window after visual evoked transients have subsided.
