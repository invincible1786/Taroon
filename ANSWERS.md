# ANSWERS.md — EEG × Deep Learning Live Exam
**Candidate:** AI Research Engineer Candidate  
**Repository:** `eeg_dl_exam_3hr`  
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
