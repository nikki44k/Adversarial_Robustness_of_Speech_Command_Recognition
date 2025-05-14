# Exploring the Adversarial Robustness of Speech-Command Recognition

**Team**  
- Nikhitha Kilari  
- Jishnuvardhan Karpuram  

---

## Overview
We study how small, imperceptible perturbations can fool a 1D-CNN speech classifier and which lightweight defenses can mitigate those attacks. To ensure reproducibility and drive-friendly execution, we swapped the originally proposed Google Speech Commands v2 dataset (>1 GB) for the smaller, publicly available **Free Spoken Digit Dataset (FSDD)** (~50 MB, 10 digit classes).

---

## Dataset & Preprocessing

- **Free Spoken Digit Dataset (FSDD)**  
  – 10 classes (0–9), ~1 500 recordings  
  – Downloaded automatically via Kaggle CLI into Google Drive

- **Preprocessing steps**  
  1. Resample to 8 kHz (using `torchaudio.functional.resample`)  
  2. Trim or pad to exactly 1 second (8 000 samples)  
  3. Normalize amplitude to [-1, 1] and cast to `float32`

- **Train/Test Split**  
  – 80/20 split using a fixed random seed for reproducibility

---

## Baseline Model

- **Conv1DSpeech**  
  – 3× (Conv1d → BatchNorm1d → ReLU) blocks  
  – Global average pooling → Linear(64 → 10)  
- **Clean Accuracy Goal**: ~85–88% on the FSDD test set

---

## Adversarial Attacks

We implement **six** novel black-box and psychoacoustic attacks in pure PyTorch/NumPy (no external ART dependencies), operating on small batches (size=8) to avoid OOMs:

1. **SPSA** – gradient approximation via finite differences  
2. **GenAttack** – evolutionary algorithm, gradient-free optimization  
3. **SimBA Audio** – iterative single-sample perturbations  
4. **Spatial Transform** – pitch-shifting & time-stretching via `librosa`  
5. **Psychoacoustic Masking** – Gaussian noise injected at 30 dB SNR  
6. **Hidden-Voice** – mixing a low-amplitude digit phrase into the waveform

For each attack we will:

- Tune the attack budget (iterations, ε, population size, transform magnitude)  
- Measure **top-1 accuracy drop**, **mean signal-to-noise ratio (SNR)** as a stealth proxy  
- Visualize **Accuracy vs. SNR** to illustrate the stealth–success trade-off

---

## Defense Strategies

We evaluate **three** lightweight defense mechanisms on the same test split:

- **Randomized Smoothing**  
  – Add Gaussian noise at inference (σ = 0.001, 0.002, 0.004)  
  – Majority-vote over 10 noisy copies

- **Feature Squeezing**  
  – Quantize audio bit-depth to 2, 4, and 8 bits  
  – Removes high-frequency adversarial artifacts

- **Defensive Distillation**  
  – Train a “student” on teacher’s soft labels (temperature T = 20)  
  – 5 epochs of KL-divergence distillation

For each defense we will report both **clean** and **robust** accuracy, then present a summary bar chart comparing all three.

---

## Deliverables & Timeline

- **Notebooks**  
  1. `download_fsdd.ipynb` – automated Kaggle download  
  2. `1_data_preprocessing.ipynb` – audio loading & pipeline demo  
  3. `2_baseline_model.ipynb` – training `Conv1DSpeechBig` baseline  
  4. `3_attacks.ipynb` – six attack implementations & stealth evaluation  
  5. `4_defenses.ipynb` – three defense evaluations & summary plots  

- **Code modules**  
  – `models.py`, `audio_utils.py`, attack/defense helper scripts  

- **Final report**  
  – PDF & Word summarizing methods, results, and recommendations  

- **GitHub repository**  
  – All code, notebooks, data pointers, and the final report  

**Milestones**  
- **Week 1–2:** Data download, baseline training, clean accuracy  
- **Week 3:** Implement & tune six adversarial attacks  
- **Week 4:** Implement three defenses & gather results  
- **Week 5:** Write report, polish notebooks, prepare GitHub submission  

---

## Impact

By focusing on a smaller, manageable dataset and fully self-contained code, our pipeline is reproducible in limited-resource environments while still yielding actionable insights into adversarial robustness for real-world speech systems.  
