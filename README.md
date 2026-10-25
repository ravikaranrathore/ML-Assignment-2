# Machine Learning Assignment: Ensembles, Bagging, Boosting, and Bias-Variance Decomposition

**Author:** Ravikaran Singh  
**Roll Number:** `M26CSE012`  
**Dataset:** Olivetti Faces (40 identities, 10 images per identity, $64 \times 64$ raw flattened pixels)  
**Deterministic Split Seed:** `26012` (all numeric characters of complete roll number in order)  

---

## 📌 Overview

This repository contains the complete from-scratch implementation and empirical evaluation for the Machine Learning Assignment on **Ensemble Learning, Decision Trees, Bagging, SAMME Multiclass Boosting, and Bias-Variance Decomposition** on the Olivetti Faces benchmark dataset.

Key components of the assignment:
- **No Dimensionality Reduction:** All models operate purely on raw normalized pixel vectors ($x \in [0, 1]^{4096}$).
- **Deterministic Partitioning:** Stratified random split within each identity: 6 training, 2 validation, and 2 test images per identity ($N_{\text{train}} = 240$, $N_{\text{val}} = 80$, $N_{\text{test}} = 80$). Saved to `olivetti_split.npz` and reused everywhere.
- **Robustness Masking Protocol:** A second test set with the centered $16 \times 16$ block (rows 24 to 39, columns 24 to 39) zeroed out to evaluate model resilience to facial occlusion. Models are trained **strictly on clean images**.
- **Scratch Implementations:** While `DecisionTreeClassifier` is permitted for base tree nodes, **Bagging** and **SAMME Multiclass Boosting** are implemented from scratch.

---

## 📂 Repository Structure

```text
├── README.md                      # Comprehensive project documentation & reproduction guide
├── requirements.txt               # Pip dependencies
├── environment.yml                # Conda environment definition
├── olivetti_split.npz             # Serialized deterministic train/val/test split indices
├── run_experiments.py             # Q1: Decision tree baseline & bagging from scratch
├── q2_boosting.py                 # Q2: SAMME multiclass boosting from scratch
├── q3_bias_variance.py            # Q3: Bias and variance measured quantities (30 refits)
└── figures/                       # Generated high-resolution plots
    ├── sample_clean_vs_masked.png          # Visualizing 16x16 center occlusion
    ├── q1c_bagging_metrics_vs_B.png        # OOB accuracy, disagreement, and vote margin vs B
    ├── q1d_case_studies.png                # 4 case study faces (bagging corrections vs joint errors)
    ├── q2b_boosting_dynamics.png           # SAMME boosting error, alpha, and sample weights over rounds
    ├── q2d_different_response_face.png     # Differential face response to occlusion (Bagging vs Boosting)
    └── q3_bias_variance_comparison.png     # Measured Bias^2 and Variance across models & depths
```

---

## ⚙️ Environment Setup & Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/ravikaranrathore/ML-Assignment-2.git
cd ML-Assignment-2

# Create and activate environment
conda env create -f environment.yml
conda activate ml_ensembles
```

### Option 2: Using Pip and Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Exact Reproduction Commands

All scripts are completely deterministic and self-contained. Run them in order:

### 1. Run Q1 (Decision Tree Baseline & Bagging from Scratch)
```bash
python run_experiments.py
```
- Performs hyperparameter grid search over `max_depth` $\in \{2, 4, 8, 16, \text{None}\}$ and `min_samples_leaf` $\in \{1, 2, 4\}$.
- Selects frozen baseline model on **validation accuracy only** (`max_depth = None`, `min_samples_leaf = 1`).
- Evaluates Bagging from scratch for $B \in \{1, 5, 10, 25, 50, 100, 200\}$.
- Verifies empirical bootstrap distinct samples against theoretical expectation ($151.89$ samples).
- Computes Out-of-Bag (OOB) accuracy, pairwise tree disagreement, and mean vote margin.
- Analyzes clean vs. masked robustness and outputs 4 case-study face comparisons.

### 2. Run Q2 (Multiclass Boosting - SAMME from Scratch)
```bash
python q2_boosting.py
```
- Traces 1 complete SAMME round on a 3-class toy dataset (verifying random guessing guard $\frac{K-1}{K}$).
- Performs grid search on Olivetti Faces: `max_depth` $\in \{1, 2, 3, 4\}$ and `rounds` $\in \{10, 25, 50, 100\}$.
- Selects optimal boosting model on **clean validation accuracy only** (`depth = 4`, `rounds = 100`).
- Plots training dynamics: weighted error $\epsilon_t$, learner weight $\alpha_t$, max sample weight, and top-10 sample weights.
- Compares Bagging vs. Boosting across 25, 50, and 100 learners (accuracies and training runtimes).
- Identifies and inspects the heaviest training sample (Sample 180, Class 30, $14.7\times$ weight amplification).
- Finds and explains a differential test face under occlusion masking.

### 3. Run Q3 (Bias and Variance as Measured Quantities across 30 Refits)
```bash
python q3_bias_variance.py
```
- Generates 30 independent training sets by resampling 6 faces with replacement per identity from the original training set.
- Refits the selected Single Tree, Final Bagged Model, and Selected Boosting Model on each resampled set without re-tuning.
- Computes empirical $\text{Bias}^2 = \frac{1}{N}\sum_i \|\bar{p}(x_i) - e_i\|_2^2$ and $\text{Variance} = \frac{1}{NR}\sum_{i,r} \|p_r(x_i) - \bar{p}(x_i)\|_2^2$.
- Evaluates single trees and 50-tree bagged ensembles across depth regimes: Shallow (Depth 2), Medium (Depth 4), Deep (Depth 8).
- Quantifies variance reduction and accuracy gain as a function of base learner depth.
- Compares clean test accuracy, masked test accuracy, $\text{Bias}^2$, and $\text{Variance}$ across all models.

---

## 📊 Summary of Experimental Results

### Q1: Decision Tree Baseline & Bagging from Scratch

#### 1. Single Tree Hyperparameter Tuning (Selected on Clean Validation Accuracy Only)
- **Selected Frozen Configuration:** `max_depth = None`, `min_samples_leaf = 1`
- **Train Accuracy:** $1.0000$ ($100.0\%$)
- **Validation Accuracy:** $0.5125$ ($51.25\%$)
- **Test Accuracy:** $0.6875$ ($68.75\%$)
- **Tree Structure:** 46 leaves, mean 5.22 samples/leaf, $13.04\%$ singleton leaves ($n=1$).

#### 2. Bootstrap Diversity: Theory vs. Experiment
- **Theoretical Expectation:**
  $$\mathbb{E}[U] = N \left[1 - \left(1 - \frac{1}{N}\right)^N\right] = 240 \times \left[1 - \left(\frac{239}{240}\right)^{240}\right] \approx \mathbf{151.89} \quad (63.29\%)$$
  $$\text{Asymptotic Limit: } N (1 - e^{-1}) \approx 240 \times 0.63212 = \mathbf{151.71} \quad (63.21\%)$$
- **Empirical Average ($B=200$):** **$151.9$** unique samples per bootstrap draw, validating theoretical derivation.

#### 3. Bagging Ensemble Trajectory Across $B$

| $B$ | Train Acc | Val Acc | Test Acc | Avg Unique Samples | OOB Acc (Coverage) | Pairwise Disagreement | Mean Vote Margin |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.692 | 0.288 | 0.338 | 148.0 | 0.196 (38%) | 0.000 | 1.000 |
| 5 | 0.975 | 0.650 | 0.625 | 151.6 | 0.283 (91%) | 0.795 | 0.240 |
| 10 | 0.996 | 0.775 | 0.738 | 151.8 | 0.382 (99%) | 0.787 | 0.260 |
| 25 | 1.000 | 0.762 | 0.838 | 151.3 | 0.571 (100%) | 0.777 | 0.268 |
| 50 | 1.000 | 0.825 | 0.850 | 152.2 | 0.692 (100%) | 0.782 | 0.271 |
| 100 | 1.000 | 0.788 | 0.838 | 152.2 | 0.750 (100%) | 0.779 | 0.273 |
| **200** | **1.000** | **0.838** | **0.850** | **151.9** | **0.792 (100%)** | **0.783** | **0.276** |

- **Selected Ensemble Size:** $B^* = 200$ (Validation Accuracy $= 83.75\%$, OOB Accuracy $= 79.17\%$, Test Accuracy $= 85.00\%$).
- **Metric Concordance:** Pairwise disagreement stays constant around $\approx 78\%$ (reflecting intrinsic base learner diversity), while OOB accuracy and vote margin rise monotonically as variance cancels out by the Law of Large Numbers.

---

### Q2: Multiclass Boosting (SAMME) from Scratch

#### 1. 3-Class Toy Trace (1 Full SAMME Round)
- $N = 9$ samples, $K = 3$ classes. Random guessing threshold: $(K-1)/K = 0.6667$.
- Weak learner misclassifies 3 samples $\implies \epsilon_1 = 3/9 = \mathbf{0.3333}$.
- Learner weight: $\alpha_1 = \log(\frac{1 - 0.3333}{0.3333}) + \log(2) = 0.6931 + 0.6931 = \mathbf{1.3863}$.
- Correct samples normalized weight: $\mathbf{0.0556}$; misclassified samples: $\mathbf{0.2222}$ ($4\times$ amplification).

#### 2. Selected Boosting Model on Faces
- **Selected Configuration (Val Acc only):** `max_depth = 4`, `rounds = 100` (Val Acc: $50.00\%$, Test Acc: $62.50\%$).
- **Dynamics:** By round 100, the top-10 heaviest samples carry over **$35\%$** of all sample weight.
- **Outlier Sample 180 (Identity 30):** Amplified to $14.7\times$ uniform weight; misclassified in $87\%$ of rounds due to atypical lighting and head tilt that shallow depth-4 trees cannot separate.

#### 3. Bagging vs. Boosting Comparison

| Learners $M$ | Model | Train Acc | Val Acc | Clean Test Acc | Training Time |
|:---:|:---|:---:|:---:|:---:|:---:|
| **25** | Boosting (Depth 4) | 0.5125 | 0.2750 | 0.2375 | **10.82s** |
| | Bagging (Unpruned) | 1.0000 | 0.7750 | 0.8125 | 36.51s |
| **50** | Boosting (Depth 4) | 0.9292 | 0.3750 | 0.4625 | **21.74s** |
| | Bagging (Unpruned) | 1.0000 | 0.8125 | 0.8875 | 75.85s |
| **100** | Boosting (Depth 4) | 1.0000 | 0.5000 | 0.6250 | **42.29s** |
| | Bagging (Unpruned) | 1.0000 | 0.8250 | 0.8875 | 155.03s |

---

### Q3: Bias and Variance as Measured Quantities (30 Refits)

#### 1. Measured Quantities Across 30 Resampled Refits

| Model | Specification | Measured $\text{Bias}^2$ | Measured $\text{Variance}$ | Test Accuracy ($\text{Mean} \pm \text{Std}$) |
|:---|:---|:---:|:---:|:---:|
| **Selected Single Tree** | Unpruned (`max_depth=None`) | **0.3023** | **0.6002** | $54.87\% \pm 4.88\%$ |
| **Final Bagged Model** | Unpruned Trees ($B=50$) | **0.4608** | **0.0331** | **$77.42\% \pm 3.79\%$** |
| **Selected Boosting Model** | SAMME ($M=50$, `max_depth=4`) | **0.8759** | **0.0218** | $29.88\% \pm 8.86\%$ |

- **Bagging removes $94.5\%$ of single tree variance** ($0.6002 \to 0.0331$), lifting test accuracy by $+22.55\%$.
- **Lowest variance does not imply best accuracy:** Boosting achieves the lowest variance ($0.0218$), but suffers from high bias ($0.8759$) because depth-4 trees underfit 40 classes under resampling.

#### 2. Depth Progression & Variance Reduction

| Depth | Model Family | $\text{Bias}^2$ | $\text{Variance}$ | Test Acc | Variance Removed by Bagging | Var Reduction % | Accuracy Gain | Mean Leaves |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2** | Single Tree / Bagging | 0.9101 / 0.9276 | 0.0419 / 0.0197 | 6.54% / 10.54% | **0.0222** | **52.9%** | **+4.00%** | 3.1 |
| **4** | Single Tree / Bagging | 0.8500 / 0.8798 | 0.0805 / 0.0228 | 10.54% / 22.71% | **0.0576** | **71.6%** | **+12.17%** | 5.3 |
| **8** | Single Tree / Bagging | 0.7537 / 0.7879 | 0.1421 / 0.0253 | 17.79% / 45.08% | **0.1168** | **82.2%** | **+27.29%** | 9.6 |

- Deeper trees exhibit higher variance and lower bias, providing maximum headroom where bagging removes variance most effectively.

#### 3. Robustness Comparison: Clean vs. Occlusion Masked Test Set

| Model | Clean Test Acc | Masked Test Acc | Occlusion Accuracy Drop | $\text{Bias}^2$ | $\text{Variance}$ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Selected Single Tree** | $54.87\%$ | $34.04\%$ | **$-20.83\%$** | **0.3023** | **0.6002** |
| **Final Bagged Model** | **$77.42\%$** | **$63.08\%$** | **$-14.33\%$** | **0.4608** | **0.0331** |
| **Selected Boosting Model** | $29.88\%$ | $25.96\%$ | **$-3.92\%$** | **0.8759** | **0.0218** |

- **Most Stable Model:** Boosting ($\text{Variance} = 0.0218$) and Bagging ($\text{Variance} = 0.0331$). Single tree is highly unstable ($0.6002$).
- **Most Sensitive to Occlusion:** Single tree suffers the largest drop ($-20.83\%$). Bagging is more resilient ($-14.33\%$) due to decorrelated tree feature selection across peripheral pixels (hairline, jawline, forehead).

---

## 📜 Reproducibility & Citation

```bibtex
@misc{singh2026ensembles,
  author = {Ravikaran Singh},
  title = {Empirical Analysis of Bagging, SAMME Boosting, and Bias-Variance Decomposition on Face Identification},
  year = {2026},
  institution = {Indian Institute of Technology Jodhpur},
  note = {Roll Number: M26CSE012}
}
```
