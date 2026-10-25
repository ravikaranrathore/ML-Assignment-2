import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Set random seed
SEED = 26012
np.random.seed(SEED)

print("="*60)
print("Q2. MULTICLASS BOOSTING (SAMME) FROM SCRATCH")
print("="*60)

# SAMME AdaBoost Implementation (Simple / Beginner-Friendly)

class SimpleSAMME:
    def __init__(self, n_estimators=50, max_depth=2, random_state=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.alphas = []
        # Store training history for plotting
        self.weighted_errors = []
        self.max_sample_weights = []
        self.top10_sample_weights = []
        self.sample_weight_history = []
        self.pred_history = [] # predictions of each tree on train set
        
    def fit(self, X, y, K=None):
        N = len(y)
        if K is None:
            self.K = len(np.unique(y))
        else:
            self.K = K
            
        # Step 1: Initialize weights equally
        # w_i = 1 / N
        w = np.ones(N, dtype=np.float64) / N
        
        self.trees = []
        self.alphas = []
        self.weighted_errors = []
        self.max_sample_weights = []
        self.top10_sample_weights = []
        self.sample_weight_history = []
        self.pred_history = []
        
        # Random guessing error for K classes is (K - 1) / K
        random_error = (self.K - 1.0) / self.K
        
        for t in range(self.n_estimators):
            # Save current weight stats
            self.sample_weight_history.append(w.copy())
            self.max_sample_weights.append(np.max(w))
            # Sum of 10 heaviest weights
            sorted_w = np.sort(w)[::-1]
            self.top10_sample_weights.append(np.sum(sorted_w[:10]))
            
            # Fit weak learner with sample weights
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state + t
            )
            tree.fit(X, y, sample_weight=w)
            
            # Predictions of this weak learner
            y_pred = tree.predict(X)
            self.pred_history.append(y_pred)
            
            # Indicator of errors: 1 if wrong, 0 if correct
            incorrect = (y_pred != y).astype(float)
            
            # Weighted error epsilon_t
            eps_t = np.sum(w * incorrect) / np.sum(w)
            self.weighted_errors.append(eps_t)
            
            # Explicitly handle weak learner that is not better than random multiclass prediction
            if eps_t >= random_error:
                print(f"Warning: At round {t+1}, weak learner error {eps_t:.4f} >= random guessing {random_error:.4f}.")
                print("Stopping boosting early because weak learner is not better than random.")
                break
                
            # Avoid division by zero if error is 0
            if eps_t <= 1e-10:
                alpha_t = np.log((1.0 - 1e-10) / 1e-10) + np.log(self.K - 1.0)
                self.trees.append(tree)
                self.alphas.append(alpha_t)
                print(f"Round {t+1}: Perfect classification achieved (eps=0). Stopping early.")
                break
                
            # SAMME formula: alpha_t = log((1 - eps_t) / eps_t) + log(K - 1)
            alpha_t = np.log((1.0 - eps_t) / eps_t) + np.log(self.K - 1.0)
            
            self.trees.append(tree)
            self.alphas.append(alpha_t)
            
            # Weight update: w_i = w_i * exp(alpha_t * 1[y_i != h_t(x_i)])
            w = w * np.exp(alpha_t * incorrect)
            
            # Normalize weights so sum(w) = 1
            w = w / np.sum(w)
            
        return self
        
    def predict_scores_and_conf(self, X):
        # S_k(x) = sum_t alpha_t * 1[h_t(x) == k]
        N_samples = len(X)
        scores = np.zeros((N_samples, self.K), dtype=np.float64)
        total_alpha = np.sum(self.alphas)
        
        for tree, alpha in zip(self.trees, self.alphas):
            preds = tree.predict(X)
            for i in range(N_samples):
                pred_class = preds[i]
                scores[i, pred_class] += alpha
                
        # Predictions: argmax of score
        y_pred = np.argmax(scores, axis=1)
        
        # Confidence: q_k(x) = S_k(x) / sum_t alpha_t
        if total_alpha > 0:
            conf_matrix = scores / total_alpha
        else:
            conf_matrix = np.ones((N_samples, self.K)) / self.K
            
        # Winning confidence
        winning_conf = np.max(conf_matrix, axis=1)
        
        return y_pred, conf_matrix, winning_conf
        
    def predict(self, X):
        y_pred, _, _ = self.predict_scores_and_conf(X)
        return y_pred
        
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


# (a) TOY 3-CLASS EXAMPLE (AT MOST 10 SAMPLES)

print("\n" + "="*50)
print("(a) 3-Class Toy Example: Logging 1 Complete SAMME Round")
print("="*50)

# Create a simple toy dataset: 9 samples, 2 features, 3 classes (0, 1, 2)
# 3 samples per class
X_toy = np.array([
    [1.0, 1.0], [1.2, 1.5], [1.5, 1.1], # class 0
    [5.0, 5.0], [5.2, 4.8], [4.9, 5.3], # class 1
    [1.0, 5.0], [1.5, 4.8], [5.0, 1.2]  # class 2 (sample 8 is intentionally hard/noisy)
])
y_toy = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
N_toy = len(y_toy)
K_toy = 3

print(f"Number of samples: {N_toy}, Number of classes: {K_toy}")
print("Random guessing error benchmark for K=3: (K-1)/K = 2/3 = 0.6667")

# Initial weights
w_init = np.ones(N_toy) / N_toy
print("\n1. Initial Sample Weights (w_0):")
for i in range(N_toy):
    print(f"   Sample {i} (Class {y_toy[i]}): w = {w_init[i]:.4f}")

# Train a simple weak learner (stump: depth=1)
toy_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
toy_tree.fit(X_toy, y_toy, sample_weight=w_init)
preds_toy = toy_tree.predict(X_toy)

print("\n2. Weak Learner Predictions (h_1):")
for i in range(N_toy):
    status = "CORRECT" if preds_toy[i] == y_toy[i] else "WRONG"
    print(f"   Sample {i} (True: {y_toy[i]}, Pred: {preds_toy[i]}) -> {status}")

# Calculate weighted error
incorrect_toy = (preds_toy != y_toy).astype(float)
eps_toy = np.sum(w_init * incorrect_toy) / np.sum(w_init)
print(f"\n3. Weighted Training Error (eps_1): {eps_toy:.4f}")

# Calculate alpha_1
# alpha_1 = log((1 - eps) / eps) + log(K - 1)
alpha_toy = np.log((1.0 - eps_toy) / eps_toy) + np.log(K_toy - 1.0)
print(f"4. Learner Weight (alpha_1):")
print(f"   log((1 - {eps_toy:.4f}) / {eps_toy:.4f}) + log({K_toy - 1}) = {np.log((1-eps_toy)/eps_toy):.4f} + {np.log(K_toy-1):.4f} = {alpha_toy:.4f}")

# Update weights
w_unnorm = w_init * np.exp(alpha_toy * incorrect_toy)
w_norm = w_unnorm / np.sum(w_unnorm)

print("\n5. Updated Normalized Weights (w_1):")
for i in range(N_toy):
    print(f"   Sample {i} (True: {y_toy[i]}, Pred: {preds_toy[i]}): unnorm = {w_unnorm[i]:.4f} -> normalized w = {w_norm[i]:.4f}")
print(f"   Sum of normalized weights: {np.sum(w_norm):.4f}")


# (b) OLIVETTI FACES: GRID SEARCH OVER DEPTHS & ROUNDS

print("\n" + "="*50)
print("(b) Olivetti Faces: Depths {1, 2, 3, 4} and Rounds {10, 25, 50, 100}")
print("="*50)

# Load saved split from Q1
out_dir = r"D:\ML_Assignment"
fig_dir = os.path.join(out_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

split_file = os.path.join(out_dir, "olivetti_split.npz")
split_data = np.load(split_file)
train_indices = split_data['train_indices']
val_indices = split_data['val_indices']
test_indices = split_data['test_indices']

from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces(shuffle=False)
X_all = faces.data
y_all = faces.target

X_train, y_train = X_all[train_indices], y_all[train_indices]
X_val, y_val = X_all[val_indices], y_all[val_indices]
X_test_clean, y_test = X_all[test_indices], y_all[test_indices]

# Masked test set: rows 24:40, cols 24:40 zeroed out
X_test_masked = X_test_clean.copy().reshape(-1, 64, 64)
X_test_masked[:, 24:40, 24:40] = 0.0
X_test_masked = X_test_masked.reshape(-1, 4096)

print(f"Dataset split loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test_clean)}")

depths = [1, 2, 3, 4]
rounds_list = [10, 25, 50, 100]

grid_results = []
best_val_acc = -1.0
selected_model_config = None

for d in depths:
    for r in rounds_list:
        model = SimpleSAMME(n_estimators=r, max_depth=d, random_state=SEED)
        model.fit(X_train, y_train, K=40)
        
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        test_acc = model.score(X_test_clean, y_test)
        
        grid_results.append({
            'depth': d,
            'rounds': r,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'model': model
        })
        print(f"Depth={d}, Rounds={r:3d} -> Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")

# Model selection using clean validation accuracy only
best_val_acc = max(res['val_acc'] for res in grid_results)
best_candidates = [res for res in grid_results if res['val_acc'] == best_val_acc]

# If tie, prefer lower depth or fewer rounds for simplicity
def sort_pref(res):
    return (res['depth'], res['rounds'])

best_candidates.sort(key=sort_pref)
selected_res = best_candidates[0]
selected_depth = selected_res['depth']
selected_rounds = selected_res['rounds']
selected_boost_model = selected_res['model']

print("\n" + "-"*50)
print(f"SELECTED BOOSTING MODEL (Clean Val Acc Only):")
print(f"Depth = {selected_depth}, Rounds = {selected_rounds}")
print(f"Train Acc: {selected_res['train_acc']:.4f}")
print(f"Val Acc:   {selected_res['val_acc']:.4f}")
print(f"Test Acc:  {selected_res['test_acc']:.4f}")
print("-"*50)

# Plot for selected model:
# 1. Weighted training error
# 2. Learner weight
# 3. Maximum sample weight
# 4. Total weight of 10 heaviest training images
rounds_x = list(range(1, len(selected_boost_model.alphas) + 1))
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Subplot 1: Weighted training error
axes[0, 0].plot(rounds_x, selected_boost_model.weighted_errors[:len(rounds_x)], color='red', marker='o', markersize=3)
axes[0, 0].set_title("Weighted Training Error (eps_t)")
axes[0, 0].set_xlabel("Round t")
axes[0, 0].set_ylabel("Error eps_t")
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Learner weight
axes[0, 1].plot(rounds_x, selected_boost_model.alphas, color='blue', marker='s', markersize=3)
axes[0, 1].set_title("Learner Weight (alpha_t)")
axes[0, 1].set_xlabel("Round t")
axes[0, 1].set_ylabel("alpha_t")
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Maximum sample weight
axes[0, 2 if False else 1].set_visible(True) # placeholder check
axes[1, 0].plot(rounds_x, selected_boost_model.max_sample_weights[:len(rounds_x)], color='green', marker='^', markersize=3)
axes[1, 0].set_title("Maximum Sample Weight max_i(w_i)")
axes[1, 0].set_xlabel("Round t")
axes[1, 0].set_ylabel("Max Weight")
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Total weight of 10 heaviest samples
axes[1, 1].plot(rounds_x, selected_boost_model.top10_sample_weights[:len(rounds_x)], color='purple', marker='d', markersize=3)
axes[1, 1].set_title("Total Weight of Top 10 Heaviest Samples")
axes[1, 1].set_xlabel("Round t")
axes[1, 1].set_ylabel("Sum of Top 10 Weights")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f"SAMME Boosting Dynamics (Depth={selected_depth}, Rounds={selected_rounds})", fontsize=14, fontweight='bold')
plt.tight_layout()
q2b_plot_path = os.path.join(fig_dir, "q2b_boosting_dynamics.png")
plt.savefig(q2b_plot_path, dpi=300)
plt.close()
print(f"Saved Q2(b) plot to {q2b_plot_path}")


# (c) COMPARE BOOSTING AND BAGGING (25, 50, 100 LEARNERS)

print("\n" + "="*50)
print("(c) Compare Boosting vs Bagging (25, 50, 100 Learners)")
print("="*50)

# Import bagging from previous script or implement simple bagging here
# Remember Q1 frozen tree setting: max_depth=None, min_samples_leaf=1
class SimpleBagging:
    def __init__(self, n_estimators=50, random_state=SEED):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        N = len(y)
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for i in range(self.n_estimators):
            boot_idx = rng.choice(N, size=N, replace=True)
            tree = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=int(rng.integers(0, 1000000)))
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
        return self
        
    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees])
        # majority vote
        final_preds = []
        for col in preds.T:
            counts = np.bincount(col, minlength=40)
            final_preds.append(np.argmax(counts))
        return np.array(final_preds)
        
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

learner_counts = [25, 50, 100]
comparison_results = []

for M in learner_counts:
    # 1. Boosting with depth = selected_depth
    t0 = time.time()
    boost_m = SimpleSAMME(n_estimators=M, max_depth=selected_depth, random_state=SEED)
    boost_m.fit(X_train, y_train, K=40)
    boost_time = time.time() - t0
    
    b_tr_acc = boost_m.score(X_train, y_train)
    b_va_acc = boost_m.score(X_val, y_val)
    b_te_acc = boost_m.score(X_test_clean, y_test)
    
    # 2. Bagging with frozen unpruned trees
    t0 = time.time()
    bag_m = SimpleBagging(n_estimators=M, random_state=SEED)
    bag_m.fit(X_train, y_train)
    bag_time = time.time() - t0
    
    g_tr_acc = bag_m.score(X_train, y_train)
    g_va_acc = bag_m.score(X_val, y_val)
    g_te_acc = bag_m.score(X_test_clean, y_test)
    
    comparison_results.append({
        'M': M,
        'boost_tr': b_tr_acc, 'boost_va': b_va_acc, 'boost_te': b_te_acc, 'boost_time': boost_time,
        'bag_tr': g_tr_acc, 'bag_va': g_va_acc, 'bag_te': g_te_acc, 'bag_time': bag_time
    })
    
    print(f"Learners M = {M:3d}:")
    print(f"  Boosting: Train={b_tr_acc:.4f}, Val={b_va_acc:.4f}, Test={b_te_acc:.4f}, Time={boost_time:.2f}s")
    print(f"  Bagging:  Train={g_tr_acc:.4f}, Val={g_va_acc:.4f}, Test={g_te_acc:.4f}, Time={bag_time:.2f}s")

# Inspect one training face that receives a very large boosting weight
# Look at the final sample weights of the selected boosting model
final_weights = selected_boost_model.sample_weight_history[-1]
heaviest_idx = np.argmax(final_weights)
heaviest_weight = final_weights[heaviest_idx]
heaviest_class = y_train[heaviest_idx]

print("\n--- Heaviest Training Sample Inspection ---")
print(f"Sample Index in Train Set: {heaviest_idx}")
print(f"True Class: {heaviest_class}")
print(f"Final Weight: {heaviest_weight:.4f} (Base uniform weight was {1.0/len(y_train):.4f})")
print(f"Weight is {heaviest_weight / (1.0/len(y_train)):.1f}x higher than uniform!")

# Check prediction history of this face across rounds
preds_history = np.array(selected_boost_model.pred_history) # shape (T, N_train)
sample_preds = preds_history[:, heaviest_idx]
misclassified_rounds = np.where(sample_preds != heaviest_class)[0]
n_misclass = len(misclassified_rounds)

print(f"Out of {len(sample_preds)} weak learners, it was misclassified in {n_misclass} rounds ({n_misclass/len(sample_preds)*100:.1f}% of rounds)!")
print("Top predicted classes for this sample by weak learners:")
pred_counts = np.bincount(sample_preds, minlength=40)
for c in np.argsort(pred_counts)[::-1][:4]:
    if pred_counts[c] > 0:
        print(f"  Class {c}: {pred_counts[c]} times")


# (d) COMPARISON ON CLEAN AND MASKED TEST SETS

print("\n" + "="*50)
print("(d) Freeze Selected Model: Baseline Tree vs Bagging vs Boosting")
print("="*50)

# Single tree baseline (frozen from Q1: max_depth=None, min_samples_leaf=1)
tree_baseline = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=SEED)
tree_baseline.fit(X_train, y_train)

# Final bagged model (from Q1: 200 trees)
final_bagging = SimpleBagging(n_estimators=200, random_state=SEED)
final_bagging.fit(X_train, y_train)

# Selected boosting model
frozen_boosting = selected_boost_model

# 1. Accuracies and confidences on Clean Test Set
# Single tree clean
t_clean_pred = tree_baseline.predict(X_test_clean)
t_clean_acc = np.mean(t_clean_pred == y_test)
t_clean_proba = tree_baseline.predict_proba(X_test_clean)
t_clean_conf = np.mean(np.max(t_clean_proba, axis=1))

# Bagging clean
g_clean_preds_matrix = np.array([t.predict(X_test_clean) for t in final_bagging.trees])
g_clean_pred = []
g_clean_confs = []
for col in g_clean_preds_matrix.T:
    counts = np.bincount(col, minlength=40)
    g_clean_pred.append(np.argmax(counts))
    g_clean_confs.append(np.max(counts) / len(final_bagging.trees))
g_clean_pred = np.array(g_clean_pred)
g_clean_acc = np.mean(g_clean_pred == y_test)
g_clean_conf = np.mean(g_clean_confs)

# Boosting clean
b_clean_pred, _, b_clean_confs = frozen_boosting.predict_scores_and_conf(X_test_clean)
b_clean_acc = np.mean(b_clean_pred == y_test)
b_clean_conf = np.mean(b_clean_confs)

# 2. Accuracies and confidences on Masked Test Set
# Single tree masked
t_masked_pred = tree_baseline.predict(X_test_masked)
t_masked_acc = np.mean(t_masked_pred == y_test)
t_masked_proba = tree_baseline.predict_proba(X_test_masked)
t_masked_conf = np.mean(np.max(t_masked_proba, axis=1))

# Bagging masked
g_masked_preds_matrix = np.array([t.predict(X_test_masked) for t in final_bagging.trees])
g_masked_pred = []
g_masked_confs = []
for col in g_masked_preds_matrix.T:
    counts = np.bincount(col, minlength=40)
    g_masked_pred.append(np.argmax(counts))
    g_masked_confs.append(np.max(counts) / len(final_bagging.trees))
g_masked_pred = np.array(g_masked_pred)
g_masked_acc = np.mean(g_masked_pred == y_test)
g_masked_conf = np.mean(g_masked_confs)

# Boosting masked
b_masked_pred, _, b_masked_confs = frozen_boosting.predict_scores_and_conf(X_test_masked)
b_masked_acc = np.mean(b_masked_pred == y_test)
b_masked_conf = np.mean(b_masked_confs)

print("Model Comparison Summary:")
print(f"Single Tree Baseline: Clean Acc={t_clean_acc:.4f} (Conf={t_clean_conf:.4f}) | Masked Acc={t_masked_acc:.4f} (Conf={t_masked_conf:.4f})")
print(f"Final Bagged Model:   Clean Acc={g_clean_acc:.4f} (Conf={g_clean_conf:.4f}) | Masked Acc={g_masked_acc:.4f} (Conf={g_masked_conf:.4f})")
print(f"Selected Boosting:    Clean Acc={b_clean_acc:.4f} (Conf={b_clean_conf:.4f}) | Masked Acc={b_masked_acc:.4f} (Conf={b_masked_conf:.4f})")

# Find a test face where Bagging and Boosting respond differently to the masked region
# For example, Bagging gets it right on masked test, while Boosting gets it wrong!
diff_indices = np.where((g_masked_pred == y_test) & (b_masked_pred != y_test))[0]
if len(diff_indices) == 0:
    diff_indices = np.where((g_masked_pred != y_test) & (b_masked_pred == y_test))[0]

diff_idx = diff_indices[0]
true_diff_class = y_test[diff_idx]
bag_diff_p = g_masked_pred[diff_idx]
boost_diff_p = b_masked_pred[diff_idx]

# Tree vote count for Bagging on this sample
bag_votes = np.bincount(g_masked_preds_matrix[:, diff_idx], minlength=40)
bag_top = np.argsort(bag_votes)[::-1][:3]

# Boosting scores
_, b_scores_matrix, _ = frozen_boosting.predict_scores_and_conf(X_test_masked[diff_idx:diff_idx+1])
boost_top = np.argsort(b_scores_matrix[0])[::-1][:3]

print(f"\n--- Differently Responding Test Face (Index {diff_idx}) ---")
print(f"True Class: {true_diff_class}")
print(f"Bagging Prediction: {bag_diff_p} (CORRECT: {bag_diff_p == true_diff_class}) | Top Votes: {[(c, bag_votes[c]) for c in bag_top]}")
print(f"Boosting Prediction: {boost_diff_p} (CORRECT: {boost_diff_p == true_diff_class}) | Top Probs: {[(c, round(b_scores_matrix[0, c], 3)) for c in boost_top]}")

# Save visualization of this face (Clean vs Masked)
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
axes[0].imshow(X_test_clean[diff_idx].reshape(64, 64), cmap='gray')
axes[0].set_title(f"Clean Face (Class {true_diff_class})")
axes[0].axis('off')

axes[1].imshow(X_test_masked[diff_idx].reshape(64, 64), cmap='gray')
axes[1].set_title(f"Masked Face (Sample {diff_idx})\nBagging: Class {bag_diff_p} (✓)\nBoosting: Class {boost_diff_p} (✗)")
axes[1].axis('off')

plt.tight_layout()
diff_plot_path = os.path.join(fig_dir, "q2d_different_response_face.png")
plt.savefig(diff_plot_path, dpi=300)
plt.close()
print(f"Saved differential face plot to {diff_plot_path}")
