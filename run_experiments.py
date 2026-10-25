import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROLL_NUMBER = "M26CSE012"
NUMERIC_SEED = 26012

print(f"=== Olivetti Faces Machine Learning Assignment ===")
print(f"Roll Number: {ROLL_NUMBER}")
print(f"RNG Seed: {NUMERIC_SEED}")

# Create output directory
out_dir = r"D:\ML_Assignment"
fig_dir = os.path.join(out_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 1. Load Dataset
print("\n[Step 1] Loading Olivetti faces dataset...")
faces = fetch_olivetti_faces(shuffle=False)
X_all = faces.data # 400 x 4096, float32 in [0, 1]
y_all = faces.target # 400, ints in [0, 39]
n_samples, n_features = X_all.shape
n_classes = len(np.unique(y_all))

print(f"Dataset shape: {n_samples} samples, {n_features} features (64x64 raw pixels), {n_classes} identities.")

# 2. Train / Val / Test Split
# "Within each identity, randomly split the 10 images into 6 train, 2 validation and 2 test images. Save this split and use it everywhere."
rng_split = np.random.default_rng(NUMERIC_SEED)

train_indices = []
val_indices = []
test_indices = []

for c in range(n_classes):
    c_indices = np.where(y_all == c)[0]
    perm = rng_split.permutation(c_indices)
    train_indices.extend(perm[:6])
    val_indices.extend(perm[6:8])
    test_indices.extend(perm[8:10])

train_indices = np.array(train_indices)
val_indices = np.array(val_indices)
test_indices = np.array(test_indices)

# Save split
split_file = os.path.join(out_dir, "olivetti_split.npz")
np.savez(split_file, train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)
print(f"Saved split to {split_file}")
print(f"Split sizes: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

X_train, y_train = X_all[train_indices], y_all[train_indices]
X_val, y_val = X_all[val_indices], y_all[val_indices]
X_test_clean, y_test = X_all[test_indices], y_all[test_indices]

# 3. Masked Test Set
# "create a second test set by setting the centred 16 × 16 block, rows and columns 24 to 39, to zero. Models must still be trained only on clean images."
X_test_masked = X_test_clean.copy().reshape(-1, 64, 64)
# Rows 24 to 39 inclusive, columns 24 to 39 inclusive (Python slice [24:40, 24:40])
X_test_masked[:, 24:40, 24:40] = 0.0
X_test_masked = X_test_masked.reshape(-1, 4096)
print(f"Masked test set created: center 16x16 block zeroed out.")

# Save a sample clean vs masked image visualization
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(X_test_clean[0].reshape(64, 64), cmap='gray')
axes[0].set_title("Clean Test Image (Class {})".format(y_test[0]))
axes[0].axis('off')
axes[1].imshow(X_test_masked[0].reshape(64, 64), cmap='gray')
axes[1].set_title("Masked Test Image (Center 16x16)")
axes[1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "sample_clean_vs_masked.png"), dpi=200)
plt.close()

# Q1(a): Decision Tree Baseline
# max_depth in {2, 4, 8, 16, None}
# min_samples_leaf in {1, 2, 4}

print("\n" + "="*50)
print("Q1(a) Single Tree Baseline Evaluation")
print("="*50)

max_depths = [2, 4, 8, 16, None]
min_samples_leafs = [1, 2, 4]

tree_results = []
best_val_acc = -1.0
best_config = None
best_model = None

for md in max_depths:
    for msl in min_samples_leafs:
        clf = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=NUMERIC_SEED)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        test_acc = clf.score(X_test_clean, y_test)
        
        # Tree structure analysis
        n_leaves = clf.get_n_leaves()
        is_leaf = (clf.tree_.children_left == -1)
        leaf_samples = clf.tree_.n_node_samples[is_leaf]
        mean_samples_leaf = np.mean(leaf_samples)
        frac_single_sample = np.mean(leaf_samples == 1)
        
        tree_results.append({
            'max_depth': md,
            'min_samples_leaf': msl,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'n_leaves': n_leaves,
            'mean_samples_leaf': mean_samples_leaf,
            'frac_single_sample': frac_single_sample,
            'model': clf
        })
        
        print(f"max_depth={str(md):>5}, min_samples_leaf={msl}: "
              f"Train={train_acc:.3f}, Val={val_acc:.3f}, Test={test_acc:.3f} | "
              f"Leaves={n_leaves:3d}, MeanSamples={mean_samples_leaf:.2f}, SingleSampleFrac={frac_single_sample:.3f}")

# Model selection based on validation accuracy
best_val_acc = max(r['val_acc'] for r in tree_results)
candidates = [r for r in tree_results if r['val_acc'] == best_val_acc]
# Sort candidates: prefer lower depth, then larger min_samples_leaf
def sort_key(c):
    d = 999 if c['max_depth'] is None else c['max_depth']
    return (d, -c['min_samples_leaf'])
candidates.sort(key=sort_key)
frozen_res = candidates[0]
frozen_clf = frozen_res['model']
frozen_depth = frozen_res['max_depth']
frozen_msl = frozen_res['min_samples_leaf']

print("\n" + "-"*50)
print(f"FROZEN TREE SETTING (Selected on Val Acc):")
print(f"max_depth = {frozen_depth}, min_samples_leaf = {frozen_msl}")
print(f"Train Acc: {frozen_res['train_acc']:.4f}")
print(f"Val Acc:   {frozen_res['val_acc']:.4f}")
print(f"Test Acc:  {frozen_res['test_acc']:.4f}")
print(f"Number of Leaves: {frozen_res['n_leaves']}")
print(f"Mean Samples per Leaf: {frozen_res['mean_samples_leaf']:.2f}")
print(f"Fraction of Singleton Leaves: {frozen_res['frac_single_sample']:.4f}")
print("-"*50)

# Q1(b) & Q1(c): Bagging from Scratch
# B in {1, 5, 10, 25, 50, 100, 200}

print("\n" + "="*50)
print("Q1(b) & Q1(c) Bagging from Scratch")
print("="*50)

B_list = [1, 5, 10, 25, 50, 100, 200]
max_B = max(B_list)
N_train = len(X_train)

# Theoretical expectation of distinct examples
prob_selected_exact = 1.0 - (1.0 - 1.0 / N_train)**N_train
expected_distinct_exact = N_train * prob_selected_exact
prob_selected_asymp = 1.0 - np.exp(-1.0)
expected_distinct_asymp = N_train * prob_selected_asymp

print(f"N_train = {N_train}")
print(f"Theoretical Expectation of distinct samples in bootstrap:")
print(f"  Exact [N * (1 - (1 - 1/N)^N)]: {expected_distinct_exact:.2f} ({prob_selected_exact*100:.2f}%)")
print(f"  Asymptotic [N * (1 - 1/e)]:    {expected_distinct_asymp:.2f} ({prob_selected_asymp*100:.2f}%)")

# We train up to max_B trees sequentially with bootstrap
# Use a dedicated generator for bootstrap indices and base trees
rng_bag = np.random.default_rng(NUMERIC_SEED + 100)

trees = []
bootstrap_indices_list = []
distinct_samples_list = []

for b in range(max_B):
    # Bootstrap sample with replacement
    boot_idx = rng_bag.choice(N_train, size=N_train, replace=True)
    bootstrap_indices_list.append(boot_idx)
    distinct_samples_list.append(len(np.unique(boot_idx)))
    
    # Train base tree
    # Notice: we can vary tree random_state or fix it, sklearn uses random_state for tie-breaking splits
    tree = DecisionTreeClassifier(
        max_depth=frozen_depth,
        min_samples_leaf=frozen_msl,
        random_state=int(rng_bag.integers(0, 1000000))
    )
    tree.fit(X_train[boot_idx], y_train[boot_idx])
    trees.append(tree)

print(f"Successfully trained {max_B} bootstrap trees.")

# Precompute predictions for all trees on Train, Val, Test, Masked Test
train_preds_all = np.array([t.predict(X_train) for t in trees]) # shape (max_B, N_train)
val_preds_all = np.array([t.predict(X_val) for t in trees])     # shape (max_B, N_val)
test_preds_all = np.array([t.predict(X_test_clean) for t in trees]) # shape (max_B, N_test)
test_masked_preds_all = np.array([t.predict(X_test_masked) for t in trees]) # shape (max_B, N_test)

# Function to compute ensemble majority vote predictions & vote counts
def predict_ensemble(preds_subset, num_classes=n_classes):
    # preds_subset shape: (B, N)
    B_curr, N_samples = preds_subset.shape
    # Count votes for each sample
    vote_counts = np.zeros((N_samples, num_classes), dtype=int)
    for c in range(num_classes):
        vote_counts[:, c] = np.sum(preds_subset == c, axis=0)
    
    # Winning class
    ensemble_preds = np.argmax(vote_counts, axis=1)
    
    # Sorted vote fractions
    sorted_votes = np.sort(vote_counts, axis=1)[:, ::-1] # descending
    winner_frac = sorted_votes[:, 0] / B_curr
    runnerup_frac = sorted_votes[:, 1] / B_curr if num_classes > 1 else np.zeros(N_samples)
    vote_margin = winner_frac - runnerup_frac
    
    return ensemble_preds, vote_counts, vote_margin

# Function to compute Out Of Bag (OOB) predictions
def compute_oob(trees_subset, boot_indices_subset, X_tr, y_tr, num_classes=n_classes):
    B_curr = len(trees_subset)
    N_tr = len(X_tr)
    
    # Track votes only from trees where sample i was OOB
    oob_votes = np.zeros((N_tr, num_classes), dtype=int)
    oob_counts = np.zeros(N_tr, dtype=int)
    
    for b in range(B_curr):
        in_bag = np.zeros(N_tr, dtype=bool)
        in_bag[boot_indices_subset[b]] = True
        oob_mask = ~in_bag
        
        if np.any(oob_mask):
            oob_indices = np.where(oob_mask)[0]
            preds = trees_subset[b].predict(X_tr[oob_indices])
            for idx, p in zip(oob_indices, preds):
                oob_votes[idx, p] += 1
                oob_counts[idx] += 1
                
    # Evaluated only on samples that have at least one OOB prediction
    valid_oob = oob_counts > 0
    if np.sum(valid_oob) == 0:
        return 0.0, 0.0
    
    oob_preds = np.argmax(oob_votes[valid_oob], axis=1)
    oob_acc = np.mean(oob_preds == y_tr[valid_oob])
    coverage = np.mean(valid_oob)
    return oob_acc, coverage

# Function to compute average pairwise disagreement between trees
def compute_pairwise_disagreement(preds_subset):
    # preds_subset shape: (B, N)
    B_curr = preds_subset.shape[0]
    if B_curr < 2:
        return 0.0
    
    disagreements = []
    for i in range(B_curr):
        for j in range(i + 1, B_curr):
            disagreements.append(np.mean(preds_subset[i] != preds_subset[j]))
    return np.mean(disagreements)

# Evaluation across B in B_list
bagging_results = []

for B in B_list:
    sub_trees = trees[:B]
    sub_boots = bootstrap_indices_list[:B]
    
    # Train, Val, Test ensemble predictions
    ens_train_pred, _, _ = predict_ensemble(train_preds_all[:B])
    ens_val_pred, val_votes, val_margin = predict_ensemble(val_preds_all[:B])
    ens_test_pred, _, _ = predict_ensemble(test_preds_all[:B])
    
    train_acc = np.mean(ens_train_pred == y_train)
    val_acc = np.mean(ens_val_pred == y_val)
    test_acc = np.mean(ens_test_pred == y_test)
    
    # Average distinct examples per bootstrap in ensemble
    avg_distinct = np.mean(distinct_samples_list[:B])
    
    # OOB accuracy
    oob_acc, oob_cov = compute_oob(sub_trees, sub_boots, X_train, y_train)
    
    # Pairwise disagreement on validation set
    pairwise_disag = compute_pairwise_disagreement(val_preds_all[:B])
    
    # Mean vote margin on validation set
    mean_margin = np.mean(val_margin)
    
    bagging_results.append({
        'B': B,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'avg_distinct': avg_distinct,
        'oob_acc': oob_acc,
        'oob_cov': oob_cov,
        'pairwise_disag': pairwise_disag,
        'mean_margin': mean_margin
    })
    
    print(f"B={B:3d}: TrainAcc={train_acc:.3f}, ValAcc={val_acc:.3f}, TestAcc={test_acc:.3f} | "
          f"AvgDistinct={avg_distinct:.1f} (Exp: {expected_distinct_exact:.1f}) | "
          f"OOB_Acc={oob_acc:.3f} (Cov:{oob_cov*100:.0f}%), Disagreement={pairwise_disag:.3f}, MeanMargin={mean_margin:.3f}")

# Select final ensemble size using validation set
best_bag_val_acc = max(r['val_acc'] for r in bagging_results)
best_B_candidates = [r for r in bagging_results if r['val_acc'] == best_bag_val_acc]
final_B_res = best_B_candidates[0]
final_B = final_B_res['B']
print(f"\nFinal Selected Ensemble Size B* = {final_B} (Validation Accuracy = {final_B_res['val_acc']:.4f})")

# Plot Q1(c): OOB Accuracy, Disagreement, Mean Vote Margin vs B

Bs = [r['B'] for r in bagging_results]
oobs = [r['oob_acc'] for r in bagging_results]
disags = [r['pairwise_disag'] for r in bagging_results]
margins = [r['mean_margin'] for r in bagging_results]
val_accs = [r['val_acc'] for r in bagging_results]

plt.figure(figsize=(10, 6))
plt.plot(Bs, oobs, marker='o', linewidth=2, label='Out-of-Bag (OOB) Accuracy (Train OOB)', color='#1f77b4')
plt.plot(Bs, val_accs, marker='s', linewidth=2, linestyle='--', label='Validation Set Accuracy', color='#2ca02c')
plt.plot(Bs, disags, marker='^', linewidth=2, label='Avg Pairwise Tree Disagreement (Val)', color='#d62728')
plt.plot(Bs, margins, marker='d', linewidth=2, label='Mean Vote Margin (Val)', color='#9467bd')

plt.axvline(x=final_B, color='gray', linestyle=':', label=f'Selected B* = {final_B}')
plt.xlabel('Ensemble Size B (Number of Trees)', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Bagging Dynamics: OOB Accuracy, Tree Disagreement, and Vote Margin vs B', fontsize=14, fontweight='bold')
plt.xticks(Bs)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='center right')
plt.tight_layout()
q1c_plot_path = os.path.join(fig_dir, "q1c_bagging_metrics_vs_B.png")
plt.savefig(q1c_plot_path, dpi=300)
plt.close()
print(f"Saved Q1(c) metric plot to {q1c_plot_path}")

# Q1(d): Robustness and Error Breakdown

print("\n" + "="*50)
print("Q1(d) Clean vs Masked Robustness & Deep Error Analysis")
print("="*50)

# Evaluate Single Tree on Clean & Masked
tree_clean_pred = frozen_clf.predict(X_test_clean)
tree_clean_acc = np.mean(tree_clean_pred == y_test)
tree_clean_proba = frozen_clf.predict_proba(X_test_clean)
tree_clean_conf = np.mean(np.max(tree_clean_proba, axis=1))

tree_masked_pred = frozen_clf.predict(X_test_masked)
tree_masked_acc = np.mean(tree_masked_pred == y_test)
tree_masked_proba = frozen_clf.predict_proba(X_test_masked)
tree_masked_conf = np.mean(np.max(tree_masked_proba, axis=1))

# Evaluate Bagged Model (B*) on Clean & Masked
final_trees = trees[:final_B]
ens_test_clean_pred, test_clean_votes, _ = predict_ensemble(test_preds_all[:final_B])
bag_clean_acc = np.mean(ens_test_clean_pred == y_test)
bag_clean_conf = np.mean(np.max(test_clean_votes, axis=1) / final_B)

ens_test_masked_pred, test_masked_votes, _ = predict_ensemble(test_masked_preds_all[:final_B])
bag_masked_acc = np.mean(ens_test_masked_pred == y_test)
bag_masked_conf = np.mean(np.max(test_masked_votes, axis=1) / final_B)

print(f"Model Performance Comparison:")
print(f"Single Tree Clean:   Accuracy = {tree_clean_acc:.4f}, Mean Confidence = {tree_clean_conf:.4f}")
print(f"Bagged Model Clean:  Accuracy = {bag_clean_acc:.4f}, Mean Confidence = {bag_clean_conf:.4f}")
print(f"Single Tree Masked:  Accuracy = {tree_masked_acc:.4f}, Mean Confidence = {tree_masked_conf:.4f}")
print(f"Bagged Model Masked: Accuracy = {bag_masked_acc:.4f}, Mean Confidence = {bag_masked_conf:.4f}")

# Find test sample cases on Clean Test set (or Clean/Masked)
# Case 1: Bagging corrects the tree (Tree wrong, Bagging correct)
# Case 2: Both are wrong (Tree wrong, Bagging wrong)
correct_corrected_indices = np.where((tree_clean_pred != y_test) & (ens_test_clean_pred == y_test))[0]
both_wrong_indices = np.where((tree_clean_pred != y_test) & (ens_test_clean_pred != y_test))[0]

print(f"\nNumber of clean test samples where Bagging corrects Tree: {len(correct_corrected_indices)}")
print(f"Number of clean test samples where Both are wrong: {len(both_wrong_indices)}")

# Select two of each
case_corrected = correct_corrected_indices[:2]
case_both_wrong = both_wrong_indices[:2]

# If there aren't enough on clean test, check masked test, but let's see how many on clean test!
print("Corrected Case Indices (test set index):", case_corrected)
print("Both Wrong Case Indices (test set index):", case_both_wrong)

# Generate visualization of the 4 cases
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
case_items = [
    ("Corrects Tree #1", case_corrected[0], True),
    ("Corrects Tree #2", case_corrected[1], True),
    ("Both Wrong #1", case_both_wrong[0], False),
    ("Both Wrong #2", case_both_wrong[1], False),
]

case_details = []

for ax, (label, idx, is_corrected) in zip(axes.flatten(), case_items):
    true_label = y_test[idx]
    t_pred = tree_clean_pred[idx]
    b_pred = ens_test_clean_pred[idx]
    b_votes = test_clean_votes[idx]
    
    # Sort top 3 classes by vote
    top_classes = np.argsort(b_votes)[::-1][:3]
    vote_str = ", ".join([f"C{c}:{b_votes[c]}" for c in top_classes if b_votes[c] > 0])
    
    ax.imshow(X_test_clean[idx].reshape(64, 64), cmap='gray')
    ax.set_title(f"{label} (Sample {idx})\nTrue: Class {true_label}\nTree Pred: Class {t_pred} | Bag Pred: Class {b_pred}\nVotes: [{vote_str}]", fontsize=10)
    ax.axis('off')
    
    case_details.append({
        'type': label,
        'test_idx': idx,
        'true_class': true_label,
        'tree_pred': t_pred,
        'tree_conf': np.max(tree_clean_proba[idx]),
        'bag_pred': b_pred,
        'bag_conf': b_votes[b_pred] / final_B,
        'top_votes': [(c, b_votes[c]) for c in top_classes]
    })

plt.tight_layout()
cases_plot_path = os.path.join(fig_dir, "q1d_case_studies.png")
plt.savefig(cases_plot_path, dpi=300)
plt.close()
print(f"Saved case studies plot to {cases_plot_path}")

print("\n--- Summary of the 4 Selected Faces ---")
for c in case_details:
    print(c)
