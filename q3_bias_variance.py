import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from joblib import Parallel, delayed

SEED = 26012
R = 30  # 30 repetitions
N_CLASSES = 40
N_JOBS = 6  # Use 6 cores in parallel for speed

out_dir = r"D:\ML_Assignment"
fig_dir = os.path.join(out_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

print("="*65)
print("Q3. BIAS AND VARIANCE AS MEASURED QUANTITIES (30 REFITS)")
print("="*65)

# Load dataset and original split
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces(shuffle=False)
X_all = faces.data
y_all = faces.target

split_file = os.path.join(out_dir, "olivetti_split.npz")
split_data = np.load(split_file)
train_indices = split_data['train_indices']
val_indices = split_data['val_indices']
test_indices = split_data['test_indices']

X_train_orig = X_all[train_indices]
y_train_orig = y_all[train_indices]

X_test_clean = X_all[test_indices]
y_test = y_all[test_indices]
N_test = len(y_test)  # 80

# Masked test set (rows 24:40, cols 24:40 zeroed)
X_test_masked = X_test_clean.copy().reshape(-1, 64, 64)
X_test_masked[:, 24:40, 24:40] = 0.0
X_test_masked = X_test_masked.reshape(-1, 4096)

# One-hot target matrix e_i for test set: shape (N_test, N_CLASSES)
e_test = np.zeros((N_test, N_CLASSES), dtype=np.float64)
for i in range(N_test):
    e_test[i, y_test[i]] = 1.0

print(f"Clean Test set: {N_test} samples, Classes: {N_CLASSES}")
print(f"Masked Test set: {N_test} samples (center 16x16 zeroed)")
print(f"Original Train set: {len(y_train_orig)} samples (6 per identity)")

# Worker function for one repetition (Simple & Beginner-Friendly)
def run_single_repetition(r):
    t_start = time.time()
    rep_seed = SEED + r * 100
    rng_rep = np.random.default_rng(rep_seed)
    
    # 1. Resample: within each identity, sample 6 faces with replacement
    resampled_idx = []
    for c in range(N_CLASSES):
        c_train_idx = np.where(y_train_orig == c)[0]
        chosen = rng_rep.choice(c_train_idx, size=6, replace=True)
        resampled_idx.extend(chosen)
    resampled_idx = np.array(resampled_idx)
    
    X_tr = X_train_orig[resampled_idx]
    y_tr = y_train_orig[resampled_idx]
    N_tr = len(y_tr)
    
    results = {}
    
    # --- MODEL 1: Selected Single Tree (max_depth=None, min_samples_leaf=1) ---
    tree_sel = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=rep_seed)
    tree_sel.fit(X_tr, y_tr)
    p_tree_clean = tree_sel.predict_proba(X_test_clean)
    pred_tree_clean = np.argmax(p_tree_clean, axis=1)
    acc_tree_clean = np.mean(pred_tree_clean == y_test)
    pred_tree_masked = tree_sel.predict(X_test_masked)
    acc_tree_masked = np.mean(pred_tree_masked == y_test)
    
    results['p_tree_clean'] = p_tree_clean
    results['acc_tree_clean'] = acc_tree_clean
    results['acc_tree_masked'] = acc_tree_masked
    
    # --- MODEL 2: Final Bagged Model (50 trees, max_depth=None) ---
    N_BAG_TREES = 50
    bag_trees_none = []
    for b in range(N_BAG_TREES):
        boot_idx = rng_rep.choice(N_tr, size=N_tr, replace=True)
        t = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=rep_seed + b)
        t.fit(X_tr[boot_idx], y_tr[boot_idx])
        bag_trees_none.append(t)
        
    # Vote vector clean
    votes_bag_clean = np.zeros((N_test, N_CLASSES), dtype=np.float64)
    for t in bag_trees_none:
        t_preds = t.predict(X_test_clean)
        for i in range(N_test):
            votes_bag_clean[i, t_preds[i]] += 1.0
    p_bag_clean = votes_bag_clean / N_BAG_TREES
    acc_bag_clean = np.mean(np.argmax(p_bag_clean, axis=1) == y_test)
    
    # Masked prediction
    votes_bag_masked = np.zeros((N_test, N_CLASSES), dtype=np.float64)
    for t in bag_trees_none:
        t_preds_m = t.predict(X_test_masked)
        for i in range(N_test):
            votes_bag_masked[i, t_preds_m[i]] += 1.0
    p_bag_masked = votes_bag_masked / N_BAG_TREES
    acc_bag_masked = np.mean(np.argmax(p_bag_masked, axis=1) == y_test)
    
    results['p_bag_clean'] = p_bag_clean
    results['acc_bag_clean'] = acc_bag_clean
    results['acc_bag_masked'] = acc_bag_masked
    
    # --- MODEL 3: Selected Boosting Model (SAMME, depth=4, 50 rounds) ---
    N_BOOST_ROUNDS = 50
    w = np.ones(N_tr, dtype=np.float64) / N_tr
    boost_trees = []
    boost_alphas = []
    random_err = (N_CLASSES - 1.0) / N_CLASSES
    
    for b in range(N_BOOST_ROUNDS):
        t_b = DecisionTreeClassifier(max_depth=4, random_state=rep_seed + b)
        t_b.fit(X_tr, y_tr, sample_weight=w)
        pred_tr = t_b.predict(X_tr)
        err = np.sum(w * (pred_tr != y_tr)) / np.sum(w)
        
        if err >= random_err:
            break
        if err <= 1e-10:
            alpha = np.log((1.0 - 1e-10)/1e-10) + np.log(N_CLASSES - 1.0)
            boost_trees.append(t_b)
            boost_alphas.append(alpha)
            break
            
        alpha = np.log((1.0 - err) / err) + np.log(N_CLASSES - 1.0)
        boost_trees.append(t_b)
        boost_alphas.append(alpha)
        w = w * np.exp(alpha * (pred_tr != y_tr))
        w = w / np.sum(w)
        
    # Normalised weighted vote: q_k(x) = S_k(x) / sum(alpha)
    scores_boost_clean = np.zeros((N_test, N_CLASSES), dtype=np.float64)
    total_alpha = np.sum(boost_alphas)
    for t_b, a in zip(boost_trees, boost_alphas):
        preds_te = t_b.predict(X_test_clean)
        for i in range(N_test):
            scores_boost_clean[i, preds_te[i]] += a
    p_boost_clean = scores_boost_clean / total_alpha if total_alpha > 0 else np.ones((N_test, N_CLASSES))/N_CLASSES
    acc_boost_clean = np.mean(np.argmax(p_boost_clean, axis=1) == y_test)
    
    scores_boost_masked = np.zeros((N_test, N_CLASSES), dtype=np.float64)
    for t_b, a in zip(boost_trees, boost_alphas):
        preds_tem = t_b.predict(X_test_masked)
        for i in range(N_test):
            scores_boost_masked[i, preds_tem[i]] += a
    p_boost_masked = scores_boost_masked / total_alpha if total_alpha > 0 else np.ones((N_test, N_CLASSES))/N_CLASSES
    acc_boost_masked = np.mean(np.argmax(p_boost_masked, axis=1) == y_test)
    
    results['p_boost_clean'] = p_boost_clean
    results['acc_boost_clean'] = acc_boost_clean
    results['acc_boost_masked'] = acc_boost_masked
    
    # --- PART (b) DEPTH EXPERIMENT: Single Trees & 50-Tree Bagging at Depths [2, 4, 8] ---
    depth_res = {}
    for d in [2, 4, 8]:
        # Single tree at depth d
        t_d = DecisionTreeClassifier(max_depth=d, min_samples_leaf=1, random_state=rep_seed)
        t_d.fit(X_tr, y_tr)
        p_td = t_d.predict_proba(X_test_clean)
        acc_td = np.mean(np.argmax(p_td, axis=1) == y_test)
        
        # Leaf occupancy
        n_leaves = t_d.get_n_leaves()
        is_leaf = (t_d.tree_.children_left == -1)
        leaf_samples = t_d.tree_.n_node_samples[is_leaf]
        frac_single = np.mean(leaf_samples == 1)
        
        # 50-tree Bagging at depth d
        bag_d_trees = []
        for b in range(N_BAG_TREES):
            boot_idx = rng_rep.choice(N_tr, size=N_tr, replace=True)
            t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=1, random_state=rep_seed + b)
            t.fit(X_tr[boot_idx], y_tr[boot_idx])
            bag_d_trees.append(t)
            
        votes_d = np.zeros((N_test, N_CLASSES), dtype=np.float64)
        for t in bag_d_trees:
            t_preds = t.predict(X_test_clean)
            for i in range(N_test):
                votes_d[i, t_preds[i]] += 1.0
        p_bd = votes_d / N_BAG_TREES
        acc_bd = np.mean(np.argmax(p_bd, axis=1) == y_test)
        
        # Disagreement between trees on clean test set
        disags = []
        preds_all_trees = np.array([t.predict(X_test_clean) for t in bag_d_trees])  # (50, 80)
        for i in range(min(15, N_BAG_TREES)):
            for j in range(i + 1, min(15, N_BAG_TREES)):
                disags.append(np.mean(preds_all_trees[i] != preds_all_trees[j]))
        mean_disag = np.mean(disags) if len(disags) > 0 else 0.0
        
        depth_res[d] = {
            'p_tree': p_td, 'acc_tree': acc_td,
            'p_bag': p_bd, 'acc_bag': acc_bd,
            'n_leaves': n_leaves, 'frac_single': frac_single,
            'mean_disag': mean_disag
        }
        
    results['depth_res'] = depth_res
    print(f"Repetition {r+1:2d}/30 finished in {time.time()-t_start:.1f}s")
    return results

# Execute 30 Repetitions in Parallel

print(f"\nLaunching {R} parallel repetitions across {N_JOBS} CPU cores...")
t_global_start = time.time()
rep_results = Parallel(n_jobs=N_JOBS)(delayed(run_single_repetition)(r) for r in range(R))
print(f"\nAll {R} repetitions finished in {time.time()-t_global_start:.1f}s!\n")

# Bias and Variance Computation Helper
# Bias^2 = (1/N) * sum_i || p_bar(x_i) - e_i ||^2
# Variance = (1 / (N * R)) * sum_{i, r} || p_r(x_i) - p_bar(x_i) ||^2

def compute_bv(p_reps):
    # p_reps: shape (R, N_test, N_CLASSES)
    p_bar = np.mean(p_reps, axis=0) # (N_test, N_CLASSES)
    
    # Bias^2
    diff_bias = p_bar - e_test
    bias_sq = np.mean(np.sum(diff_bias**2, axis=1))
    
    # Variance
    diff_var = p_reps - p_bar # (R, N_test, N_CLASSES)
    variance = np.mean(np.sum(diff_var**2, axis=2))
    
    return bias_sq, variance

# Q3(a): Results for Selected Models

p_tree_reps = np.array([res['p_tree_clean'] for res in rep_results])
p_bag_reps = np.array([res['p_bag_clean'] for res in rep_results])
p_boost_reps = np.array([res['p_boost_clean'] for res in rep_results])

bias_sq_tree, var_tree = compute_bv(p_tree_reps)
bias_sq_bag, var_bag = compute_bv(p_bag_reps)
bias_sq_boost, var_boost = compute_bv(p_boost_reps)

accs_tree = [res['acc_tree_clean'] for res in rep_results]
accs_bag = [res['acc_bag_clean'] for res in rep_results]
accs_boost = [res['acc_boost_clean'] for res in rep_results]

print("="*60)
print("Q3(a) MEASURED BIAS SQUARED, VARIANCE, AND ACCURACY")
print("="*60)
print(f"1. Single Tree (Unpruned None):")
print(f"   Bias^2   = {bias_sq_tree:.4f}")
print(f"   Variance = {var_tree:.4f}")
print(f"   Test Acc = {np.mean(accs_tree)*100:.2f}% ± {np.std(accs_tree)*100:.2f}%")

print(f"\n2. Final Bagged Model (Unpruned Trees):")
print(f"   Bias^2   = {bias_sq_bag:.4f}")
print(f"   Variance = {var_bag:.4f}")
print(f"   Test Acc = {np.mean(accs_bag)*100:.2f}% ± {np.std(accs_bag)*100:.2f}%")

print(f"\n3. Selected Boosting Model (SAMME Depth 4):")
print(f"   Bias^2   = {bias_sq_boost:.4f}")
print(f"   Variance = {var_boost:.4f}")
print(f"   Test Acc = {np.mean(accs_boost)*100:.2f}% ± {np.std(accs_boost)*100:.2f}%")

# Q3(b): Depth Experiment Analysis (Depths 2, 4, 8)

print("\n" + "="*60)
print("Q3(b) DEPTH EXPERIMENT: SINGLE TREES VS BAGGED ENSEMBLES")
print("="*60)

depths = [2, 4, 8]
depth_stats = []

for d in depths:
    p_t_d = np.array([res['depth_res'][d]['p_tree'] for res in rep_results])
    p_b_d = np.array([res['depth_res'][d]['p_bag'] for res in rep_results])
    
    b_sq_t, v_t = compute_bv(p_t_d)
    b_sq_b, v_b = compute_bv(p_b_d)
    
    acc_t = [res['depth_res'][d]['acc_tree'] for res in rep_results]
    acc_b = [res['depth_res'][d]['acc_bag'] for res in rep_results]
    
    var_removed = v_t - v_b
    var_reduc_pct = (var_removed / v_t) * 100.0 if v_t > 0 else 0.0
    acc_gain = np.mean(acc_b) - np.mean(acc_t)
    
    leaves_mean = np.mean([res['depth_res'][d]['n_leaves'] for res in rep_results])
    single_leaf_pct = np.mean([res['depth_res'][d]['frac_single'] for res in rep_results]) * 100.0
    disag_mean = np.mean([res['depth_res'][d]['mean_disag'] for res in rep_results])
    
    depth_stats.append({
        'depth': d,
        'b_sq_t': b_sq_t, 'v_t': v_t, 'acc_t_mean': np.mean(acc_t),
        'b_sq_b': b_sq_b, 'v_b': v_b, 'acc_b_mean': np.mean(acc_b),
        'var_removed': var_removed, 'var_reduc_pct': var_reduc_pct,
        'acc_gain': acc_gain,
        'leaves': leaves_mean, 'single_leaf_pct': single_leaf_pct,
        'disag': disag_mean
    })
    
    print(f"Depth {d}:")
    print(f"   Single Tree : Bias^2 = {b_sq_t:.4f}, Variance = {v_t:.4f}, Acc = {np.mean(acc_t)*100:.2f}% | Leaves = {leaves_mean:.1f}")
    print(f"   Bagging     : Bias^2 = {b_sq_b:.4f}, Variance = {v_b:.4f}, Acc = {np.mean(acc_b)*100:.2f}%")
    print(f"   Variance Removed by Bagging: {var_removed:.4f} ({var_reduc_pct:.1f}% reduction)")
    print(f"   Accuracy Gain: +{acc_gain*100:.2f}%")
    print(f"   Tree Disagreement: {disag_mean:.3f}, Singleton Leaf %: {single_leaf_pct:.1f}%\n")

# Q3(c): Is the deepest tree necessarily the largest improvement?

print("="*60)
print("Q3(c) DEEPEST TREE ACCURACY IMPROVEMENT ANALYSIS")
print("="*60)
for ds in depth_stats:
    print(f"Depth {ds['depth']}: Acc Gain = +{ds['acc_gain']*100:.2f}% (Single: {ds['acc_t_mean']*100:.2f}% -> Bag: {ds['acc_b_mean']*100:.2f}%)")
    print(f"         Var Removed = {ds['var_removed']:.4f}, Disagreement = {ds['disag']:.3f}, Leaves = {ds['leaves']:.1f}")

# Q3(d): Final Comprehensive Comparison Table

print("\n" + "="*60)
print("Q3(d) FINAL COMPREHENSIVE COMPARISON TABLE")
print("="*60)

accs_tree_m = [res['acc_tree_masked'] for res in rep_results]
accs_bag_m = [res['acc_bag_masked'] for res in rep_results]
accs_boost_m = [res['acc_boost_masked'] for res in rep_results]

mean_tree_clean = np.mean(accs_tree)
mean_bag_clean = np.mean(accs_bag)
mean_boost_clean = np.mean(accs_boost)

mean_tree_mask = np.mean(accs_tree_m)
mean_bag_mask = np.mean(accs_bag_m)
mean_boost_mask = np.mean(accs_boost_m)

drop_tree = (mean_tree_clean - mean_tree_mask) * 100.0
drop_bag = (mean_bag_clean - mean_bag_mask) * 100.0
drop_boost = (mean_boost_clean - mean_boost_mask) * 100.0

print(f"{'Model':<25} | {'Clean Acc':<12} | {'Masked Acc':<12} | {'Occlusion Drop':<15} | {'Bias^2':<10} | {'Variance':<10}")
print("-" * 95)
print(f"{'Selected Single Tree':<25} | {mean_tree_clean*100:>10.2f}% | {mean_tree_mask*100:>10.2f}% | {drop_tree:>13.2f}% | {bias_sq_tree:>10.4f} | {var_tree:>10.4f}")
print(f"{'Final Bagged Model':<25} | {mean_bag_clean*100:>10.2f}% | {mean_bag_mask*100:>10.2f}% | {drop_bag:>13.2f}% | {bias_sq_bag:>10.4f} | {var_bag:>10.4f}")
print(f"{'Selected Boosting Model':<25} | {mean_boost_clean*100:>10.2f}% | {mean_boost_mask*100:>10.2f}% | {drop_boost:>13.2f}% | {bias_sq_boost:>10.4f} | {var_boost:>10.4f}")

# Generate Visualization Plot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Bias^2 and Variance across Models
models = ['Single Tree', 'Bagging', 'Boosting']
b_vals = [bias_sq_tree, bias_sq_bag, bias_sq_boost]
v_vals = [var_tree, var_bag, var_boost]

x = np.arange(len(models))
w = 0.35

ax1.bar(x - w/2, b_vals, w, label='Bias^2', color='#1f77b4')
ax1.bar(x + w/2, v_vals, w, label='Variance', color='#ff7f0e')
ax1.set_title('Measured Bias^2 and Variance (30 Refits)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylabel('Metric Value', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Variance Removed by Bagging Across Depths
d_labels = [f"Depth {d['depth']}" for d in depth_stats]
v_single = [d['v_t'] for d in depth_stats]
v_bag = [d['v_b'] for d in depth_stats]

x_d = np.arange(len(d_labels))
ax2.plot(x_d, v_single, marker='o', linewidth=2.5, label='Single Tree Variance', color='#d62728')
ax2.plot(x_d, v_bag, marker='s', linewidth=2.5, label='Bagging Ensemble Variance', color='#2ca02c')
ax2.set_title('Variance Reduction Across Depths', fontsize=12, fontweight='bold')
ax2.set_xticks(x_d)
ax2.set_xticklabels(d_labels, fontsize=11)
ax2.set_ylabel('Variance', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
q3_plot_path = os.path.join(fig_dir, "q3_bias_variance_comparison.png")
plt.savefig(q3_plot_path, dpi=300)
plt.close()
print(f"\nSaved Q3 plot to {q3_plot_path}")
