import numpy as np
from utils import loadOpenMLdata, class_labels_sanity_check, plot_dendrogram, get_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # pip install xgboost
from time import time
from HiGen import HiGen
from HiCl import hier_binary_tree
import copy

# -----------------------------
# PARAMETERS
# -----------------------------
did_ = 41               # OpenML Dataset ID
dname_ = 'Glass'        # Dataset name (optional)
test_size = 0.2         # Train/Test split ratio
hc_type = 'lcl+'        # Hierarchical classification scheme
rseed = 0               # Random seed
eval_metric = 'f1'      # Evaluation metric

# -----------------------------
# LOAD DATA
# -----------------------------
(X_num, X_cat), y = loadOpenMLdata(
    dset_id=did_,
    dset_name=dname_,
    verbose=0
)
X = np.c_[X_num, X_cat]     # Combine numeric and categorical features
y = np.array(y)

# Split into training and testing sets
x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rseed, stratify=y)

# Sanity check for class labels
classes = class_labels_sanity_check(y_tr, y_te)
n_class = len(classes)

# -----------------------------
# FLAT CLASSIFICATION (FC)
# -----------------------------
clf_fc = copy.deepcopy(XGBClassifier())
clf_fc.fit(x_tr, y_tr)
y_pred = clf_fc.predict(x_te)
y_pred_proba_fc = clf_fc.predict_proba(x_te)

score_fc = get_score(y_te, y_pred=y_pred, pred_proba=y_pred_proba_fc, eval_metric=eval_metric)

# -----------------------------
# HIERARCHY GENERATION (HG)
# -----------------------------
clf_cbd = copy.deepcopy(XGBClassifier())

t0 = time()
model_hg = HiGen(
    X, y,
    dissimilarity_type='tsd',
    dissimilarity_output_type='diss_mat',
    metric_cc='euclidean',
    precomputed_pred=False,
    y_pred_proba=None,
    y_pred=None,
    conf_mat=None,
    clf_cbd=clf_cbd,
    cbd_val_size=0.25,
    build_type='hdc',               # Hierarchical Divisive Clustering
    dist_hac='complete',
    split_fun='kmed'
)
dur_hg = time() - t0

Z, PNs = model_hg.build_hierarchy()
plot_dendrogram(Z, close_all=True)

# -----------------------------
# HIERARCHICAL CLASSIFICATION (HC)
# -----------------------------
tree = hier_binary_tree(
    pnodes=PNs,
    y_train=y_tr,
    y_test=y_te,
    link_mat=Z,
    pred_proba_fc=y_pred_proba_fc  # for HE+F variants, pass FC predictions
)
# Display Exteded Linkage Table
tree.display_extended_linkage(hc_type=hc_type)

clf_base = copy.deepcopy(XGBClassifier())

# Train HC
t0 = time()
tree.fit(clf_base, x_tr, hc_type=hc_type, multi_process=True)
dur_hc_fit = time() - t0

# Predict with HC
t0 = time()
tree.predict_proba(x_te, hc_type=hc_type)
dur_hc_pred = time() - t0

# Evaluate HC
score_hc = tree.score(eval_metric=eval_metric)

# -----------------------------
# FINAL RESULTS
# -----------------------------
print(f'\nPerformance Comparison:')
print(f'- Flat Classification ({eval_metric}): {score_fc:.4f}')
print(f'- Hierarchical {hc_type} ({eval_metric}): {score_hc:.4f}')
