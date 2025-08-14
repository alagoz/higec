# HiGEC: Hierarchy Generation and Extended Classification Framework
HiGEC is a Python framework for performing hierarchical classification with automated hierarchy generation, flexible exploitation strategies, and integration with modern classifiers.

HiGeC is evaluated on 100 multi-class datasets, demonstrating significant improvements over traditional flat classification (FC) approaches, particularly with advanced classifiers like XGBoost, RF, ETC, and LGBM.

![fig_cdd_hge_vs_FC_vs_OVA_f1]()

| Dataset | RF | XGB  |
| --- | --- | --- |
| dataset-1 | <span style=\"color:red;\">0.92</span> | <span style=\"color:blue;\">0.85±0.01</span>  |
| dataset-2 | 0.88±0.02 | <span style=\"color:green;\">0.90</span>  |


🔧 Installation
```
git clone https://github.com/your-username/higec.git
cd higec
pip install -r requirements.txt
```
Dependencies:
numpy
scikit-learn
xgboost
scipy
matplotlib


📊 What This Project Does
HiGEC provides:
- Automated Hierarchy Generation from flat-labeled datasets
- Probabilistic and hybrid Hierarchy Exploitation strategies
- Support for any multi-class base classifier
- Benchmark-ready structure using OpenML datasets


🚀 Quick Start

Run the Example:
```
python run_higec_example.py
```
What It Does:
1. Downloads the Glass dataset from OpenML
2. Performs flat classification using XGBoost
3. Automatically constructs a class hierarchy using TSD (Task Similarity Distance)
4. Trains a hierarchical classifier using the LCL+ scheme
5. Compares F1-score of flat vs hierarchical classification


🧱 Core Components

| Component | Description |
| --- | --- |
| HiGen | Constructs hierarchy using representative- or classifier-based distances |
| hier_binary_tree | Hierarchical classifier wrapper for training/prediction |
| utils.py | Data loader, metric scorer, plotting, label checks |


🧪 Customization

You can change the following parameters in run_higec_example.py:
```
did_ = 41            # Dataset ID (from OpenML)
hc_type = 'lcl+'     # HC strategy: 'lcl+', 'lcpn', 'lcn+f', etc.
diss_type = 'jsd'    # Dissimilarity type: 'tsd', 'jsd', 'ccm', 'cmd'
build_type = 'hdc'   # Hierarchy build type: 'hac' or 'hdc'
eval_metric = 'f1'   # Metric: 'f1', 'accuracy', 'auc'
```
You can also replace XGBClassifier() with any sklearn-compatible classifier.


📈 Example Output
```
Extended linkage table for levels:
level_id:0, subsets:[[3, 4], [0, 1, 2, 5]], branch_id:[8, 9]
level_id:1, subsets:[[1, 2], [0, 5], [4], [3]], branch_id:[7, 6, 4, 3]

Performance Comparison:
- Flat Classification (f1): 0.6470
- Hierarchical lcl+f (f1): 0.7377
```
Generated Hierarchy:

![generated_hier](https://github.com/user-attachments/assets/fa009a38-bb18-4355-9249-2e9d4264da18)

📂 Project Structure

├── run_higec_example.py     # Main script

├── HiGen.py                 # Hierarchy generation module

├── HiCl.py                  # Hierarchical classifier logic

├── utils.py                 # Data loading and utility functions

├── README.md

├── requirements.txt

