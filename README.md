# HiGEC: Hierarchy Generation and Extended Classification Framework
HiGEC is a Python framework for performing hierarchical classification with automated hierarchy generation, flexible exploitation strategies, and integration with modern classifiers.

HiGeC is evaluated on 115 multi-class datasets, demonstrating significant improvements over traditional flat classification (FC) and one-vs-all (OVA) approaches, particularly with advanced classifiers like XGBoost.

![fig_cdd_hge_vs_FC_vs_OVA_f1](https://github.com/user-attachments/assets/7853f94c-3064-49af-a1fb-7693d9c09928)


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
eval_metric = 'f1'   # Metric: 'f1', 'accuracy', 'auc'
```
You can also replace XGBClassifier() with any sklearn-compatible classifier.


📈 Example Output
```
Extended linkage table for levels:
level_id:0, subsets:[[1, 5, 0], [2, 3, 4]], branch_id:[8, 9]
level_id:1, subsets:[[3, 4], [1, 5], [2], [0]], branch_id:[7, 6, 2, 0]
level_id:2, subsets:[[5], [4], [3], [1]], branch_id:[5, 4, 3, 1]

Performance Comparison:
Flat Classification (f1): 0.6470- 
Hierarchical lcl+ (f1): 0.6595
```
Generated Hierarchy:

![Figure_1](https://github.com/user-attachments/assets/6dcfc0af-169a-4fe6-9a7b-5787a3472128)

📂 Project Structure

├── run_higec_example.py     # Main script

├── HiGen.py                 # Hierarchy generation module

├── HiCl.py                  # Hierarchical classifier logic

├── utils.py                 # Data loading and utility functions

├── README.md

├── requirements.txt
