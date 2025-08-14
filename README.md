# ![HiGEC Logo](https://github.com/user-attachments/assets/96e78795-541b-41a1-a7bb-a945b65411fa) HiGEC  
**Hierarchy Generation and Extended Classification Framework**  

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![OpenML](https://img.shields.io/badge/OpenML-datasets-orange.svg)](https://www.openml.org/)  

HiGEC is a Python framework for improving multi-class classification via **automated hierarchy generation** and **flexible exploitation strategies**. It supports hybrid approaches integrating hierarchical and flat classifier outputs, delivering robust F1-score improvements on high-dimensional tabular datasets.  

---

<details>
<summary>🎬 Demo GIF</summary>

![HiGEC Demo](https://github.com/user-attachments/assets/96e78795-541b-41a1-a7bb-a945b65411fa)  
*Quick visualization of hierarchy generation and exploitation on a sample dataset.*
</details>

---

<details>
<summary>🔧 Installation</summary>

```bash
git clone https://github.com/alagoz/higec.git
cd higec
pip install -r requirements.txt
```

# HiGEC: Hierarchy Generation and Extended Classification Framework
HiGEC is a Python framework for performing hierarchical classification with automated hierarchy generation, flexible exploitation strategies, and integration with modern classifiers.

🔧 Installation
```
git clone https://github.com/your-username/higec.git
cd higec
pip install -r requirements.txt
```
Dependencies:
numpy
scipy
matplotlib
scikit-learn
scikit-learn-extra
proglearn
xgboost
lightgbm

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
1. Downloads a dataset from OpenML by providing a dataset id
2. Performs flat classification using selected classifier
3. Automatically constructs a class hierarchy
4. Trains a hierarchical classifier
5. Compares F1-score of flat vs hierarchical classification


🧱 Core Components

| Component | Description |
| --- | --- |
| HG.py | Hierarchy Generation: Constructs hierarchy using representative- or classifier-based distances |
| HE.py | Hierarcy Exploitation: Hierarchical classifier wrapper for training/prediction |
| diss_mat_embedding.py | Embeds dissimilarity matrix into vector space |
| hdc.py | Hierarchical Divisive Clustering: Performs top-down hierarchy construction |
| jsd.py | Performs Jenssen-Shannon Distance  |
| tsd.py | Performs Task Similarity Distance |
| run_higec_example.py | runs demo exmaple |
| utils.py | Data loader, metric scorer, plotting, label checks |


🧪 Customization

You can change the following parameters in run_higec_example.py:
```
DID = 46264          # Dataset ID (from OpenML)
HiGEC = 'CCM[HAC|COMPLETE]-LCPN[ETC]+F[XGB]'     # DissType[BuildType|BuildFun]-HE[ClfBase]+F[ClfPF]
CLF_NAME_FC = 'RF'
```
Four Classifiers RF, XGB, ETC, and LGB are defined in demo. 


📈 Example Output
```
Extended linkage table for LCPN nodes:
node_id:0, node_type:parent, subsets:[[0], [1, 2, 3, 4]], branch_ids:[0, 7], parent_id:None, left_id:None, right_id:1
node_id:1, node_type:parent, subsets:[[3, 4], [1, 2]], branch_ids:[5, 6], parent_id:0, left_id:3, right_id:2

Performance Comparison:
- Flat Classification (RF) (f1): 0.3517 in 0.4309 seconds
- HiGEC: CCM[HAC|COMPLETE]-LCPN[ETC]+F[XGB] (f1): 0.3700 in 1.1853 seconds
```
Generated Hierarchy:  
![example_hierarchy](https://github.com/user-attachments/assets/96e78795-541b-41a1-a7bb-a945b65411fa)


Results<br>
HiGeC is evaluated on 100 multi-class datasets, demonstrating significant improvements over traditional flat classification (FC) approaches, particularly with advanced classifiers like XGBoost, RF, ETC, and LGBM.

MCM comparing the mean AUC of selected HiGEC schemes and top FC including XGB, RF, ETC, and LGB base classifiers.
<img width="1476" height="387" alt="fig_mcm_higec_vs_fc" src="https://github.com/user-attachments/assets/614581db-e193-44dc-a5d2-998db14887b5" />

MCCV mean F1 scores and std values for selected FC and HiGEC schemes. Best mean score for each dataset is highlighted. (CSV file is also shared in the repository)
![table](https://github.com/user-attachments/assets/7e8000ef-de32-4aa2-87a6-76da536a9d26)

