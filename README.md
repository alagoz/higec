# ![HiGEC Logo](https://github.com/user-attachments/assets/a415cb44-9bb8-4a8b-b638-624c3ea02667)

**HiGEC**  
**Hierarchy Generation and Extended Classification Framework**  

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![OpenML](https://img.shields.io/badge/OpenML-datasets-orange)](https://www.openml.org)  

HiGEC is a Python framework for enhancing multi-class classification through **automated hierarchy generation (HG)** and **flexible hierarchy exploitation (HE)** strategies. It supports hybrid approaches that integrate hierarchical and flat classifier outputs.

---
<details>
<summary>🔧 Installation</summary>

```bash
git clone https://github.com/alagoz/higec.git
cd higec
pip install -r requirements.txt
```

**Dependencies:**  
`numpy` `scipy` `matplotlib` `scikit-learn` `scikit-learn-extra` `proglearn` `xgboost` `lightgbm`
</details>

<details> <summary>⚡ Key Features</summary>
- � **Automatic hierarchy generation** from flat class labels  
- 🧩 **Hybrid HE+F classification strategies**  
- 🖇️ Support for **any scikit-learn compatible classifier**  
- 📊 **Benchmark-ready** with OpenML integration  
- 🌳 **Visualization tools** for hierarchy inspection


Dependencies: numpy, scipy, matplotlib, scikit-learn, scikit-learn-extra, proglearn, xgboost, lightgbm

</details>
<details> <summary>⚡ Features</summary>

Automatic Hierarchy Generation from flat-labeled datasets

Advanced Hierarchy Exploitation (HE+) and Hybrid HE+F approaches

Support for any multi-class base classifier: RF, XGB, ETC, LGB, etc.

Benchmark-ready with OpenML datasets

Visualizations: dendrograms & extended linkage tables

</details>
<details> <summary>🚀 Quick Start</summary>

Run the example:

python run_higec_example.py


Steps performed:

Download dataset from OpenML

Flat classification with a base classifier

Construct hierarchical class structure

Train hierarchical classifier

Compare F1-score (flat vs hierarchical)

</details>
<details> <summary>🛠 Core Components</summary>
Component	Purpose
HG.py	Hierarchy Generation (distance- or classifier-based)
HE.py	Hierarchy Exploitation (training/prediction wrapper)
diss_mat_embedding.py	Embed dissimilarity matrices
hdc.py	Hierarchical Divisive Clustering
jsd.py	Jensen-Shannon Distance computation
tsd.py	Task Similarity Distance computation
utils.py	Data loading, scoring, plotting, label checks
run_higec_example.py	Demo script with configurable HiGEC settings
</details>
<details> <summary>🧪 Customization</summary>

Adjust parameters in run_higec_example.py:

DID = 46264                       # OpenML dataset ID
HiGEC = 'CCM[HAC|COMPLETE]-LCPN[ETC]+F[XGB]'  # Hierarchy + Exploitation scheme
CLF_NAME_FC = 'RF'                # Flat classifier


Available classifiers: RF, XGB, ETC, LGB.

</details>
<details> <summary>📈 Example Output</summary>

Extended Linkage Table:

node_id:0, node_type:parent, subsets:[[0], [1,2,3,4]], branch_ids:[0,7], parent_id:None
node_id:1, node_type:parent, subsets:[[3,4],[1,2]], branch_ids:[5,6], parent_id:0


Performance Comparison:

- Flat Classification (RF) (f1): 0.3517 in 0.4309 seconds
- HiGEC: CCM[HAC|COMPLETE]-LCPN[ETC]+F[XGB] (f1): 0.3700 in 1.1853 seconds


Generated Hierarchy:
![example_hierarchy](https://github.com/user-attachments/assets/96e78795-541b-41a1-a7bb-a945b65411fa)

</details>
<details> <summary>📊 Benchmark Results</summary>

HiGEC was evaluated on 100 multi-class tabular datasets, showing consistent F1-score gains, particularly with hybrid HE+F configurations.

Mean F1 Comparison (HiGEC vs FC):
<img width="1476" height="387" alt="fig_mcm_higec_vs_fc" src="https://github.com/user-attachments/assets/614581db-e193-44dc-a5d2-998db14887b5" />

Mean F1 Scores & Std Values:
![table](https://github.com/user-attachments/assets/7e8000ef-de32-4aa2-87a6-76da536a9d26)

<details> <summary>📖 References</summary>

For more details on methodology, datasets, and evaluations, see the HiGEC GitHub repository.

</details> ```
