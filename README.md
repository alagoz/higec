# HiGEC: Hierarchy Generation and Extended Classification Framework
HiGEC is a Python framework for performing hierarchical classification with automated hierarchy generation, flexible exploitation strategies, and integration with modern classifiers.

HiGeC is evaluated on 100 multi-class datasets, demonstrating significant improvements over traditional flat classification (FC) approaches, particularly with advanced classifiers like XGBoost, RF, ETC, and LGBM.

![fig_cdd_hge_vs_FC_vs_OVA_f1]()

<div style="overflow-x:auto;">

<table>
  <thead>
    <tr style="background-color:#f2f2f2;">
      <th>name</th>
      <th>RF</th>
      <th>XGB</th>
      <th>ETC</th>
      <th>LGB</th>
      <th>LCN[XGB]+</th>
      <th>LCPN[ETC]+F[XGB]</th>
      <th>LCPN[RF]+F[XGB]</th>
      <th>LCPN[XGB]+F[XGB]</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color:#ffffff;">
      <td>air-quality-and-pollution</td>
      <td>0.9291±0.0089</td>
      <td>0.9284±0.0116</td>
      <td>0.9247±0.0110</td>
      <td>0.9290±0.0105</td>
      <td>0.9271±0.0119</td>
      <td>0.9291±0.0114</td>
      <td>0.9289±0.0112</td>
      <td>0.9290±0.0101</td>
    </tr>
    <tr style="background-color:#f9f9f9;">
      <td>alizadeh-2000-v2</td>
      <td>0.9485±0.0630</td>
      <td>0.8594±0.1081</td>
      <td>0.9743±0.0515</td>
      <td>0.9440±0.0668</td>
      <td>0.8841±0.1039</td>
      <td>0.8809±0.1219</td>
      <td>0.8681±0.1153</td>
      <td>0.9357±0.0643</td>
    </tr>
    <tr style="background-color:#ffffff;">
      <td>amazon-commerce-reviews</td>
      <td>0.6670±0.0281</td>
      <td>0.7090±0.0253</td>
      <td>0.7273±0.0098</td>
      <td>0.7077±0.0294</td>
      <td>0.5945±0.0203</td>
      <td>0.7402±0.0221</td>
      <td>0.7257±0.0262</td>
      <td>0.5890±0.0157</td>
    </tr>
    <tr style="background-color:#f9f9f9;">
      <td>analcatdata-halloffame</td>
      <td>0.6588±0.0652</td>
      <td>0.6516±0.0563</td>
      <td>0.6663±0.0507</td>
      <td>0.6490±0.0648</td>
      <td>0.6565±0.0733</td>
      <td>0.6530±0.0621</td>
      <td>0.6532±0.0624</td>
      <td>0.6474±0.0738</td>
    </tr>
    <!-- Continue alternating #ffffff / #f9f9f9 for all remaining rows -->
  </tbody>
</table>

</div>



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

