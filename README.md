# HiGEC: Hierarchy Generation and Extended Classification Framework
HiGEC is a Python framework for performing hierarchical classification with automated hierarchy generation, flexible exploitation strategies, and integration with modern classifiers.

HiGeC is evaluated on 100 multi-class datasets, demonstrating significant improvements over traditional flat classification (FC) approaches, particularly with advanced classifiers like XGBoost, RF, ETC, and LGBM.

![fig_cdd_hge_vs_FC_vs_OVA_f1]()

<div style="overflow-x:auto;">

| name                      | RF            | XGB           | ETC           | LGB           | LCN[XGB]+      | LCPN[ETC]+F[XGB] | LCPN[RF]+F[XGB] | LCPN[XGB]+F[XGB] |
|----------------------------|---------------|---------------|---------------|---------------|----------------|-----------------|-----------------|-----------------|
| air-quality-and-pollution | 0.9291±0.0089 | 0.9284±0.0116 | 0.9247±0.0110 | 0.9290±0.0105 | 0.9271±0.0119 | 0.9291±0.0114   | 0.9289±0.0112   | 0.9290±0.0101   |
| alizadeh-2000-v2          | 0.9485±0.0630 | 0.8594±0.1081 | 0.9743±0.0515 | 0.9440±0.0668 | 0.8841±0.1039 | 0.8809±0.1219   | 0.8681±0.1153   | 0.9357±0.0643   |
| amazon-commerce-reviews    | 0.6670±0.0281 | 0.7090±0.0253 | 0.7273±0.0098 | 0.7077±0.0294 | 0.5945±0.0203 | 0.7402±0.0221   | 0.7257±0.0262   | 0.5890±0.0157   |
| analcatdata-halloffame     | 0.6588±0.0652 | 0.6516±0.0563 | 0.6663±0.0507 | 0.6490±0.0648 | 0.6565±0.0733 | 0.6530±0.0621   | 0.6532±0.0624   | 0.6474±0.0738   |
| analcatdata-happiness      | 0.3204±0.1043 | 0.3990±0.0990 | 0.2605±0.1078 | 0.5259±0.1395 | 0.5285±0.0643 | 0.4019±0.1092   | 0.4019±0.1092   | 0.3601±0.0962   |
| anneal                     | 0.7733±0.0739 | 0.7976±0.0746 | 0.7814±0.0627 | 0.7722±0.0749 | 0.7690±0.0866 | 0.7995±0.0811   | 0.8021±0.0887   | 0.7871±0.0717   |
| arrythmia                  | 0.5761±0.0438 | 0.5924±0.0744 | 0.5257±0.0488 | 0.5946±0.0609 | 0.6397±0.0380 | 0.6305±0.0654   | 0.6281±0.0664   | 0.6267±0.0755   |
| artificial-characters       | 0.9102±0.0058 | 0.8998±0.0078 | 0.9095±0.0065 | 0.9119±0.0081 | 0.8917±0.0055 | 0.9199±0.0060   | 0.9139±0.0067   | 0.9102±0.0069   |
| auto-ml-selector           | 0.3392±0.0961 | 0.3710±0.1210 | 0.2765±0.0984 | 0.3560±0.1240 | 0.3635±0.1234 | 0.3682±0.1178   | 0.3782±0.1265   | 0.3922±0.1208   |
| auto-univ-au4-2500         | 0.4071±0.0133 | 0.4562±0.0118 | 0.3850±0.0127 | 0.4627±0.0145 | 0.4670±0.0118 | 0.4560±0.0119   | 0.4560±0.0119   | 0.4071±0.0133   |
| ...                        | ...           | ...           | ...           | ...           | ...            | ...             | ...             | ...             |
| led24                     | 0.7198±0.0114 | 0.6880±0.0185 | 0.7182±0.0130 | 0.6972±0.0196 | 0.6946±0.0184 | 0.6968±0.0155   | 0.6969±0.0149   | 0.7105±0.0122   |
| led-7digit                | 0.6838±0.0302 | 0.6875±0.0398 | 0.6842±0.0377 | 0.6917±0.0420 | 0.6995±0.0331 | 0.6866±0.0457   | 0.6858±0.0352   | 0.7006±0.0449   |
| leukemia                  | 0.7361±0.1254 | 0.8904±0.1465 | 0.8661±0.0404 | 0.9422±0.0632 | 0.8894±0.1482 | 0.9156±0.1121   | 0.9395±0.0634   | 0.8799±0.1330   |
| lung                      | 0.7055±0.0890 | 0.7351±0.1063 | 0.7230±0.0884 | 0.8637±0.1030 | 0.8545±0.0653 | 0.7760±0.1018   | 0.7532±0.1029   | 0.7181±0.0490   |
| lymphoma-burkitt          | 0.7961±0.0830 | 0.8254±0.0677 | 0.7631±0.0627 | 0.7967±0.0499 | 0.7636±0.0822 | 0.8255±0.0705   | 0.8255±0.0705   | 0.7995±0.0957   |
| mabbob-ela-as-2d          | 0.3216±0.0474 | 0.3783±0.0332 | 0.3099±0.0553 | 0.3652±0.0377 | 0.3770±0.0306 | 0.3717±0.0275   | 0.3841±0.0305   | 0.3291±0.0511   |
| mental-health-detection    | 0.4143±0.0427 | 0.4162±0.0467 | 0.4135±0.0449 | 0.4136±0.0520 | 0.4261±0.0453 | 0.4255±0.0456   | 0.4269±0.0434   | 0.4281±0.0458   |
| meta-all                  | 0.4832±0.1442 | 0.4996±0.1652 | 0.4828±0.1230 | 0.3979±0.1432 | 0.4590±0.1504 | 0.5044±0.1712   | 0.5044±0.1712   | 0.4859±0.1372   |

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

