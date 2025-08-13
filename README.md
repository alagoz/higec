# HiGEC: Hierarchy Generation and Extended Classification Framework
HiGEC is a Python framework for performing hierarchical classification with automated hierarchy generation, flexible exploitation strategies, and integration with modern classifiers.

HiGeC is evaluated on 100 multi-class datasets, demonstrating significant improvements over traditional flat classification (FC) approaches, particularly with advanced classifiers like XGBoost, RF, ETC, and LGBM.

![fig_cdd_hge_vs_FC_vs_OVA_f1]()

| name                     | RF           | XGBoost          | ETC          | LGBM          | LCN[XGB]+      | LCPN[ETC]+F[XGB] | LCPN[RF]+F[XGB] | LCPN[XGB]+F[XGB] |
|--------------------------|--------------|------------------|--------------|---------------|----------------|------------------|-----------------|------------------|
| air-quality-and-pollution| 0.9291±0.0089 | 0.9284±0.0116 | 0.9247±0.0110 | 0.9290±0.0105 | 0.9271±0.0119 | 0.9291±0.0114 | 0.9289±0.0112 | 0.9290±0.0101 |
| alizadeh-2000-v2         | 0.9485±0.0630 | 0.8594±0.1081 | 0.9743±0.0515 | 0.9440±0.0668 | 0.8841±0.1039 | 0.8809±0.1219 | 0.8681±0.1153 | 0.9357±0.0643 |
| amazon-commerce-reviews   | 0.6670±0.0281 | 0.7090±0.0253 | 0.7273±0.0098 | 0.7077±0.0294 | 0.5945±0.0203 | 0.7402±0.0221 | 0.7257±0.0262 | 0.5890±0.0157 |
| analcatdata-halloffame    | 0.6588±0.0652 | 0.6516±0.0563 | 0.6663±0.0507 | 0.6490±0.0648 | 0.6565±0.0733 | 0.6530±0.0621 | 0.6532±0.0624 | 0.6474±0.0738 |
| analcatdata-happiness     | 0.3204±0.1043 | 0.3990±0.0990 | 0.2605±0.1078 | 0.5259±0.1395 | 0.5285±0.0643 | 0.4019±0.1092 | 0.4019±0.1092 | 0.3601±0.0962 |
| anneal                    | 0.7733±0.0739 | 0.7976±0.0746 | 0.7814±0.0627 | 0.7722±0.0749 | 0.7690±0.0866 | 0.7995±0.0811 | 0.8021±0.0887 | 0.7871±0.0717 |
| arrythmia                 | 0.5761±0.0438 | 0.5924±0.0744 | 0.5257±0.0488 | 0.5946±0.0609 | 0.6397±0.0380 | 0.6305±0.0654 | 0.6281±0.0664 | 0.6267±0.0755 |
| artificial-characters     | 0.9102±0.0058 | 0.8998±0.0078 | 0.9095±0.0065 | 0.9119±0.0081 | 0.8917±0.0055 | 0.9199±0.0060 | 0.9139±0.0067 | 0.9102±0.0069 |
| auto-ml-selector          | 0.3392±0.0961 | 0.3710±0.1210 | 0.2765±0.0984 | 0.3560±0.1240 | 0.3635±0.1234 | 0.3682±0.1178 | 0.3782±0.1265 | 0.3922±0.1208 |
| auto-univ-au4-2500        | 0.4071±0.0133 | 0.4562±0.0118 | 0.3850±0.0127 | 0.4627±0.0145 | 0.4670±0.0118 | 0.4560±0.0119 | 0.4560±0.0119 | 0.4071±0.0133 |
| auto-univ-au7-1100        | 0.3775±0.0307 | 0.3680±0.0245 | 0.3614±0.0357 | 0.3701±0.0338 | 0.3813±0.0320 | 0.3785±0.0342 | 0.3772±0.0261 | 0.3598±0.0346 |
| bach                      | 0.5470±0.0379 | 0.5756±0.0389 | 0.5575±0.0381 | 0.0231±0.0127 | 0.5916±0.0295 | 0.5876±0.0374 | 0.5823±0.0395 | 0.5447±0.0353 |
| baseball                  | 0.6634±0.0590 | 0.6516±0.0563 | 0.6663±0.0507 | 0.6490±0.0648 | 0.6565±0.0733 | 0.6530±0.0621 | 0.6532±0.0624 | 0.6439±0.0713 |
| bridges                   | 0.5828±0.1390 | 0.5335±0.0787 | 0.5567±0.0822 | 0.4521±0.1287 | 0.5701±0.0998 | 0.5448±0.0780 | 0.5582±0.1020 | 0.5771±0.1267 |
| calendar-dow              | 0.5704±0.0303 | 0.5569±0.0385 | 0.5794±0.0415 | 0.5620±0.0320 | 0.5655±0.0320 | 0.5669±0.0278 | 0.5651±0.0304 | 0.5730±0.0283 |
| cars                      | 0.7804±0.0451 | 0.7802±0.0356 | 0.7710±0.0384 | 0.7790±0.0671 | 0.7782±0.0398 | 0.7797±0.0415 | 0.7824±0.0436 | 0.7863±0.0533 |
| cervical-cancer-risk      | 0.2480±0.0224 | 0.2315±0.0283 | 0.2510±0.0198 | 0.2082±0.0202 | 0.2234±0.0243 | 0.2328±0.0281 | 0.2304±0.0287 | 0.2454±0.0207 |
| cleveland-nominal         | 0.2638±0.0286 | 0.2663±0.0453 | 0.2825±0.0412 | 0.2605±0.0307 | 0.2519±0.0434 | 0.2669±0.0565 | 0.2604±0.0490 | 0.2601±0.0439 |
| cnae-9                    | 0.9304±0.0055 | 0.9194±0.0154 | 0.9405±0.0139 | 0.8423±0.0145 | 0.8958±0.0143 | 0.9255±0.0152 | 0.9242±0.0165 | 0.9291±0.0104 |
| collins                   | 0.2055±0.0232 | 0.2030±0.0180 | 0.2057±0.0199 | 0.1932±0.0213 | 0.1868±0.0181 | 0.2164±0.0183 | 0.2088±0.0141 | 0.2033±0.0224 |
| deng-reads                | 0.9339±0.0218 | 0.9430±0.0197 | 0.9368±0.0126 | 0.9540±0.0563 | 0.9750±0.0244 | 0.9623±0.0154 | 0.9623±0.0154 | 0.9382±0.0211 |
| dgf                       | 0.6104±0.0533 | 0.6355±0.0690 | 0.6394±0.0533 | 0.6333±0.0725 | 0.6352±0.0960 | 0.6506±0.0920 | 0.6347±0.0676 | 0.6151±0.0719 |
| diabetes-130-us           | 0.3920±0.0024 | 0.4066±0.0031 | 0.3906±0.0023 | 0.3999±0.0033 | 0.3563±0.0028 | 0.3481±0.0023 | 0.3505±0.0022 | 0.3426±0.0027 |
| diggle-table-a2           | 0.9585±0.0304 | 0.9628±0.0144 | 0.9747±0.0186 | 0.9791±0.0113 | 0.9690±0.0175 | 0.9644±0.0143 | 0.9643±0.0143 | 0.9731±0.0168 |
| eda-mortgage-ny           | 0.5001±0.0036 | 0.4850±0.0031 | 0.4927±0.0031 | 0.4824±0.0038 | 0.4681±0.0051 | 0.4861±0.0036 | 0.4823±0.0022 | 0.4845±0.0042 |
| energy-efficiency         | 0.6951±0.0253 | 0.6972±0.0373 | 0.6870±0.0363 | 0.6857±0.0261 | 0.6951±0.0388 | 0.7036±0.0372 | 0.6946±0.0417 | 0.6948±0.0309 |
| financial-risk-assessment | 0.2796±0.0039 | 0.3045±0.0065 | 0.2999±0.0117 | 0.2686±0.0032 | 0.3098±0.0115 | 0.3081±0.0184 | 0.3076±0.0192 | 0.2942±0.0216 |
| flags                     | 0.5562±0.0746 | 0.5597±0.0591 | 0.5161±0.0586 | 0.5663±0.1027 | 0.5712±0.0675 | 0.5833±0.0612 | 0.5741±0.0580 | 0.5922±0.0808 |
| flare                     | 0.6143±0.0239 | 0.6207±0.0187 | 0.6050±0.0166 | 0.6126±0.0280 | 0.6127±0.0248 | 0.6235±0.0212 | 0.6212±0.0202 | 0.6152±0.0255 |
| football-player-position  | 0.8476±0.0134 | 0.8456±0.0152 | 0.8466±0.0184 | 0.8483±0.0124 | 0.8491±0.0128 | 0.8495±0.0133 | 0.8479±0.0132 | 0.8507±0.0113 |
| golub-1999-v2             | 0.8575±0.1599 | 0.9233±0.0789 | 0.8984±0.1400 | 0.9164±0.1282 | 0.9065±0.0629 | 0.9250±0.0675 | 0.9301±0.0671 | 0.8371±0.1300 |
| hayes-roth                | 0.8212±0.0303 | 0.7848±0.0417 | 0.8247±0.0573 | 0.8224±0.0793 | 0.8020±0.0528 | 0.8160±0.0474 | 0.8177±0.0452 | 0.8212±0.0303 |
| heart-h                   | 0.3579±0.0385 | 0.3402±0.0646 | 0.3581±0.0673 | 0.3054±0.0637 | 0.2958±0.0667 | 0.3602±0.0564 | 0.3579±0.0562 | 0.3459±0.0475 |
| heart-switzerland         | 0.2571±0.0463 | 0.2760±0.0837 | 0.2455±0.0646 | 0.2354±0.0555 | 0.2420±0.0773 | 0.2883±0.0886 | 0.2878±0.0842 | 0.2742±0.0510 |
| helena                    | 0.2037±0.0044 | 0.2124±0.0047 | 0.2053±0.0051 | 0.0197±0.0083 | 0.1903±0.0052 | 0.2169±0.0033 | 0.2152±0.0039 | 0.2102±0.0032 |
| hepatitis-c               | 0.5256±0.0714 | 0.5570±0.0762 | 0.5596±0.0546 | 0.5880±0.0833 | 0.6150±0.0925 | 0.5922±0.0498 | 0.5922±0.0498 | 0.5773±0.0623 |
| hypothyroid               | 0.3608±0.0188 | 0.3575±0.0236 | 0.3622±0.0201 | 0.3544±0.0225 | 0.3513±0.0207 | 0.3599±0.0197 | 0.3626±0.0230 | 0.3595±0.0179 |
| internet-firewall         | 0.8427±0.0375 | 0.8234±0.0531 | 0.8291±0.0413 | 0.8115±0.0950 | 0.8310±0.0492 | 0.8234±0.0531 | 0.8234±0.0531 | 0.8431±0.0376 |
| ipums-la-98-small         | 0.4097±0.0138 | 0.4079±0.0171 | 0.4157±0.0185 | 0.4071±0.0194 | 0.4057±0.0207 | 0.4108±0.0160 | 0.4105±0.0196 | 0.4105±0.0142 |
| ipums-la-99-small         | 0.4437±0.0088 | 0.4454±0.0145 | 0.4264±0.0146 | 0.4435±0.0145 | 0.4358±0.0126 | 0.4405±0.0164 | 0.4439±0.0148 | 0.4401±0.0135 |
| khan-2001                 | 0.9881±0.0357 | 0.9699±0.0331 | 0.9945±0.0165 | 0.9782±0.0348 | 0.9740±0.0263 | 0.9845±0.0239 | 0.9845±0.0239 | 0.9875±0.0376 |
| kropt                     | 0.8306±0.0083 | 0.8622±0.0115 | 0.8003±0.0122 | 0.5748±0.0984 | 0.8466±0.0101 | 0.8632±0.0062 | 0.8625±0.0080 | 0.8351±0.0089 |


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

