# HiGEC
HiGEC: Hierarchy Generation and Extended Classification Framework
HiGEC is an open-source framework for advancing multi-class classification through automated class hierarchy generation (HG) and extended hierarchy exploitation (HE). The framework introduces novel HC+ and HC+F schemes that combine global and local classification strategies to improve performance.

HiGeC is evaluated on 115 multi-class datasets, demonstrating significant improvements over traditional flat classification (FC) and one-vs-all (OVA) approaches, particularly with advanced classifiers like XGBoost.
[Uploading fig_cdd_hge_vs_FC_vs_OVA_f1.eps…]()

Key Features
Automated class hierarchy generation from flat-label datasets.
Implementation of advanced hierarchy exploitation strategies (HC+ and HC+F).
Compatible with popular classifiers, including XGBoost, CART, ETC, and GNB.
Comprehensive benchmarking using open-source datasets from OpenML.
Installation
To use HiGeC, install the following required packages:

bash
Copy
Edit
pip install openml xgboost proglearn prince  
Usage
Load datasets via OpenML.
Configure hierarchy generation and classifier settings.
Run the HiGeC framework to evaluate multi-class classification performance.
