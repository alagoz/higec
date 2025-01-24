# HiGEC: Hierarchy Generation and Extended Classification Framework
HiGEC is an open-source framework for advancing multi-class classification through automated class hierarchy generation (HG) and extended hierarchy exploitation (HE). The framework introduces novel HC+ and HC+F schemes that combine global and local classification strategies to improve performance.

HiGeC is evaluated on 115 multi-class datasets, demonstrating significant improvements over traditional flat classification (FC) and one-vs-all (OVA) approaches, particularly with advanced classifiers like XGBoost.
![fig_cdd_hge_vs_FC_vs_OVA_f1](https://github.com/user-attachments/assets/7853f94c-3064-49af-a1fb-7693d9c09928)

Key Features
Automated class hierarchy generation from flat-label datasets.
Implementation of advanced hierarchy exploitation strategies (HC+ and HC+F).
Compatible with popular classifiers, including XGBoost, CART, ETC, and GNB.
Comprehensive benchmarking using open-source datasets from OpenML.
Installation
To use HiGeC, install the following required packages:

Required Packages
pip install openml xgboost proglearn prince

Usage
Load datasets via OpenML.
Configure hierarchy generation and classifier settings.
Run the HiGeC framework to evaluate multi-class classification performance.
