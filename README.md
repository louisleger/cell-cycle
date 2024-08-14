# Cell Cycle Project
Repository for the Naef Lab project on finding Latent Coordinates of the Cell Phase from Live-Imaging

The main findings of this project are presented in the 3 main notebooks:
- data_overview.ipynb 
    - Quantifies the number of cells and data points we have 
- analysis_overview.ipynb 
    - Analyses the dynamics of the data and the cell cycle phases with our chosen FUCCI system
- regression_overview.ipynb
    - Regresses the FUCCI using deep learning models and explains their performance based on different subsets of data

The rest of the repository contains the scripts used to visualize data, train and evaluate neural networks
```
.
├── 0_data_overview.ipynb
├── 1_analysis_overview.ipynb
├── 2_regression_overview.ipynb
├── README.md
├── modules
│   ├── __init__.py
│   ├── learning
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── mamba.py
│   │   ├── models.py
│   │   ├── pscan.py
│   │   └── train.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   ├── track_extraction.py
│   │   └── track_filter.py
│   ├── utils.py
│   └── visualize.py
├── weights
└── well_info.csv
```

