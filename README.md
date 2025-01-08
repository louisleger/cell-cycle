# Cell Cycle Project

**Repository for the Naef Lab project on finding Latent Coordinates of the Cell Phase from Live-Imaging**

This repository contains code for a project investigating latent cell phase coordinates using live-imaging data for the Naef Lab.



## How to Run the Training Script

To train a neural network model with a transformer encoder:


    ```bash
    python modules/learning/train.py --temporal_encoder transformer --epochs 20 --slice_p 0.7 --random_len True
    ```

For all training options go to `modules/learning/train.py`

## nb_pretty_louis

Contains nice plots and a nice recap 0f the project

*   **data_overview.ipynb**: Analyzes data characteristics, including cell counts and data point distribution.
*   **analysis_overview.ipynb**: Explores data dynamics and cell cycle phases using your chosen FUCCI system.
*   **regression_overview.ipynb**: Investigates deep learning models for FUCCI regression, evaluating their performance on different data subsets.
*   **scripts** (folder, likely exists): Contains scripts for data visualization, neural network training, and evaluation (add if it exists).



