import pandas as pd

# Example DataFrame
data = {
    "Head": ["mlp", "cnn", "lstm", "xtransformer-causal", "mamba", "xtransformer"],
    "Modality": ["bf", "bf", "bf", "bf", "bf", "bf"],
    "L1_green": [0.130453, 0.104319, 0.084402, 0.083386, 0.082574, 0.083386],
    "L1_red": [0.174561, 0.126753, 0.099000, 0.097733, 0.094411, 0.097733],
    "t_g": [167.974860, 138.156425, 102.164804, 98.784916, 99.245810, 98.784916],
    "t_r": [120.377095, 95.125698, 88.198324, 83.645251, 83.533520, 83.645251],
}
df = pd.DataFrame(data)

# Convert to LaTeX
latex_table = df.style.format(precision=3).to_latex(hrules=True)

print(latex_table)
