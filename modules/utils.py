import json
import torch
import numpy as np
from scipy.stats import wasserstein_distance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Quick function to turn GPU PyTorch tensors from models to numpy arrays
def tensor_to_array(x): return x.squeeze().cpu().detach().numpy()

# Function to save the running training configuration and model
def save_config(config, model, config_dir): 
    with open(f'{config_dir}weights/config-{config["number"]}.json', 'w') as file:
        json.dump(config, file, indent=4)
    torch.save(model.state_dict(), config['path'] + f'model-{config["number"]}.pt')

# Count the learnable parameters of a model
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Make any large number readable, useful for numbers > 1k 
def human_readable(num, dec=2):
    magnitude = 0
    while abs(num) >= 1000: magnitude += 1; num /= 1000.0
    return ('{:.' + str(dec) + 'f}{}').format(num, ['', 'K', 'M', 'B', 'T'][magnitude])

# Get the number of model parameters in a readable number
def hc(model): return human_readable(count_parameters(model))

# Compute tolerance accuracy of a prediction y_pred vs groundtruth y
def tolerance_accuracy(y, y_pred, tol=0.15): return np.mean(np.abs(y-y_pred) < tol)

# Compute Wasserstein distance for (n, d) dimensional distributions x and y
def nd_wasserstein(x, y):
    result = 0 
    for dim in range(x.shape[1]): result += wasserstein_distance(x[:, dim], y[:, dim])
    return result/x.shape[1]

# Run a moving/rolling average on a time-series signal
#def moving_average(arr, window = 5): 
#    if (window==0): return arr
#    if arr.ndim > 1: 
#        result = np.zeros_like(arr)
#        for i in range(arr.shape[1]):  result[:, i] = np.convolve(arr[:, i], np.ones(window), mode='valid')/window
#        return result
#    return np.convolve(arr, np.ones(window), mode='valid')/window

def moving_average(arr, window=5):
    if window == 0: 
        return arr
    pad_width = window // 2

    if arr.ndim > 1:
        result = np.zeros_like(arr)
        for i in range(arr.shape[1]):
            # Padding with the edge values to avoid zero-padding effects
            padded_col = np.pad(arr[:, i], pad_width, mode='edge')
            result[:, i] = np.convolve(padded_col, np.ones(window), mode='valid') / window
        return result

    # Padding with the edge values to avoid zero-padding effects
    padded_arr = np.pad(arr, pad_width, mode='edge')
    return np.convolve(padded_arr, np.ones(window), mode='valid') / window

# Run multiple moving averages with different window sizes
def multi_layer_moving_average(x, layers = 3, windows = [5, 5, 5]):
    for ldx in range(layers): x = moving_average(x, window=windows[ldx])
    return x