import json
import numpy as np
import matplotlib.pyplot as plt


# Function to plot training and test loss during training, from saved configuration files
def plot_loss(configs=()):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for config in configs:
        data = json.load(open(config))
        ax[0].plot(data["train_loss"], label=data["name"])
        ax[1].plot(data["test_loss"])
    ax[0].set_title("Train Loss")
    ax[1].set_title("Test Loss")
    ax[0].set_ylabel("$\ell_2$", fontsize=14, rotation=0, labelpad=10)
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylim(0, max(data["train_loss"]))
    ax[1].set_ylim(0, max(data["train_loss"]))
    plt.tight_layout()
    plt.show()


# Example Usage
# plot_loss(('../configs/config-1.json','../configs/config-2.json', '../configs/config-3.json', '../configs/config-4.json'))


# Plot how well our model performs with increasing number of frames given to the model
def plot_context_length_error(evaluations):
    # Get errors and context lengths of multiple evaluations ran
    context_lengths = [eval.slice_len for eval in evaluations]
    errors = [eval.prediction_df["\ell_1"].mean() for eval in evaluations]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(context_lengths, errors, marker="s", ls="-", color="blue", label="Model")
    # matplotlib mumbo jumbo
    ax.set_xscale("log")
    ax.set_title("Error vs Context Length", fontsize=14)
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel("$\ell_1$", fontsize=14, labelpad=10, rotation=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    return ax


def plot_normalized_time_error(error):
    mean = error.mean(axis=0)
    std = error.std(axis=0)
    tau = np.linspace(0, 1, error.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for i, signal in enumerate([1, 0]):
        axes[i].plot(
            tau, mean[:, signal], label="$\mu$", color=["green", "red"][signal]
        )
        axes[i].fill_between(
            tau,
            mean[:, signal] - std[:, signal],
            mean[:, signal] + std[:, signal],
            alpha=0.2,
            label="$\sigma$",
            color=["green", "red"][signal],
        )

        axes[i].set_xlabel("$\\tau$", fontsize=14)
        axes[i].set_ylabel("$\ell_1$", rotation=0, fontsize=14, labelpad=10)
        axes[i].set_title(f"$y_{['g', 'r'][signal]}$", fontsize=16)
        axes[i].legend()

    plt.tight_layout()
    plt.show()
