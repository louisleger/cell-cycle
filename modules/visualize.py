import json
import matplotlib.pyplot as plt

# Function to plot training and test loss during training, from saved configuration files
def plot_loss(configs = ()):
    fig, ax = plt.subplots(1, 2, figsize = (10, 4), sharey=True)
    for config in configs:
        data = json.load(open(config))
        ax[0].plot(data['train_loss'], label = data['name']); ax[1].plot(data['test_loss'])
    ax[0].set_title('Train Loss'); ax[1].set_title('Test Loss'); ax[0].set_ylabel('$\ell_2$', fontsize = 14, rotation=0); ax[0].legend()
    ax[0].set_xlabel('Epochs');  ax[1].set_xlabel('Epochs')
    plt.tight_layout(); plt.show()

# Example Usage
# plot_loss(('../configs/config-1.json','../configs/config-2.json', '../configs/config-3.json', '../configs/config-4.json'))

def plot_normalized_time_errors(errors, bar = True):
    # Get Average L1 Loss of Model for different normalized times
    tau, mean = np.linspace(0, 1, errors.shape[1]), np.mean(errors, axis=0)/2
    
    # Get Vanilla prediction to overlay error barsr
    vanilla_prediction = np.load('../../vanilla_prediction.npy'); vanilla_at_tau = vanilla_prediction[(tau*len(vanilla_prediction)).astype(int).tolist()[:-1] + [-1]]
    
    # Plot the track normalized time wise errors in errorbars or fill betweens
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(np.linspace(0, 1, vanilla_prediction.shape[0]), vanilla_prediction[:, 0], 'green',np.linspace(0, 1, vanilla_prediction.shape[0]),  vanilla_prediction[:, 1], 'red')
    if bar == True: 
        for i in range(2): ax.errorbar(tau, vanilla_at_tau[:, i], yerr=mean[:, i], fmt=' ', ecolor = ['darkgreen', 'darkred'][i], capsize = 3)
    else: 
        for i in range(2): ax.fill_between(tau, vanilla_at_tau[:, i] - mean[:, i], vanilla_at_tau[:, i] + mean[:, i], color=['darkgreen', 'darkred'][i], alpha=0.4)
    
    # matplotlib mumbo jumbo
    ax.set_xlim(0,1); ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.set_title('Track Normalized Time Error Profiles'); ax.set_xlabel('$\\tau$', fontsize = 14); ax.set_ylabel('$F_{R,G}$', fontsize=14, rotation=0, labelpad=10);plt.tight_layout();plt.show()

# Plot how well our model performs with increasing number of frames given to the model
def plot_context_length_error(evaluations):
    # Get errors and context lengths of multiple evaluations ran
    context_lengths = [eval.slice_len for eval in evaluations]; errors = [eval.prediction_df['\ell_1'].mean() for eval in evaluations]
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(context_lengths, errors, marker='s', ls = '-', color='blue', label='Model')
    # matplotlib mumbo jumbo
    ax.set_xscale('log'); ax.set_title('Error vs Context Length', fontsize=14); ax.set_xlabel('t', fontsize=14); ax.set_ylabel('$\ell_1$', fontsize=14, labelpad=10, rotation=0)
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False);plt.tight_layout()
    return ax