from gridworld_dqn import *


def plot_models_snr(models_gradients, ax):
    snr_dict = compute_snr(gradient_dictionary_to_numpy(models_gradients))

    for linear_type in linear_types:
        ax.plot(snr_dict[linear_type], label=linear_full_name[linear_type])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Signal-to-Noise Ratio')
    ax.set_title('SNRs across episodes (Scale-Invariant Variance)')
    ax.legend()

def compute_snr(models_grads, epsilon=1e-3):
    """
    models_grads: {linear_type: 4D numpy array}
                4D numpy array: [num_params x num_episodes x (<episode_steps)
                        x (gradients tensor)]
    """
    models_snr = {}
    for linear_type in linear_types:
        grads = models_grads[linear_type]
        mean_across_steps = np.nanmean(np.nan_to_num(grads, nan=0.0), axis=2)
        std_across_steps = np.nanstd(np.nan_to_num(grads, nan=0.0), axis=2)
        snr_across_steps = np.abs(mean_across_steps) / (std_across_steps + epsilon)
        # Now shape: [num_params x num_episodes x (gradients tensor length)]
        snr_across_flattened = np.nanmean(snr_across_steps, axis=2)
        # Now shape: [num_params x num_episodes]
        snr_across_params = np.nanmean(snr_across_flattened, axis=0)
        # Now shape: [num_episodes]
        models_snr[linear_type] = snr_across_params
    return models_snr

def train_and_plot_snr():
    models_losses, models_gradients = train_models(model_types=['kp', 'bp'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    plot_models_training_loss(models_losses, ax1)
    plot_models_snr(models_gradients, ax2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_and_plot_snr()


