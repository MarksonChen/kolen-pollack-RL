from gridworld_dqn import *


def plot_models_SNR(snr_dict, ax):
    for linear_type in linear_types:
        ax.plot(snr_dict[linear_type], label=linear_full_name[linear_type])
    ax.set_xlabel('Episode')
    ax.set_ylabel('SNRs')
    ax.set_title('SNRs across episodes')
    ax.legend()


def gradient_dictionary_to_numpy(models_gradients):
    """
    input: model_gradients: {linear_type: gradients_dict}
                gradients_dict: {param_names: [num_episodes x (<episode_steps)
                        x (gradients tensor)]}
    output: {linear_type: 4D numpy array}
                4D numpy array: [num_params x num_episodes x (<episode_steps)
                        x (gradients tensor)]
    """
    models_grad_np = {}
    for linear_type in linear_types:
        param_grads = models_gradients[linear_type]
        for key in param_grads:
            if "feedback" in key:
                del param_grads[key]

        num_params = len(param_grads)
        max_flattened_size = max(param.numel() for episodes in param_grads.values() for steps in episodes for param in steps)
        grads_array = np.full((num_params, num_episodes, episode_steps, max_flattened_size), np.nan)

        # Gradient dictionary to 4D numpy array
        for i, (key, episodes) in enumerate(param_grads.items()):
            for j, steps in enumerate(episodes):
                for k, param in enumerate(steps):
                    flattened_tensor = param.flatten().numpy()
                    grads_array[i, j, k, :len(flattened_tensor)] = flattened_tensor

        models_grad_np[linear_type] = grads_array
    return models_grad_np


# def row_std_with_nan(arr, axis=2):
#     # Replace standard deviation of rows filled with np.nan with np.nan
#     nan_rows = np.all(np.isnan(arr), axis=axis)
#     std_devs = np.nanstd(arr, axis=axis)
#     std_devs[nan_rows] = np.nan
#     return std_devs

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
    model_losses = {}
    models_gradients = {}
    for linear_type in linear_types:
        losses, param_grads = train_dqn(linear_type, 42, get_gradients=True)
        model_losses[linear_type] = losses
        keys_to_delete = [key for key in param_grads.keys() if "feedback" in key]
        for key in keys_to_delete:
            del param_grads[key]
        models_gradients[linear_type] = param_grads

    snr_dict = compute_snr(gradient_dictionary_to_numpy(models_gradients))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    plot_models_training_loss(model_losses, ax1)
    plot_models_SNR(snr_dict, ax2)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train_and_plot_snr()


