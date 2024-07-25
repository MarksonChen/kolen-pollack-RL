from gridworld_dqn import *


def compute_cosine_similarity_with_bp(models_grads, epsilon=1e-3):
    """
    models_grads: {linear_type: 4D numpy array}
                4D numpy array: [num_params x num_episodes x (<episode_steps)
                        x (gradients tensor)]
    """
    bp_grads = models_grads['bp']
    cs_dict = {}
    for linear_type in linear_types:
        grads1 = np.nan_to_num(bp_grads, nan=0.0)
        grads2 = np.nan_to_num(models_grads[linear_type], nan=0.0)
        # 1: Sum across the episode_steps dimension
        sum1 = np.sum(grads1, axis=2)
        sum2 = np.sum(grads2, axis=2)
        # Now shape: [num_params x num_episodes x (gradients tensor length)]
        # 2: Compute cosine similarity across gradient tensors
        dot_product = np.sum(sum1 * sum2, axis=2)
        # Now shape: [num_params x num_episodes]
        norm1 = np.linalg.norm(sum1, axis=2)
        norm2 = np.linalg.norm(sum2, axis=2)
        cosine_similarity = dot_product / (norm1 * norm2 + epsilon)
        # Step 3: Average cosine similarity across parameters
        cosine_similarity_avg = np.mean(cosine_similarity, axis=0)
        # Now shape: [num_episodes]
        cs_dict[linear_type] = cosine_similarity_avg
    return cs_dict

def plot_models_cosine_similarity(models_gradients, ax):
    cs_dict = compute_cosine_similarity_with_bp(gradient_dictionary_to_numpy(models_gradients))

    for linear_type in linear_types:
        ax.plot(cs_dict[linear_type], label=linear_full_name[linear_type])
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Cosine Similarity across episodes  (Scale-Invariant Bias)')
    ax.legend()


def train_and_plot_cosine_similarity():
    models_losses, models_gradients = train_models(model_types=['kp', 'bp'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    plot_models_training_loss(models_losses, ax1)
    plot_models_cosine_similarity(models_gradients, ax2)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_and_plot_cosine_similarity()
