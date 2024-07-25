from gridworld_dqn import *


def compute_cosine_similarity(data1, data2):
    data1 = data1.reshape(-1)
    data2 = data2.reshape(-1)
    numerator = np.dot(data1, data2)
    denominator = (
            np.sqrt(np.dot(data1, data1)) * np.sqrt(np.dot(data2, data2))
    )
    cosine_sim = numerator / denominator
    return cosine_sim
def train_and_plot_cosine_similarity():
    models_losses, models_gradients = train_models(model_types=['kp', 'bp'])


    # snr_dict = compute_cosine_similarity(gradient_dictionary_to_numpy(models_gradients))
    #
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    # plot_models_training_loss(model_losses, ax1)
    # plot_models_SNR(snr_dict, ax2)
    # plt.tight_layout()
    # plt.show()
