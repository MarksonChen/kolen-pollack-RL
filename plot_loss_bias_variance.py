from cosine_similarity import *
from snr import *

models_losses, models_gradients = train_models(model_types=['kp', 'bp'])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
plot_models_training_loss(models_losses, ax1)
plot_models_cosine_similarity(models_gradients, ax2)
plot_models_snr(models_gradients, ax3)

plt.tight_layout()
plt.show()
