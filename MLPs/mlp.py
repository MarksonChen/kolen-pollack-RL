from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(MLP, self).__init__()
        layers = []

        # Add hidden layers with ReLU activations
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        # Add the final layer without activation
        layers.append(nn.Linear(input_dim, output_dim))

        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
