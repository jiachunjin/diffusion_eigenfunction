import torch


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        modules = []
        dim_input = dim + 1 # concatenated with the scalar sigma
        for i in hidden_dim:
            modules.extend([
                torch.nn.Linear(dim_input, i),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout_p)
            ])
            dim_input = i
        modules.append(torch.nn.Linear(dim_input, dim))
        self.net = torch.nn.Sequential(*modules)
    
    def forward(self, x, noise_labels):
        if not noise_labels.dim():
            noise_labels = noise_labels.expand([x.shape[0], 1])
        inputs = torch.cat([x, noise_labels], dim=1)
        return self.net(inputs)