import torch

from .nn import MLP


class EDM_Denoiser(torch.nn.Module):
    def __init__(self, model_type, sigma_data=0.5, **kwargs):
        super().__init__()
        if model_type == 'MLP':
            self.net = MLP(kwargs['dim'], kwargs['hidden_dim'], kwargs['dropout_p'])
        elif model_type == 'UNet':
            raise NotImplementedError
            # self.net = UNet(**kwargs)
        self.sigma_data = sigma_data

    def forward(self, x, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.net((c_in * x), c_noise)
        D_x = c_skip * x + c_out * F_x
        return D_x