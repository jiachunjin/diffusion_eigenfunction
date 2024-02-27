import torch
import numpy as np
import torch.distributions as td

def get_distribution_and_samples(num_samples):
    mu_x = torch.zeros((2,), dtype=torch.float32)
    cov_x = .2*torch.as_tensor([[1, 0.3], 
                            [0.3, 1]])
    px = td.MultivariateNormal(mu_x, cov_x)
    sample_x = px.sample((num_samples,))
    # --------------------------------------------------------------------
    A = torch.as_tensor([[-2, 2],
                        [4, 1]], dtype=torch.float32)
    m = torch.as_tensor([2, 1])
    L_a = .2*torch.as_tensor([[2., .3], 
                            [.3, 2.]])
    mu_a = A @ mu_x + m
    Sigma_a = L_a + A @ cov_x @ A.T
    pa = td.MultivariateNormal(mu_a, Sigma_a)
    sample_a = pa.sample((num_samples,))
    # --------------------------------------------------------------------
    B = torch.as_tensor([[-2, 1],
                        [-3, 2]], dtype=torch.float32)
    n = torch.as_tensor([-4, -4])
    L_b = .3*torch.as_tensor([[1, 0.5], 
                            [0.5, 1]])
    mu_b = B @ mu_x + n
    Sigma_b = L_b + B @ cov_x @ B.T
    pb = td.MultivariateNormal(mu_b, Sigma_b)
    sample_b = pb.sample((num_samples,))
    # --------------------------------------------------------------------
    C = torch.zeros((4, 4))
    d = torch.zeros((4,))
    L_c = .8*torch.ones((4, 4))
    mu_X = torch.zeros((4,))
    cov_X = torch.zeros((4, 4))
    C[:2, :2] = A
    C[-2:, -2:] = B
    d[:2] = m
    d[2:] = n
    L_c[:2, :2] = L_a
    L_c[-2:, -2:] = L_b
    mu_X[:2] = mu_x
    mu_X[2:] = mu_x
    cov_X[:2, :2] = cov_x
    cov_X[-2:, -2:] = cov_x

    mu_c = C @ mu_X + d
    Sigma_c = L_c + C @ cov_X @ C.T
    pab = td.MultivariateNormal(mu_c, Sigma_c)
    sample_ab = pab.sample((num_samples,))

    return px, pa, pb, pab, sample_x, sample_a, sample_b, sample_ab


class tractable_pair():
    def __init__(self, num_samples):
        _, _, _, _, _, _, _, sample_ab = get_distribution_and_samples(num_samples)
        self.data = sample_ab
    
    def __getitem__(self, index):
        return (self.data[index][:2], self.data[index][2:]), -1

    def __len__(self):
        return len(self.data)