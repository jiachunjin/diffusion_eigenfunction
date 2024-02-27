import torch
from utils.misc import append_dims


def evaluate_kernel(nef, eigenvalue, x1, x2):
    """
    Compute the positive pair kernel value k(x1, x2) with learned neural eigenfunction
    """
    nef.eval()
    return ((eigenvalue * nef(x1)) @ nef(x2).T)

def kernel_score(nef, eigenvalue, x1, x2, epsilon=1e-3):
    x1.requires_grad = True
    x2.requires_grad = True
    logK = evaluate_kernel(nef, eigenvalue, x1, x2, epsilon).log()
    
    logK.sum().backward()

    return x1.grad, x2.grad

def pair_score(nef, eigenvalue, x1, x2, score_a, score_b, sigma, epsilon=1e-3):
    s1, s2 = kernel_score(nef, eigenvalue, x1, x2)
    s1 += score_a(x1, append_dims(sigma, x1.ndim))
    s2 += score_b(x2, append_dims(sigma, x2.ndim))

def evaluate_tractable_kernel(x1, x2, pa, pb, pab):
    logk = pab.log_prob(torch.cat((x1, x2), dim=1)) - pa.log_prob(x1) - pb.log_prob(x2)
    return logk.exp()