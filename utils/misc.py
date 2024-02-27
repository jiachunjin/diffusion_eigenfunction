import torch
from model.nef_net import NeuralEigenFunctions
from model.score_net import EDM_Denoiser

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def append_dims(x, target_dims):
    """
    Appends dimensions to the end of a tensor until it has target_dims dimensions.
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def paired_dataloader2batch(dataset):
    x1_list = []
    x2_list = []
    y_list = []
    for b, (x, y) in enumerate(dataset):
        x1, x2 = x
        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(y)
    return torch.cat(x1_list), torch.cat(x2_list), torch.cat(y_list)

def load_nefnet(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    net_config = state['net_config']
    net = NeuralEigenFunctions(**net_config)
    net.load_state_dict(state['net'])
    eigenvalue = state['eigenvalue'].detach()

    return eigenvalue, net

def load_scorenet(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    net_config = state['net_config']
    net = EDM_Denoiser(**net_config)
    net.load_state_dict(state['net'])
    return net