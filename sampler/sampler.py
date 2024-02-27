import torch
import numpy as np
from tqdm.autonotebook import tqdm

from model.kernel import kernel_score
from utils.misc import append_dims


def euler_sampler(denoiser, latents, verbose=True, num_steps=200, rho=7, sigma_max=80, sigma_min=0.002):
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    x_next = latents.to(torch.float32) * t_steps[0]
    with tqdm(total=num_steps, disable=(not verbose)) as pbar:
        pbar.update(0)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            d = (x_cur - denoiser(x_cur, append_dims(t_cur.repeat(latents.shape[0]), x_cur.ndim)).to(torch.float32)) / t_cur
            x_next = x_cur + (t_next - t_cur) * d
            pbar.update(1)
    return x_next.detach()


def pair_euler_sampler(nef, eigenvalue, score_a, score_b, latents_a, latents_b, verbose=True, num_steps=200, rho=7, sigma_max=80, sigma_min=0.002):
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents_a.device)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    x_a_next = latents_a.to(torch.float32) * t_steps[0]
    x_b_next = latents_b.to(torch.float32) * t_steps[0]
    with tqdm(total=num_steps, disable=(not verbose)) as pbar:
        pbar.update(0)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_a_cur = x_a_next
            x_b_cur = x_b_next
            da = (x_a_cur - score_a(x_a_cur, append_dims(t_cur.repeat(latents_a.shape[0]), x_a_cur.ndim)).to(torch.float32)) / t_cur
            db = (x_b_cur - score_b(x_b_cur, append_dims(t_cur.repeat(latents_b.shape[0]), x_b_cur.ndim)).to(torch.float32)) / t_cur

            g1, g2 = kernel_score(nef, eigenvalue, x_a_cur, x_b_cur)

            x_a_next = (x_a_cur + (t_next - t_cur) * (da + g1)).detach()
            x_b_next = (x_b_cur + (t_next - t_cur) * (db + g2)).detach()
            pbar.update(1)
    return x_a_next.detach(), x_b_next.detach()

def sum_sampler(denoiser1, denoiser2, latents, verbose=True, num_steps=200, rho=7, sigma_max=80, sigma_min=0.002):
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    x_next = latents.to(torch.float32) * t_steps[0]
    with tqdm(total=num_steps, disable=(not verbose)) as pbar:
        pbar.update(0)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            tem1 = denoiser1(x_cur, append_dims(t_cur.repeat(latents.shape[0]), x_cur.ndim)).to(torch.float32)
            tem2 = denoiser2(x_cur, append_dims(t_cur.repeat(latents.shape[0]), x_cur.ndim)).to(torch.float32)
            d = (x_cur - tem1 - tem2) / t_cur
            x_next = x_cur + (t_next - t_cur) * d
            pbar.update(1)
    return x_next.detach()