import os

import torch
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from model.score_net import EDM_Denoiser
from sampler import euler_sampler
from loss.score_loss import Loss_EDM
from .base_trainer import Base_Trainer


class Score_Trainer(Base_Trainer):
    def __init__(self, config, logger, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, logger, log_dir, ckpt_dir, sample_dir)
        self.loss_fn = Loss_EDM()
        if self.config['training']['ema']:
            raise NotImplementedError

    def _build_net(self):
        if self.config['training']['resume']:
            raise NotImplementedError
        else:
            self.net = EDM_Denoiser(**self.config['score_net']).to(self.device)
            self.logger.info('Train from scratch')

    def train(self):
        total_iters = int(self.config['training']['iters'])
        done = False
        epoch = 0
        with tqdm(total=total_iters) as pbar:
            pbar.update(self.num_iters)
            while not done:
                for x, _ in self.train_loader:
                    if len(x) == 1:
                        x = x[0]
                    elif len(x) == 2:
                        raise NotImplementedError
                    x = x.float().to(self.device)
                    if self.config['data']['rescale']:
                        raise NotImplementedError
                    loss = self.loss_fn(net=self.net, x=x)
                    self.writer.add_scalar('loss', loss.item(), self.num_iters)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    self.num_iters += 1
                    self.history_iters += 1

                    if self.num_iters % int(self.config['training']['sample_iters']) == 0:
                        if self.config['data']['type'] == '2d':
                            self.sample_2d(num_steps=500)
                            self.logger.debug('sample')
                        else:
                            raise NotImplementedError
                    if self.num_iters % int(self.config['training']['save_iters']) == 0:
                        self.save()
                        self.logger.debug(f'Saved at history_iters: {self.history_iters}')
                    if self.num_iters >= total_iters:
                        done = True
                        break
                    epoch += 1

    def save(self):
        state = {
            'net': self.net.state_dict(),
            'net_config': self.config['score_net'],
            'history_iters': self.history_iters,
        }
        save_path = os.path.join(self.ckpt_dir, f"score_{self.config['data']['name']}.pt")
        torch.save(state, save_path)

    @torch.no_grad()
    def sample_2d(self, num_steps=1000, sampler=euler_sampler):
        self.net.eval()
        latents = torch.randn((500, 2)).to(self.device)
        samples = sampler(self.net, latents, verbose=False, num_steps=num_steps)
        samples = samples.detach().cpu()
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        self.writer.add_figure('samples', plt.gcf(), self.num_iters)
        self.net.train()
    
    @torch.no_grad()
    def sample_img_rgb(self, img_size, num_samples=16, num_steps=1000, sampler=euler_sampler):
        pass