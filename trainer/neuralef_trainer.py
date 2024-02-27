import os
import torch
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

from .base_trainer import Base_Trainer
from loss.nef_loss import Loss_NEF
from model.nef_net import NeuralEigenFunctions
from model.kernel import evaluate_kernel
from utils.misc import paired_dataloader2batch

class NeuralEF_Trainer(Base_Trainer):
    def __init__(self, config, logger, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, logger, log_dir, ckpt_dir, sample_dir)
        self.loss_fn = Loss_NEF()
        self.eigenvalue = None

    def _build_net(self):
        if self.config['training']['resume']:
            raise NotImplementedError
        else:
            self.net = NeuralEigenFunctions(**self.config['nef_net']).to(self.device)
            self.logger.info('Train from scratch')
    
    def train(self):
        total_iters = int(self.config['training']['iters'])
        done = False
        epoch = 0
        with tqdm(total=total_iters) as pbar:
            pbar.update(self.num_iters)
            while not done:
                for x, _ in self.train_loader:
                    self.net.train()
                    if len(x) == 1:
                        x = x[0]
                    elif len(x) == 2:
                        # paired data
                        x1, x2 = x
                    x1 = x1.float().to(self.device)
                    x2 = x2.float().to(self.device)
                    if self.config['data']['rescale']:
                        raise NotImplementedError
                    loss = self.loss_fn(net=self.net, x1=x1, x2=x2)
                    self.writer.add_scalar('loss', loss.item(), self.num_iters)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    self.num_iters += 1
                    self.history_iters += 1
                    self.update_eigenvalue(x1, x2)
                    if self.num_iters % int(self.config['training']['eval_iters']) == 0:
                        # print(self.net(x1).mean(dim=0).norm())
                        # print(self.eigenvalue)
                        self.evaluate()
                    if self.num_iters % int(self.config['training']['save_iters']) == 0:
                        self.save()

                    if self.num_iters >= total_iters:
                        done = True
                        break
                    epoch += 1
    
    def update_eigenvalue(self, x1, x2):
        B = x1.shape[0]
        psis_K_psis = self.net(x1).T @ self.net(x2)
        if self.eigenvalue is None:
            self.eigenvalue = psis_K_psis.diag().detach() / B
        else:
            # Q: why the first eigenvalue is always 1?
            self.eigenvalue.mul_(0.9).add_(psis_K_psis.diag() / B, alpha=0.1)
        # self.net.train()

    def evaluate(self):
        # print(self.net.eigennorm)
        x = next(iter(self.test_loader))[0]
        x1 = x[0].float().to(self.device)
        x2 = x[1].float().to(self.device)
        
        K = evaluate_kernel(self.net, self.eigenvalue, x1, x2).detach().cpu()
        self.net.train()
        plt.imshow(K)
        plt.colorbar()
        # ax = plt.gca()
        # ax.set_aspect('equal', adjustable='box')
        self.writer.add_figure('samples', plt.gcf(), self.num_iters)
        print(self.eigenvalue)

    def save(self):
        state = {
            'net': self.net.state_dict(),
            'net_config': self.config['nef_net'],
            'history_iters': self.history_iters,
            'eigenvalue': self.eigenvalue
        }
        save_path = os.path.join(self.ckpt_dir, f"nef_{self.config['data']['name']}.pt")
        torch.save(state, save_path)