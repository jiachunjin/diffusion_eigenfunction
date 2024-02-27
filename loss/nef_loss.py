import torch


class Loss_NEF():
    def __init__(self, alpha=1.):
        """
        # Eq.8 in [NEURAL EIGENFUNCTIONS ARE STRUCTURED REPRESENTATION LEARNERS]
        https://github.com/thudzj/NeuralEigenFunction/blob/main/neuralef-classic-kernels.py#L73
        """
        self.alpha = alpha

    def __call__(self, net, x1, x2):
        B, k = x1.shape
        psis_X_1 = net(x1) # (B, k)
        psis_X_2 = net(x2) # (B, k)
        term_1 = (psis_X_1.T @ psis_X_2).trace() # (k, k)

        Psi_PsiT_sg_squared = (psis_X_1.T.detach() @ psis_X_2) ** 2
        term_2 = (Psi_PsiT_sg_squared - Psi_PsiT_sg_squared.tril()).sum()

        loss = -term_1 + self.alpha * term_2
        
        return loss