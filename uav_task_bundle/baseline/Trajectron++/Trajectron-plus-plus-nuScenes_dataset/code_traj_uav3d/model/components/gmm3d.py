import torch
import torch.distributions as td
from model.model_utils import to_one_hot


class GMM3D(object):
    def __init__(self, log_pis, mus, log_sigmas, tril_offdiag, pred_state_length, device,
                 clip_lo=-10, clip_hi=10):
        if pred_state_length != 3:
            raise ValueError(f"GMM3D expects pred_state_length=3, got {pred_state_length}.")

        self.device = device
        self.pred_state_length = pred_state_length
        self.GMM_c = log_pis.shape[-1]

        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
        self.mus = self.reshape_to_components(mus, self.GMM_c)
        self.log_sigmas = self.reshape_to_components(
            torch.clamp(log_sigmas, min=clip_lo, max=clip_hi),
            self.GMM_c,
        )
        self.sigmas = torch.exp(self.log_sigmas)
        self.tril_offdiag = self.reshape_tril_offdiag(tril_offdiag, self.GMM_c)
        self.scale_tril = self.build_scale_tril()
        self.cat = td.Categorical(logits=self.log_pis)
        self.mvn = td.MultivariateNormal(loc=self.mus, scale_tril=self.scale_tril)
        self.mean = torch.sum(torch.softmax(self.log_pis, dim=-1).unsqueeze(-1) * self.mus, dim=-2)

    def reshape_to_components(self, tensor, GMM_c):
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [GMM_c, self.pred_state_length])

    @staticmethod
    def reshape_tril_offdiag(tensor, GMM_c):
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [GMM_c, 3])

    def build_scale_tril(self):
        scale_tril = torch.zeros(
            list(self.mus.shape[:-1]) + [self.pred_state_length, self.pred_state_length],
            device=self.device,
            dtype=self.mus.dtype,
        )
        scale_tril[..., 0, 0] = self.sigmas[..., 0]
        scale_tril[..., 1, 0] = self.tril_offdiag[..., 0]
        scale_tril[..., 1, 1] = self.sigmas[..., 1]
        scale_tril[..., 2, 0] = self.tril_offdiag[..., 1]
        scale_tril[..., 2, 1] = self.tril_offdiag[..., 2]
        scale_tril[..., 2, 2] = self.sigmas[..., 2]
        return scale_tril

    def sample(self):
        component_samples = self.mvn.rsample()
        cat_samples = self.cat.sample()
        selector = torch.unsqueeze(to_one_hot(cat_samples, self.GMM_c, self.device), dim=-1)
        return torch.sum(component_samples * selector, dim=-2)

    def log_prob(self, x):
        component_log_p = self.mvn.log_prob(torch.unsqueeze(x, dim=-2))
        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)
