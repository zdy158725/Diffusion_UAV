from typing import Dict

import torch

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer


class BaseImagePolicy(ModuleAttrMixin):
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            key -> B, To, *
        return:
            Dict containing at least "action"
        """
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
