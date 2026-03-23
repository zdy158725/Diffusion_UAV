if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import hashlib
import os
import random
from typing import Dict

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import (
    TrainDiffusionTransformerLowdimWorkspace,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


class BuildRerankerCandidateCacheWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        seed = int(cfg.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _load_policy(self) -> DiffusionTransformerLowdimPolicy:
        ckpt_path = os.path.expanduser(self.cfg.checkpoint_path)
        workspace = TrainDiffusionTransformerLowdimWorkspace.create_from_checkpoint(
            path=ckpt_path
        )
        policy = workspace.ema_model
        if policy is None or not bool(self.cfg.use_ema_model):
            policy = workspace.model
        return policy

    def _compute_candidate_metrics(
        self,
        cand_action: torch.Tensor,
        gt_action: torch.Tensor,
        start_pos: torch.Tensor,
        meters_per_unit: float,
    ) -> Dict[str, torch.Tensor]:
        cand_action_m = cand_action * meters_per_unit
        gt_action_m = gt_action * meters_per_unit
        start_pos_m = start_pos * meters_per_unit

        pred_xyz = start_pos_m[None, :, None, :] + torch.cumsum(cand_action_m, dim=2)
        gt_xyz = start_pos_m[:, None, :] + torch.cumsum(gt_action_m, dim=1)
        gt_xyz = gt_xyz[None, :, :, :]
        dist = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)
        gt_path_len = torch.linalg.norm(gt_action_m, dim=-1).sum(dim=1).clamp(min=1e-6)

        cand_ade_m = dist.mean(dim=2).transpose(0, 1)
        cand_fde_m = dist[:, :, -1].transpose(0, 1)
        cand_path_error_pct = (
            dist.sum(dim=2) / gt_path_len[None, :] * 100.0
        ).transpose(0, 1)
        best_idx = torch.argmin(cand_path_error_pct, dim=1)
        return {
            "cand_ade_m": cand_ade_m,
            "cand_fde_m": cand_fde_m,
            "cand_path_error_pct": cand_path_error_pct,
            "best_idx": best_idx,
        }

    def _build_split_cache(
        self,
        dataset: BaseLowdimDataset,
        policy: DiffusionTransformerLowdimPolicy,
        output_path: str,
        split_name: str,
        max_batches: int = None,
    ):
        cfg = self.cfg
        dataloader = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=bool(cfg.num_workers > 0 and cfg.persistent_workers),
            collate_fn=dataset.get_collate_fn(),
        )

        device = torch.device(cfg.device)
        policy.to(device)
        policy.eval()
        meters_per_unit = float(getattr(dataset, "meters_per_unit", 1.0))
        position_slice = getattr(dataset, "position_slice", slice(0, 3))
        start, end = policy.get_action_window_indices()
        anchor_idx = policy.get_action_anchor_obs_index()

        obs_hist_all = []
        gt_action_all = []
        cand_action_all = []
        cand_rel_pos_all = []
        cand_ade_all = []
        cand_fde_all = []
        cand_path_error_all = []
        best_idx_all = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                obs_dict = {"obs": batch["obs"]}
                obs_hist = batch["obs"][:, : policy.n_obs_steps]
                gt_action = batch["action"][:, start:end]
                start_pos = batch["obs"][:, anchor_idx, position_slice]

                cand_actions = []
                for _ in range(int(cfg.num_candidates)):
                    result = policy.predict_action(obs_dict)
                    cand_actions.append(result["action"])
                cand_action = torch.stack(cand_actions, dim=1)
                cand_rel_pos = torch.cumsum(cand_action, dim=2)
                metrics = self._compute_candidate_metrics(
                    cand_action=cand_action.transpose(0, 1),
                    gt_action=gt_action,
                    start_pos=start_pos,
                    meters_per_unit=meters_per_unit,
                )

                obs_hist_all.append(obs_hist.detach().cpu().numpy().astype(np.float32))
                gt_action_all.append(gt_action.detach().cpu().numpy().astype(np.float32))
                cand_action_all.append(cand_action.detach().cpu().numpy().astype(np.float32))
                cand_rel_pos_all.append(cand_rel_pos.detach().cpu().numpy().astype(np.float32))
                cand_ade_all.append(metrics["cand_ade_m"].detach().cpu().numpy().astype(np.float32))
                cand_fde_all.append(metrics["cand_fde_m"].detach().cpu().numpy().astype(np.float32))
                cand_path_error_all.append(
                    metrics["cand_path_error_pct"].detach().cpu().numpy().astype(np.float32)
                )
                best_idx_all.append(metrics["best_idx"].detach().cpu().numpy().astype(np.int64))

                if max_batches is not None and (batch_idx + 1) >= int(max_batches):
                    break

        if len(obs_hist_all) == 0:
            raise RuntimeError(f"No samples collected for split {split_name}.")

        obs_hist = np.concatenate(obs_hist_all, axis=0)
        gt_action = np.concatenate(gt_action_all, axis=0)
        cand_action = np.concatenate(cand_action_all, axis=0)
        cand_rel_pos = np.concatenate(cand_rel_pos_all, axis=0)
        cand_ade_m = np.concatenate(cand_ade_all, axis=0)
        cand_fde_m = np.concatenate(cand_fde_all, axis=0)
        cand_path_error_pct = np.concatenate(cand_path_error_all, axis=0)
        best_idx = np.concatenate(best_idx_all, axis=0)

        store = zarr.DirectoryStore(os.path.expanduser(output_path))
        root = zarr.group(store=store, overwrite=True)
        root.attrs["source_checkpoint_path"] = os.path.expanduser(self.cfg.checkpoint_path)
        root.attrs["source_checkpoint_sha256"] = _file_sha256(os.path.expanduser(self.cfg.checkpoint_path))
        root.attrs["split"] = split_name
        root.attrs["seed"] = int(cfg.seed)
        root.attrs["num_candidates"] = int(cfg.num_candidates)
        root.attrs["n_obs_steps"] = int(policy.n_obs_steps)
        root.attrs["n_action_steps"] = int(policy.n_action_steps)
        root.attrs["obs_dim"] = int(policy.obs_dim)
        root.attrs["action_dim"] = int(policy.action_dim)
        root.attrs["meters_per_unit"] = meters_per_unit
        root.attrs["num_groups"] = int(obs_hist.shape[0])

        data_group = root.create_group("data")
        data_group.create_dataset("obs_hist", data=obs_hist, chunks=(256, obs_hist.shape[1], obs_hist.shape[2]))
        data_group.create_dataset("gt_action", data=gt_action, chunks=(256, gt_action.shape[1], gt_action.shape[2]))
        data_group.create_dataset(
            "cand_action",
            data=cand_action,
            chunks=(64, cand_action.shape[1], cand_action.shape[2], cand_action.shape[3]),
        )
        data_group.create_dataset(
            "cand_rel_pos",
            data=cand_rel_pos,
            chunks=(64, cand_rel_pos.shape[1], cand_rel_pos.shape[2], cand_rel_pos.shape[3]),
        )
        data_group.create_dataset("cand_ade_m", data=cand_ade_m, chunks=(512, cand_ade_m.shape[1]))
        data_group.create_dataset("cand_fde_m", data=cand_fde_m, chunks=(512, cand_fde_m.shape[1]))
        data_group.create_dataset(
            "cand_path_error_pct",
            data=cand_path_error_pct,
            chunks=(512, cand_path_error_pct.shape[1]),
        )
        data_group.create_dataset("best_idx", data=best_idx, chunks=(1024,))

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if cfg.checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided.")

        os.makedirs(os.path.dirname(os.path.expanduser(cfg.train_output_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.expanduser(cfg.val_output_path)), exist_ok=True)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseLowdimDataset):
            raise TypeError("task.dataset must instantiate BaseLowdimDataset.")
        val_dataset = dataset.get_validation_dataset()

        policy = self._load_policy()
        self._build_split_cache(
            dataset=dataset,
            policy=policy,
            output_path=cfg.train_output_path,
            split_name="train",
            max_batches=cfg.max_train_batches,
        )
        self._build_split_cache(
            dataset=val_dataset,
            policy=policy,
            output_path=cfg.val_output_path,
            split_name="val",
            max_batches=cfg.max_val_batches,
        )
