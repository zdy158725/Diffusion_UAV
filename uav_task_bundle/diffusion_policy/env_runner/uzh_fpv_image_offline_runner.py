import math
import os

import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class UZHFPVImageOfflineRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        dataset,
        batch_size=16,
        num_workers=0,
        pin_memory=True,
        device=None,
        max_batches=None,
        multi_sample_eval_samples=1,
        multi_sample_eval_max_batches=None,
    ):
        super().__init__(output_dir)
        if isinstance(dataset, (DictConfig, ListConfig, dict, list)):
            self.dataset = hydra.utils.instantiate(dataset)
        else:
            self.dataset = dataset
        self.val_dataset = self.dataset.get_validation_dataset()
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.device = torch.device(device) if device is not None else None
        self.max_batches = max_batches
        self.multi_sample_eval_samples = max(int(multi_sample_eval_samples), 1)
        self.multi_sample_eval_max_batches = (
            None if multi_sample_eval_max_batches is None else max(int(multi_sample_eval_max_batches), 1)
        )
        self.action_scale_to_meter = float(getattr(self.dataset, "meters_per_unit", 1.0))
        self.position_slice = getattr(self.dataset, "position_slice", slice(0, 3))
        self.state_key = "state"
        raw_position_max_abs = getattr(self.dataset, "raw_position_max_abs", None)
        if raw_position_max_abs is not None:
            self.position_norm_max_abs = torch.as_tensor(raw_position_max_abs, dtype=torch.float32)
        else:
            state_all = self.dataset.replay_buffer["state"][:, self.position_slice].astype(np.float32)
            self.position_norm_max_abs = torch.from_numpy(
                np.maximum(np.max(np.abs(state_all), axis=0), 1e-6).astype(np.float32)
            )

    def _normalize_position(self, xyz_m):
        max_abs = self.position_norm_max_abs.to(device=xyz_m.device, dtype=xyz_m.dtype)
        return xyz_m / max_abs

    def _compute_path_stats(self, pred_action_m, gt_action_m, start_pos_m):
        pred_xyz = start_pos_m[:, None, :] + torch.cumsum(pred_action_m, dim=1)
        gt_xyz = start_pos_m[:, None, :] + torch.cumsum(gt_action_m, dim=1)
        dist = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)
        gt_path_len = torch.linalg.norm(gt_action_m, dim=-1).sum(dim=1).clamp(min=1e-6)
        return pred_xyz, gt_xyz, dist, gt_path_len

    def _reduce_metrics(self, pred_action_m, gt_action_m, pred_xyz_m, gt_xyz_m, dist, gt_path_len):
        mse = torch.nn.functional.mse_loss(pred_action_m, gt_action_m)
        mae = torch.nn.functional.l1_loss(pred_action_m, gt_action_m)
        action_rmse = torch.sqrt(mse)
        pred_xyz_norm = self._normalize_position(pred_xyz_m)
        gt_xyz_norm = self._normalize_position(gt_xyz_m)
        traj_mse_norm = torch.nn.functional.mse_loss(pred_xyz_norm, gt_xyz_norm)
        traj_rmse_norm = torch.sqrt(traj_mse_norm)
        ade = dist.mean()
        fde = dist[:, -1].mean()
        path_error_pct = (dist.sum(dim=1) / gt_path_len * 100.0).mean()
        fde_ratio_pct = (dist[:, -1] / gt_path_len * 100.0).mean()
        return {
            "action_mse": mse.item(),
            "action_mae": mae.item(),
            "action_rmse": action_rmse.item(),
            "traj_mse_norm": traj_mse_norm.item(),
            "traj_rmse_norm": traj_rmse_norm.item(),
            "traj_ade_m": ade.item(),
            "traj_fde_m": fde.item(),
            "traj_path_error_pct": path_error_pct.item(),
            "traj_fde_ratio_pct": fde_ratio_pct.item(),
        }

    def _select_best_of_n(self, pred_action_m_all, gt_action_m, start_pos_m):
        pred_xyz = start_pos_m[None, :, None, :] + torch.cumsum(pred_action_m_all, dim=2)
        gt_xyz = start_pos_m[None, :, None, :] + torch.cumsum(gt_action_m[None, :, :, :], dim=2)
        dist = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)
        gt_path_len = torch.linalg.norm(gt_action_m, dim=-1).sum(dim=1).clamp(min=1e-6)
        path_error_pct = dist.sum(dim=2) / gt_path_len[None, :] * 100.0
        best_idx = torch.argmin(path_error_pct, dim=0)
        batch_idx = torch.arange(gt_action_m.shape[0], device=gt_action_m.device)
        best_pred_action_m = pred_action_m_all[best_idx, batch_idx]
        best_pred_xyz_m = pred_xyz[best_idx, batch_idx]
        best_gt_xyz_m = gt_xyz[0, batch_idx]
        best_dist = dist[best_idx, batch_idx]
        best_metrics = self._reduce_metrics(
            best_pred_action_m,
            gt_action_m,
            best_pred_xyz_m,
            best_gt_xyz_m,
            best_dist,
            gt_path_len,
        )
        return best_metrics, best_idx

    def run(self, policy):
        device = self.device if self.device is not None else policy.device
        mse_vals = []
        mae_vals = []
        action_rmse_vals = []
        traj_mse_norm_vals = []
        traj_rmse_norm_vals = []
        ade_vals = []
        fde_vals = []
        path_error_pct_vals = []
        fde_ratio_pct_vals = []
        top1_subset_ade_vals = []
        top1_subset_fde_vals = []
        top1_subset_traj_mse_norm_vals = []
        top1_subset_traj_rmse_norm_vals = []
        top1_subset_path_error_pct_vals = []
        top1_subset_fde_ratio_pct_vals = []
        best_of_n_mse_vals = []
        best_of_n_mae_vals = []
        best_of_n_action_rmse_vals = []
        best_of_n_traj_mse_norm_vals = []
        best_of_n_traj_rmse_norm_vals = []
        best_of_n_ade_vals = []
        best_of_n_fde_vals = []
        best_of_n_path_error_pct_vals = []
        best_of_n_fde_ratio_pct_vals = []
        best_of_n_batch_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                obs_dict = batch["obs"]
                gt_action = batch["action"]

                if hasattr(policy, "get_action_window_indices"):
                    start, end = policy.get_action_window_indices()
                else:
                    start = int(policy.n_obs_steps)
                    end = start + int(policy.n_action_steps)
                anchor_idx = int(policy.n_obs_steps) - 1

                result = policy.predict_action(obs_dict)
                if getattr(policy, "pred_action_steps_only", False):
                    pred_action = result["action"]
                    gt_action = gt_action[:, start:end]
                else:
                    pred_action = result["action_pred"]

                pred_action_m = pred_action * self.action_scale_to_meter
                gt_action_m = gt_action * self.action_scale_to_meter
                start_pos_m = batch["obs"][self.state_key][:, anchor_idx, self.position_slice] * self.action_scale_to_meter

                pred_xyz_m, gt_xyz_m, dist, gt_traj_len = self._compute_path_stats(
                    pred_action_m=pred_action_m,
                    gt_action_m=gt_action_m,
                    start_pos_m=start_pos_m,
                )
                top1_metrics = self._reduce_metrics(
                    pred_action_m=pred_action_m,
                    gt_action_m=gt_action_m,
                    pred_xyz_m=pred_xyz_m,
                    gt_xyz_m=gt_xyz_m,
                    dist=dist,
                    gt_path_len=gt_traj_len,
                )
                mse_vals.append(top1_metrics["action_mse"])
                mae_vals.append(top1_metrics["action_mae"])
                action_rmse_vals.append(top1_metrics["action_rmse"])
                traj_mse_norm_vals.append(top1_metrics["traj_mse_norm"])
                traj_rmse_norm_vals.append(top1_metrics["traj_rmse_norm"])
                ade_vals.append(top1_metrics["traj_ade_m"])
                fde_vals.append(top1_metrics["traj_fde_m"])
                path_error_pct_vals.append(top1_metrics["traj_path_error_pct"])
                fde_ratio_pct_vals.append(top1_metrics["traj_fde_ratio_pct"])

                if self.multi_sample_eval_samples > 1:
                    use_multi_sample_batch = (
                        self.multi_sample_eval_max_batches is None
                        or (batch_idx + 1) <= self.multi_sample_eval_max_batches
                    )
                    if use_multi_sample_batch:
                        pred_action_samples_m = [pred_action_m]
                        for _ in range(1, self.multi_sample_eval_samples):
                            sample_result = policy.predict_action(obs_dict)
                            pred_action_samples_m.append(
                                sample_result["action"] * self.action_scale_to_meter
                            )
                        pred_action_m_all = torch.stack(pred_action_samples_m, dim=0)
                        best_of_n_metrics, _ = self._select_best_of_n(
                            pred_action_m_all=pred_action_m_all,
                            gt_action_m=gt_action_m,
                            start_pos_m=start_pos_m,
                        )
                        top1_subset_ade_vals.append(top1_metrics["traj_ade_m"])
                        top1_subset_fde_vals.append(top1_metrics["traj_fde_m"])
                        top1_subset_traj_mse_norm_vals.append(top1_metrics["traj_mse_norm"])
                        top1_subset_traj_rmse_norm_vals.append(top1_metrics["traj_rmse_norm"])
                        top1_subset_path_error_pct_vals.append(top1_metrics["traj_path_error_pct"])
                        top1_subset_fde_ratio_pct_vals.append(top1_metrics["traj_fde_ratio_pct"])
                        best_of_n_mse_vals.append(best_of_n_metrics["action_mse"])
                        best_of_n_mae_vals.append(best_of_n_metrics["action_mae"])
                        best_of_n_action_rmse_vals.append(best_of_n_metrics["action_rmse"])
                        best_of_n_traj_mse_norm_vals.append(best_of_n_metrics["traj_mse_norm"])
                        best_of_n_traj_rmse_norm_vals.append(best_of_n_metrics["traj_rmse_norm"])
                        best_of_n_ade_vals.append(best_of_n_metrics["traj_ade_m"])
                        best_of_n_fde_vals.append(best_of_n_metrics["traj_fde_m"])
                        best_of_n_path_error_pct_vals.append(best_of_n_metrics["traj_path_error_pct"])
                        best_of_n_fde_ratio_pct_vals.append(best_of_n_metrics["traj_fde_ratio_pct"])
                        best_of_n_batch_count += 1

                if self.max_batches is not None and (batch_idx + 1 >= self.max_batches):
                    break

        if len(mse_vals) == 0:
            metrics = {
                "test/mean_score": 0.0,
                "test/mean_score_path": 0.0,
                "eval_action_mse": float("nan"),
                "eval_action_mae": float("nan"),
                "eval_action_rmse": float("nan"),
                "eval_traj_mse_norm": float("nan"),
                "eval_traj_rmse_norm": float("nan"),
                "eval_traj_ade_m": float("nan"),
                "eval_traj_fde_m": float("nan"),
                "eval_traj_path_error_pct": float("nan"),
                "eval_traj_fde_ratio_pct": float("nan"),
            }
            if best_of_n_batch_count > 0:
                metrics.update(
                    {
                        "eval_best_of_n_action_mse": float("nan"),
                        "eval_best_of_n_action_mae": float("nan"),
                        "eval_best_of_n_action_rmse": float("nan"),
                        "eval_best_of_n_traj_mse_norm": float("nan"),
                        "eval_best_of_n_traj_rmse_norm": float("nan"),
                        "eval_best_of_n_traj_ade_m": float("nan"),
                        "eval_best_of_n_traj_fde_m": float("nan"),
                        "eval_best_of_n_traj_path_error_pct": float("nan"),
                        "eval_best_of_n_traj_fde_ratio_pct": float("nan"),
                        "eval_top1_subset_traj_mse_norm": float("nan"),
                        "eval_top1_subset_traj_rmse_norm": float("nan"),
                        "eval_top1_subset_traj_ade_m": float("nan"),
                        "eval_top1_subset_traj_fde_m": float("nan"),
                        "eval_top1_subset_traj_path_error_pct": float("nan"),
                        "eval_top1_subset_traj_fde_ratio_pct": float("nan"),
                        "eval_best_of_n_gain_traj_path_error_pct": float("nan"),
                        "eval_best_of_n_samples": float(self.multi_sample_eval_samples),
                        "eval_best_of_n_batches": float(best_of_n_batch_count),
                    }
                )
            return metrics

        ade = float(np.mean(ade_vals))
        path_error_pct = float(np.mean(path_error_pct_vals))
        metrics = {
            "test/mean_score": -ade,
            "test/mean_score_path": -path_error_pct,
            "eval_action_mse": float(np.mean(mse_vals)),
            "eval_action_mae": float(np.mean(mae_vals)),
            "eval_action_rmse": float(np.mean(action_rmse_vals)),
            "eval_traj_mse_norm": float(np.mean(traj_mse_norm_vals)),
            "eval_traj_rmse_norm": float(np.mean(traj_rmse_norm_vals)),
            "eval_traj_ade_m": ade,
            "eval_traj_fde_m": float(np.mean(fde_vals)),
            "eval_traj_path_error_pct": path_error_pct,
            "eval_traj_fde_ratio_pct": float(np.mean(fde_ratio_pct_vals)),
        }
        if best_of_n_batch_count > 0:
            top1_subset_path_error_pct = float(np.mean(top1_subset_path_error_pct_vals))
            best_of_n_path_error_pct = float(np.mean(best_of_n_path_error_pct_vals))
            metrics.update(
                {
                    "eval_best_of_n_action_mse": float(np.mean(best_of_n_mse_vals)),
                    "eval_best_of_n_action_mae": float(np.mean(best_of_n_mae_vals)),
                    "eval_best_of_n_action_rmse": float(np.mean(best_of_n_action_rmse_vals)),
                    "eval_best_of_n_traj_mse_norm": float(np.mean(best_of_n_traj_mse_norm_vals)),
                    "eval_best_of_n_traj_rmse_norm": float(np.mean(best_of_n_traj_rmse_norm_vals)),
                    "eval_best_of_n_traj_ade_m": float(np.mean(best_of_n_ade_vals)),
                    "eval_best_of_n_traj_fde_m": float(np.mean(best_of_n_fde_vals)),
                    "eval_best_of_n_traj_path_error_pct": best_of_n_path_error_pct,
                    "eval_best_of_n_traj_fde_ratio_pct": float(np.mean(best_of_n_fde_ratio_pct_vals)),
                    "eval_top1_subset_traj_mse_norm": float(np.mean(top1_subset_traj_mse_norm_vals)),
                    "eval_top1_subset_traj_rmse_norm": float(np.mean(top1_subset_traj_rmse_norm_vals)),
                    "eval_top1_subset_traj_ade_m": float(np.mean(top1_subset_ade_vals)),
                    "eval_top1_subset_traj_fde_m": float(np.mean(top1_subset_fde_vals)),
                    "eval_top1_subset_traj_path_error_pct": top1_subset_path_error_pct,
                    "eval_top1_subset_traj_fde_ratio_pct": float(np.mean(top1_subset_fde_ratio_pct_vals)),
                    "eval_best_of_n_gain_traj_path_error_pct": (
                        top1_subset_path_error_pct - best_of_n_path_error_pct
                    ),
                    "eval_best_of_n_samples": float(self.multi_sample_eval_samples),
                    "eval_best_of_n_batches": float(best_of_n_batch_count),
                }
            )
        return metrics
