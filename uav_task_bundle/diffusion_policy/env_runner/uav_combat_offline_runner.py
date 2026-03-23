import os
import math
import numpy as np
import torch
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig, ListConfig
from diffusion_policy.common.json_logger import read_json_log

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


class UAVCombatOfflineRunner(BaseLowdimRunner):
    """
    Offline evaluation for UAV combat dataset.
    Reports action prediction MSE/MAE on validation set.
    """

    def __init__(
        self,
        output_dir,
        dataset,
        batch_size=256,
        num_workers=0,
        pin_memory=True,
        device=None,
        max_batches=None,
        save_plots=True,
        plot_dir="plots",
        plot_interval=1,
        plot_num_samples=2,
        plot_action=True,
        plot_trajectory3d=True,
        plot_curves=True,
        curve_log_path="logs.json.txt",
        curve_keys=None,
        curve_show_test_mse=True,
        multi_sample_eval_samples=1,
        multi_sample_eval_max_batches=None,
    ):
        super().__init__(output_dir)
        # instantiate dataset from config if needed
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
            collate_fn=self.val_dataset.get_collate_fn(),
        )
        self.device = torch.device(device) if device is not None else None
        self.max_batches = max_batches
        self.save_plots = save_plots
        self.plot_dir = os.path.join(self.output_dir, plot_dir)
        self.plot_interval = max(int(plot_interval), 1)
        self.plot_num_samples = max(int(plot_num_samples), 1)
        self.plot_action = plot_action
        self.plot_trajectory3d = plot_trajectory3d
        self.plot_curves = plot_curves
        self.curve_log_path = curve_log_path
        self.action_scale_to_meter = float(getattr(self.dataset, "meters_per_unit", 1.0))
        self.position_slice = getattr(self.dataset, "position_slice", slice(0, 3))
        self.curve_keys = curve_keys or [
            "eval_traj_ade_m",
            "eval_traj_fde_m",
            "eval_traj_path_error_pct",
            "eval_traj_fde_ratio_pct",
            "eval_top1_subset_traj_path_error_pct",
            "eval_top1_subset_traj_fde_ratio_pct",
            "eval_best_of_n_traj_ade_m",
            "eval_best_of_n_traj_fde_m",
            "eval_best_of_n_traj_path_error_pct",
            "eval_best_of_n_traj_fde_ratio_pct",
            "eval_best_of_n_gain_traj_path_error_pct",
            "eval_action_mse",
            "eval_action_mae",
            "val_loss",
            "train_action_mse_error",
            "test/mean_score",
        ]
        self.curve_show_test_mse = curve_show_test_mse
        self.multi_sample_eval_samples = max(int(multi_sample_eval_samples), 1)
        self.multi_sample_eval_max_batches = (
            None if multi_sample_eval_max_batches is None
            else max(int(multi_sample_eval_max_batches), 1)
        )
        self.eval_count = 0

    def _compute_path_stats(self, pred_action_m, gt_action_m, start_pos_m):
        pred_xyz = start_pos_m[:, None, :] + torch.cumsum(pred_action_m, dim=1)
        gt_xyz = start_pos_m[:, None, :] + torch.cumsum(gt_action_m, dim=1)
        dist = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)
        gt_path_len = torch.linalg.norm(gt_action_m, dim=-1).sum(dim=1).clamp(min=1e-6)
        return dist, gt_path_len

    def _reduce_metrics(self, pred_action_m, gt_action_m, dist, gt_path_len):
        mse = torch.nn.functional.mse_loss(pred_action_m, gt_action_m)
        mae = torch.nn.functional.l1_loss(pred_action_m, gt_action_m)
        ade = dist.mean()
        fde = dist[:, -1].mean()
        path_error_pct = (dist.sum(dim=1) / gt_path_len * 100.0).mean()
        fde_ratio_pct = (dist[:, -1] / gt_path_len * 100.0).mean()
        return {
            "action_mse": mse.item(),
            "action_mae": mae.item(),
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
        best_dist = dist[best_idx, batch_idx]
        best_metrics = self._reduce_metrics(best_pred_action_m, gt_action_m, best_dist, gt_path_len)
        return best_metrics, best_pred_action_m, best_idx

    def run(self, policy):
        device = self.device if self.device is not None else policy.device
        mse_vals = []
        mae_vals = []
        ade_vals = []
        fde_vals = []
        path_error_pct_vals = []
        fde_ratio_pct_vals = []
        top1_subset_ade_vals = []
        top1_subset_fde_vals = []
        top1_subset_path_error_pct_vals = []
        top1_subset_fde_ratio_pct_vals = []
        best_of_n_mse_vals = []
        best_of_n_mae_vals = []
        best_of_n_ade_vals = []
        best_of_n_fde_vals = []
        best_of_n_path_error_pct_vals = []
        best_of_n_fde_ratio_pct_vals = []
        best_of_n_batch_count = 0
        plot_data_top1 = None
        plot_data_best_of_n = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                obs_dict = {"obs": batch["obs"]}
                gt_action = batch["action"]
                start, end = policy.get_action_window_indices()
                anchor_idx = policy.get_action_anchor_obs_index()

                result = policy.predict_action(obs_dict)
                if policy.pred_action_steps_only:
                    pred_action = result["action"]
                    gt_action = gt_action[:, start:end]
                else:
                    pred_action = result["action_pred"]

                # For plotting, always show execution window only.
                plot_pred_action = result["action"]
                plot_gt_action = batch["action"][:, start:end]
                plot_start_pos = batch["obs"][:, anchor_idx, self.position_slice]

                pred_action_m = pred_action * self.action_scale_to_meter
                gt_action_m = gt_action * self.action_scale_to_meter
                plot_pred_m = plot_pred_action * self.action_scale_to_meter
                plot_gt_m = plot_gt_action * self.action_scale_to_meter
                plot_start_pos_m = plot_start_pos * self.action_scale_to_meter

                dist, gt_traj_len = self._compute_path_stats(
                    pred_action_m=plot_pred_m,
                    gt_action_m=plot_gt_m,
                    start_pos_m=plot_start_pos_m,
                )
                top1_metrics = self._reduce_metrics(
                    pred_action_m=plot_pred_m,
                    gt_action_m=plot_gt_m,
                    dist=dist,
                    gt_path_len=gt_traj_len,
                )
                mse_vals.append(top1_metrics["action_mse"])
                mae_vals.append(top1_metrics["action_mae"])
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
                        pred_action_samples_m = [plot_pred_m]
                        for _ in range(1, self.multi_sample_eval_samples):
                            sample_result = policy.predict_action(obs_dict)
                            pred_action_samples_m.append(
                                sample_result["action"] * self.action_scale_to_meter
                            )

                        pred_action_m_all = torch.stack(pred_action_samples_m, dim=0)
                        best_of_n_metrics, best_pred_action_m, best_idx = self._select_best_of_n(
                            pred_action_m_all=pred_action_m_all,
                            gt_action_m=plot_gt_m,
                            start_pos_m=plot_start_pos_m,
                        )
                        top1_subset_ade_vals.append(top1_metrics["traj_ade_m"])
                        top1_subset_fde_vals.append(top1_metrics["traj_fde_m"])
                        top1_subset_path_error_pct_vals.append(top1_metrics["traj_path_error_pct"])
                        top1_subset_fde_ratio_pct_vals.append(top1_metrics["traj_fde_ratio_pct"])
                        best_of_n_mse_vals.append(best_of_n_metrics["action_mse"])
                        best_of_n_mae_vals.append(best_of_n_metrics["action_mae"])
                        best_of_n_ade_vals.append(best_of_n_metrics["traj_ade_m"])
                        best_of_n_fde_vals.append(best_of_n_metrics["traj_fde_m"])
                        best_of_n_path_error_pct_vals.append(best_of_n_metrics["traj_path_error_pct"])
                        best_of_n_fde_ratio_pct_vals.append(best_of_n_metrics["traj_fde_ratio_pct"])
                        best_of_n_batch_count += 1

                        if self.save_plots and (self.plot_action or self.plot_trajectory3d) and (plot_data_best_of_n is None):
                            plot_data_best_of_n = {
                                "pred": best_pred_action_m[: self.plot_num_samples].detach().cpu().numpy(),
                                "gt": plot_gt_m[: self.plot_num_samples].detach().cpu().numpy(),
                                "start_pos": plot_start_pos_m[: self.plot_num_samples].detach().cpu().numpy(),
                                "obs_pos_full": (
                                    batch["obs"][: self.plot_num_samples, :, self.position_slice]
                                    * self.action_scale_to_meter
                                ).detach().cpu().numpy(),
                                "n_obs_steps": int(policy.n_obs_steps),
                                "variant_tag": f"bestof{self.multi_sample_eval_samples}",
                                "variant_title": f"best-of-{self.multi_sample_eval_samples}",
                                "pred_label": f"best_of_{self.multi_sample_eval_samples}",
                                "choice_idx": best_idx[: self.plot_num_samples].detach().cpu().numpy(),
                            }

                if self.save_plots and (self.plot_action or self.plot_trajectory3d) and (plot_data_top1 is None):
                    plot_data_top1 = {
                        "pred": plot_pred_m[: self.plot_num_samples].detach().cpu().numpy(),
                        "gt": plot_gt_m[: self.plot_num_samples].detach().cpu().numpy(),
                        "start_pos": plot_start_pos_m[: self.plot_num_samples].detach().cpu().numpy(),
                        "obs_pos_full": (
                            batch["obs"][: self.plot_num_samples, :, self.position_slice]
                            * self.action_scale_to_meter
                        ).detach().cpu().numpy(),
                        "n_obs_steps": int(policy.n_obs_steps),
                        "variant_tag": "top1",
                        "variant_title": "top-1",
                        "pred_label": "top1_pred",
                    }

                if (self.max_batches is not None) and (batch_idx + 1 >= self.max_batches):
                    break

        self.eval_count += 1
        if len(mse_vals) == 0:
            metrics = {
                "test/mean_score": 0.0,
                "eval_action_mse": float("nan"),
                "eval_action_mae": float("nan"),
                "eval_traj_ade_m": float("nan"),
                "eval_traj_fde_m": float("nan"),
                "eval_traj_path_error_pct": float("nan"),
                "eval_traj_fde_ratio_pct": float("nan"),
                "eval_traj_ade_ratio_pct": float("nan"),
            }
            if best_of_n_batch_count > 0:
                metrics.update({
                    "eval_best_of_n_action_mse": float("nan"),
                    "eval_best_of_n_action_mae": float("nan"),
                    "eval_best_of_n_traj_ade_m": float("nan"),
                    "eval_best_of_n_traj_fde_m": float("nan"),
                    "eval_best_of_n_traj_path_error_pct": float("nan"),
                    "eval_best_of_n_traj_fde_ratio_pct": float("nan"),
                    "eval_top1_subset_traj_ade_m": float("nan"),
                    "eval_top1_subset_traj_fde_m": float("nan"),
                    "eval_top1_subset_traj_path_error_pct": float("nan"),
                    "eval_top1_subset_traj_fde_ratio_pct": float("nan"),
                    "eval_best_of_n_gain_traj_path_error_pct": float("nan"),
                    "eval_best_of_n_samples": float(self.multi_sample_eval_samples),
                    "eval_best_of_n_batches": float(best_of_n_batch_count),
                })
        else:
            mse = float(np.mean(mse_vals))
            mae = float(np.mean(mae_vals))
            ade = float(np.mean(ade_vals))
            fde = float(np.mean(fde_vals))
            path_error_pct = float(np.mean(path_error_pct_vals))
            fde_ratio_pct = float(np.mean(fde_ratio_pct_vals))
            # higher is better for topk manager -> use negative ADE as score
            metrics = {
                "test/mean_score": -ade,
                "eval_action_mse": mse,
                "eval_action_mae": mae,
                "eval_traj_ade_m": ade,
                "eval_traj_fde_m": fde,
                "eval_traj_path_error_pct": path_error_pct,
                "eval_traj_fde_ratio_pct": fde_ratio_pct,
                # Backward-compatible alias for old logs and analysis scripts.
                "eval_traj_ade_ratio_pct": path_error_pct,
            }
            if best_of_n_batch_count > 0:
                top1_subset_path_error_pct = float(np.mean(top1_subset_path_error_pct_vals))
                best_of_n_path_error_pct = float(np.mean(best_of_n_path_error_pct_vals))
                metrics.update({
                    "eval_best_of_n_action_mse": float(np.mean(best_of_n_mse_vals)),
                    "eval_best_of_n_action_mae": float(np.mean(best_of_n_mae_vals)),
                    "eval_best_of_n_traj_ade_m": float(np.mean(best_of_n_ade_vals)),
                    "eval_best_of_n_traj_fde_m": float(np.mean(best_of_n_fde_vals)),
                    "eval_best_of_n_traj_path_error_pct": best_of_n_path_error_pct,
                    "eval_best_of_n_traj_fde_ratio_pct": float(np.mean(best_of_n_fde_ratio_pct_vals)),
                    "eval_top1_subset_traj_ade_m": float(np.mean(top1_subset_ade_vals)),
                    "eval_top1_subset_traj_fde_m": float(np.mean(top1_subset_fde_vals)),
                    "eval_top1_subset_traj_path_error_pct": top1_subset_path_error_pct,
                    "eval_top1_subset_traj_fde_ratio_pct": float(np.mean(top1_subset_fde_ratio_pct_vals)),
                    "eval_best_of_n_gain_traj_path_error_pct": (
                        top1_subset_path_error_pct - best_of_n_path_error_pct
                    ),
                    "eval_best_of_n_samples": float(self.multi_sample_eval_samples),
                    "eval_best_of_n_batches": float(best_of_n_batch_count),
                })

        if self.save_plots and (self.eval_count % self.plot_interval == 0):
            os.makedirs(self.plot_dir, exist_ok=True)
            if plot_data_top1 is not None:
                if self.plot_action:
                    self._save_action_plot(plot_data_top1)
                if self.plot_trajectory3d:
                    self._save_trajectory3d_plots(plot_data_top1)
            if plot_data_best_of_n is not None:
                if self.plot_action:
                    self._save_action_plot(plot_data_best_of_n)
                if self.plot_trajectory3d:
                    self._save_trajectory3d_plots(plot_data_best_of_n)
            if self.plot_curves:
                self._save_curve_plot()

        return metrics

    def _save_action_plot(self, plot_data):
        if plt is None:
            return
        pred = plot_data["pred"]
        gt = plot_data["gt"]
        start_pos = plot_data["start_pos"]
        obs_pos_full = plot_data.get("obs_pos_full", None)
        n_obs_steps = int(plot_data.get("n_obs_steps", 1))
        variant_tag = plot_data.get("variant_tag", "pred")
        variant_title = plot_data.get("variant_title", "prediction")
        pred_label = plot_data.get("pred_label", "pred_future")
        choice_idx = plot_data.get("choice_idx", None)
        if pred.ndim != 3 or gt.ndim != 3:
            return
        n_samples = pred.shape[0]
        t_steps = pred.shape[1]
        dims = pred.shape[2]
        ncols = 3
        base_rows = int(math.ceil(dims / ncols))
        x_delta = np.arange(t_steps)

        for s in range(n_samples):
            if obs_pos_full is not None:
                history = obs_pos_full[s, :n_obs_steps]
            else:
                history = start_pos[s : s + 1]
            future_gt = None
            if obs_pos_full is not None:
                future_end = min(n_obs_steps + t_steps, obs_pos_full.shape[1])
                future_gt = obs_pos_full[s, n_obs_steps:future_end]
            pred_future = history[-1:] + np.cumsum(pred[s], axis=0)
            if future_gt is None or future_gt.shape[0] != t_steps:
                future_gt = history[-1:] + np.cumsum(gt[s], axis=0)

            history_x = np.arange(history.shape[0])
            gt_future_plot = np.concatenate([history[-1:], future_gt], axis=0)
            pred_future_plot = np.concatenate([history[-1:], pred_future], axis=0)
            future_x = np.arange(
                history.shape[0] - 1,
                history.shape[0] - 1 + gt_future_plot.shape[0],
            )

            fig, axes = plt.subplots(
                base_rows * 2,
                ncols,
                figsize=(ncols * 3.2, base_rows * 5.0),
                sharex=False,
            )
            axes = np.array(axes).reshape(base_rows * 2, ncols)
            for d in range(dims):
                row = d // ncols
                col = d % ncols

                ax_delta = axes[row, col]
                ax_delta.plot(x_delta, gt[s, :, d], label="gt", linewidth=1.0)
                ax_delta.plot(x_delta, pred[s, :, d], label="pred", linewidth=1.0)
                ax_delta.set_title(f"delta dim {d}")
                m = max(
                    np.max(np.abs(gt[s, :, d])),
                    np.max(np.abs(pred[s, :, d]))
                )
                if m < 1e-6:
                    m = 1e-3
                ax_delta.set_ylim(-1.2 * m, 1.2 * m)
                ax_delta.set_ylabel("m/step")
                ax_delta.grid(True, alpha=0.25)
                if d == 0:
                    ax_delta.legend(loc="upper right", fontsize=8)

                ax_abs = axes[base_rows + row, col]
                ax_abs.plot(
                    history_x,
                    history[:, d],
                    color="tab:blue",
                    linestyle="--",
                    linewidth=1.0,
                    label="history" if d == 0 else None,
                )
                ax_abs.plot(
                    future_x,
                    gt_future_plot[:, d],
                    color="tab:blue",
                    linewidth=1.4,
                    label="gt_future" if d == 0 else None,
                )
                ax_abs.plot(
                    future_x,
                    pred_future_plot[:, d],
                    color="tab:orange",
                    linewidth=1.4,
                    label=pred_label if d == 0 else None,
                )
                ax_abs.scatter(
                    [len(history) - 1],
                    [history[-1, d]],
                    color="k",
                    s=12,
                    label="current" if d == 0 else None,
                    zorder=3,
                )
                ax_abs.set_title(f"absolute dim {d}")
                abs_min = min(
                    np.min(history[:, d]),
                    np.min(gt_future_plot[:, d]),
                    np.min(pred_future_plot[:, d]),
                )
                abs_max = max(
                    np.max(history[:, d]),
                    np.max(gt_future_plot[:, d]),
                    np.max(pred_future_plot[:, d]),
                )
                span = max(abs_max - abs_min, 1e-3)
                pad = 0.1 * span
                ax_abs.set_ylim(abs_min - pad, abs_max + pad)
                ax_abs.set_ylabel("m")
                ax_abs.grid(True, alpha=0.25)
                if d == 0:
                    ax_abs.legend(loc="upper right", fontsize=8)

            for d in range(dims, base_rows * ncols):
                row = d // ncols
                col = d % ncols
                axes[row, col].axis("off")
                axes[base_rows + row, col].axis("off")
            choice_text = ""
            if choice_idx is not None:
                choice_text = f", choice {int(choice_idx[s])}"
            fig.suptitle(
                f"UAV delta + full absolute trajectory ({variant_title}, eval {self.eval_count}, sample {s}{choice_text})"
            )
            fig.tight_layout()
            fig.subplots_adjust(top=0.90)
            out_path = os.path.join(
                self.plot_dir,
                f"action_{variant_tag}_eval{self.eval_count:04d}_s{s}.png",
            )
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

        # save raw arrays for inspection
        np.savez_compressed(
            os.path.join(self.plot_dir, f"action_{variant_tag}_eval{self.eval_count:04d}.npz"),
            pred=pred,
            gt=gt,
            start_pos=start_pos,
            obs_pos_full=obs_pos_full,
            choice_idx=choice_idx,
        )

    def _save_trajectory3d_plots(self, plot_data):
        if plt is None:
            return
        pred = plot_data["pred"]
        gt = plot_data["gt"]
        start_pos = plot_data["start_pos"]
        obs_pos_full = plot_data.get("obs_pos_full", None)
        n_obs_steps = int(plot_data.get("n_obs_steps", 1))
        variant_tag = plot_data.get("variant_tag", "pred")
        variant_title = plot_data.get("variant_title", "prediction")
        pred_label = plot_data.get("pred_label", "pred_future")
        choice_idx = plot_data.get("choice_idx", None)
        if pred.ndim != 3 or gt.ndim != 3 or pred.shape[2] != 3:
            return

        n_samples = pred.shape[0]
        for s in range(n_samples):
            if obs_pos_full is not None:
                history = obs_pos_full[s, :n_obs_steps]
            else:
                history = start_pos[s : s + 1]
            if history.shape[-1] != 3:
                continue

            future_gt = None
            if obs_pos_full is not None:
                future_end = min(n_obs_steps + pred.shape[1], obs_pos_full.shape[1])
                future_gt = obs_pos_full[s, n_obs_steps:future_end]
            pred_future = history[-1:] + np.cumsum(pred[s], axis=0)
            if future_gt is None or future_gt.shape[0] != pred.shape[1]:
                future_gt = history[-1:] + np.cumsum(gt[s], axis=0)

            self._save_trajectory3d_plot(
                history_xyz=history,
                gt_future_xyz=future_gt,
                pred_future_xyz=pred_future,
                sample_idx=s,
                variant_tag=variant_tag,
                variant_title=variant_title,
                pred_label=pred_label,
                choice_idx=None if choice_idx is None else int(choice_idx[s]),
            )

    def _save_trajectory3d_plot(
        self,
        history_xyz,
        gt_future_xyz,
        pred_future_xyz,
        sample_idx,
        variant_tag="pred",
        variant_title="prediction",
        pred_label="pred_future",
        choice_idx=None,
    ):
        history_plot = np.asarray(history_xyz, dtype=np.float32)
        gt_future_plot = np.asarray(gt_future_xyz, dtype=np.float32)
        pred_future_plot = np.asarray(pred_future_xyz, dtype=np.float32)
        start_xyz = history_plot[-1:]
        gt_plot = np.concatenate([start_xyz, gt_future_plot], axis=0)
        pred_plot = np.concatenate([start_xyz, pred_future_plot], axis=0)
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            history_plot[:, 0], history_plot[:, 1], history_plot[:, 2],
            color="tab:blue", linewidth=1.2, linestyle="--", label="history"
        )
        ax.plot(
            gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2],
            color="tab:blue", linewidth=1.8, label="gt_future"
        )
        ax.plot(
            pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2],
            color="tab:orange", linewidth=1.8, label=pred_label
        )
        ax.scatter(start_xyz[:, 0], start_xyz[:, 1], start_xyz[:, 2], c="k", s=26, label="current")
        ax.scatter(gt_plot[-1:, 0], gt_plot[-1:, 1], gt_plot[-1:, 2], c="tab:blue", s=24, marker="x")
        ax.scatter(pred_plot[-1:, 0], pred_plot[-1:, 1], pred_plot[-1:, 2], c="tab:orange", s=24, marker="x")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        choice_text = ""
        if choice_idx is not None:
            choice_text = f", choice {choice_idx}"
        ax.set_title(
            f"3D history + future trajectory ({variant_title}, eval {self.eval_count}, sample {sample_idx}{choice_text})"
        )
        ax.legend(loc="upper right", fontsize=8)
        self._set_equal_3d_axes(ax, np.vstack([history_plot, gt_plot, pred_plot]))
        fig.tight_layout()
        out_path = os.path.join(
            self.plot_dir,
            f"trajectory3d_{variant_tag}_eval{self.eval_count:04d}_s{sample_idx}.png",
        )
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _set_equal_3d_axes(ax, pts):
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = float(np.max(maxs - mins) * 0.55)
        if radius < 1e-3:
            radius = 1e-3
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def _plot_curve_group(self, ax, df, keys):
        plotted = False
        for k in keys:
            if k not in df:
                continue
            s = df[k].dropna()
            if len(s) == 0:
                continue
            ax.plot(df.loc[s.index, "global_step"], s, label=k)
            plotted = True
        return plotted

    def _save_curve_plot(self):
        if plt is None:
            return
        log_path = os.path.join(self.output_dir, self.curve_log_path)
        if not os.path.isfile(log_path):
            return
        df = read_json_log(log_path, required_keys=self.curve_keys)
        if df.empty or "global_step" not in df:
            return
        if "eval_traj_path_error_pct" not in df and "eval_traj_ade_ratio_pct" in df:
            df["eval_traj_path_error_pct"] = df["eval_traj_ade_ratio_pct"]

        main_keys = [
            "eval_traj_ade_m",
            "eval_traj_fde_m",
            "eval_traj_path_error_pct",
            "eval_traj_fde_ratio_pct",
            "eval_action_mse",
            "eval_action_mae",
            "val_loss",
            "train_action_mse_error",
        ]
        fig, ax = plt.subplots(figsize=(8, 4))
        self._plot_curve_group(ax, df, main_keys)

        if self.curve_show_test_mse and ("test/mean_score" in df):
            s = df["test/mean_score"].dropna()
            if len(s) > 0:
                ax.plot(df.loc[s.index, "global_step"], -s, label="test_metric_from_score")

        ax.set_xlabel("global_step")
        ax.set_ylabel("value")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "metrics.png"), dpi=150)
        plt.close(fig)

        best_of_n_error_keys = [
            "eval_top1_subset_traj_path_error_pct",
            "eval_best_of_n_traj_path_error_pct",
            "eval_best_of_n_gain_traj_path_error_pct",
        ]
        best_of_n_ratio_keys = [
            "eval_top1_subset_traj_fde_ratio_pct",
            "eval_best_of_n_traj_fde_ratio_pct",
        ]
        has_best_of_n = any(
            (k in df) and (len(df[k].dropna()) > 0)
            for k in (best_of_n_error_keys + best_of_n_ratio_keys)
        )
        if not has_best_of_n:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plotted_error = self._plot_curve_group(axes[0], df, best_of_n_error_keys)
        plotted_ratio = self._plot_curve_group(axes[1], df, best_of_n_ratio_keys)

        axes[0].set_xlabel("global_step")
        axes[0].set_ylabel("path_error_pct")
        axes[0].set_title("best-of-n path error")
        if plotted_error:
            axes[0].legend(fontsize=8)

        axes[1].set_xlabel("global_step")
        axes[1].set_ylabel("fde_ratio_pct")
        axes[1].set_title("best-of-n ratio")
        if plotted_ratio:
            axes[1].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "best_of_n_metrics.png"), dpi=150)
        plt.close(fig)
