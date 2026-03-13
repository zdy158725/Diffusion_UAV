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
        plot_curves=True,
        curve_log_path="logs.json.txt",
        curve_keys=None,
        curve_show_test_mse=True,
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
        self.plot_curves = plot_curves
        self.curve_log_path = curve_log_path
        self.action_scale_to_meter = float(getattr(self.dataset, "meters_per_unit", 1.0))
        self.curve_keys = curve_keys or [
            "eval_traj_ade_m",
            "eval_traj_fde_m",
            "eval_traj_ade_ratio_pct",
            "eval_action_mse",
            "eval_action_mae",
            "val_loss",
            "train_action_mse_error",
            "test/mean_score",
        ]
        self.curve_show_test_mse = curve_show_test_mse
        self.eval_count = 0

    def run(self, policy):
        device = self.device if self.device is not None else policy.device
        mse_vals = []
        mae_vals = []
        ade_vals = []
        fde_vals = []
        ade_ratio_pct_vals = []
        plot_data = None

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
                # enemy0 position in obs: own(3*7=21 dims) then enemy0 xyz
                plot_start_pos = batch["obs"][:, anchor_idx, 21:24]

                pred_action_m = pred_action * self.action_scale_to_meter
                gt_action_m = gt_action * self.action_scale_to_meter
                plot_pred_m = plot_pred_action * self.action_scale_to_meter
                plot_gt_m = plot_gt_action * self.action_scale_to_meter
                plot_start_pos_m = plot_start_pos * self.action_scale_to_meter

                # Position-space trajectory error (meters)
                pred_xyz = plot_start_pos_m[:, None, :] + torch.cumsum(plot_pred_m, dim=1)
                gt_xyz = plot_start_pos_m[:, None, :] + torch.cumsum(plot_gt_m, dim=1)
                dist = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)
                ade = dist.mean()
                fde = dist[:, -1].mean()
                # Relative error percentage: (sum of step errors / GT trajectory length) * 100
                sample_total_err = dist.sum(dim=1)
                gt_traj_len = torch.linalg.norm(plot_gt_m, dim=-1).sum(dim=1).clamp(min=1e-6)
                ade_ratio_pct = (sample_total_err / gt_traj_len * 100.0).mean()

                mse = torch.nn.functional.mse_loss(pred_action_m, gt_action_m)
                mae = torch.nn.functional.l1_loss(pred_action_m, gt_action_m)
                mse_vals.append(mse.item())
                mae_vals.append(mae.item())
                ade_vals.append(ade.item())
                fde_vals.append(fde.item())
                ade_ratio_pct_vals.append(ade_ratio_pct.item())
                if self.save_plots and self.plot_action and (plot_data is None):
                    plot_data = {
                        "pred": plot_pred_m[: self.plot_num_samples].detach().cpu().numpy(),
                        "gt": plot_gt_m[: self.plot_num_samples].detach().cpu().numpy(),
                        "start_pos": plot_start_pos_m[: self.plot_num_samples].detach().cpu().numpy(),
                        "obs_pos_full": (
                            batch["obs"][: self.plot_num_samples, :, 21:24]
                            * self.action_scale_to_meter
                        ).detach().cpu().numpy(),
                        "n_obs_steps": int(policy.n_obs_steps),
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
                "eval_traj_ade_ratio_pct": float("nan"),
            }
        else:
            mse = float(np.mean(mse_vals))
            mae = float(np.mean(mae_vals))
            ade = float(np.mean(ade_vals))
            fde = float(np.mean(fde_vals))
            ade_ratio_pct = float(np.mean(ade_ratio_pct_vals))
            # higher is better for topk manager -> use negative ADE as score
            metrics = {
                "test/mean_score": -ade,
                "eval_action_mse": mse,
                "eval_action_mae": mae,
                "eval_traj_ade_m": ade,
                "eval_traj_fde_m": fde,
                "eval_traj_ade_ratio_pct": ade_ratio_pct,
            }

        if self.save_plots and (self.eval_count % self.plot_interval == 0):
            os.makedirs(self.plot_dir, exist_ok=True)
            if self.plot_action and (plot_data is not None):
                self._save_action_plot(plot_data)
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
                    label="pred_future" if d == 0 else None,
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
            fig.suptitle(
                f"UAV delta + full absolute trajectory (eval {self.eval_count}, sample {s})"
            )
            fig.tight_layout()
            fig.subplots_adjust(top=0.90)
            out_path = os.path.join(self.plot_dir, f"action_pred_eval{self.eval_count:04d}_s{s}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            if dims == 3:
                self._save_trajectory3d_plot(
                    history_xyz=history,
                    gt_future_xyz=future_gt,
                    pred_future_xyz=pred_future,
                    sample_idx=s,
                )

        # save raw arrays for inspection
        np.savez_compressed(
            os.path.join(self.plot_dir, f"action_pred_eval{self.eval_count:04d}.npz"),
            pred=pred,
            gt=gt,
            start_pos=start_pos,
            obs_pos_full=obs_pos_full,
        )

    def _save_trajectory3d_plot(self, history_xyz, gt_future_xyz, pred_future_xyz, sample_idx):
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
            color="tab:orange", linewidth=1.8, label="pred_future"
        )
        ax.scatter(start_xyz[:, 0], start_xyz[:, 1], start_xyz[:, 2], c="k", s=26, label="current")
        ax.scatter(gt_plot[-1:, 0], gt_plot[-1:, 1], gt_plot[-1:, 2], c="tab:blue", s=24, marker="x")
        ax.scatter(pred_plot[-1:, 0], pred_plot[-1:, 1], pred_plot[-1:, 2], c="tab:orange", s=24, marker="x")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"3D history + future trajectory (eval {self.eval_count}, sample {sample_idx})")
        ax.legend(loc="upper right", fontsize=8)
        self._set_equal_3d_axes(ax, np.vstack([history_plot, gt_plot, pred_plot]))
        fig.tight_layout()
        out_path = os.path.join(self.plot_dir, f"trajectory3d_eval{self.eval_count:04d}_s{sample_idx}.png")
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

    def _save_curve_plot(self):
        if plt is None:
            return
        log_path = os.path.join(self.output_dir, self.curve_log_path)
        if not os.path.isfile(log_path):
            return
        df = read_json_log(log_path, required_keys=self.curve_keys)
        if df.empty or "global_step" not in df:
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        for k in self.curve_keys:
            if k in df:
                s = df[k].dropna()
                if len(s) == 0:
                    continue
                ax.plot(df.loc[s.index, "global_step"], s, label=k)

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
