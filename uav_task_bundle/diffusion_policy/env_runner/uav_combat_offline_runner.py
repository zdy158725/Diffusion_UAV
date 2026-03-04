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
        plot_data = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                obs_dict = {"obs": batch["obs"]}
                gt_action = batch["action"]
                start = policy.n_obs_steps - 1
                end = start + policy.n_action_steps

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
                plot_start_pos = batch["obs"][:, start, 21:24]

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

                mse = torch.nn.functional.mse_loss(pred_action_m, gt_action_m)
                mae = torch.nn.functional.l1_loss(pred_action_m, gt_action_m)
                mse_vals.append(mse.item())
                mae_vals.append(mae.item())
                ade_vals.append(ade.item())
                fde_vals.append(fde.item())
                if self.save_plots and self.plot_action and (plot_data is None):
                    plot_data = {
                        "pred": plot_pred_m[: self.plot_num_samples].detach().cpu().numpy(),
                        "gt": plot_gt_m[: self.plot_num_samples].detach().cpu().numpy(),
                        "start_pos": plot_start_pos_m[: self.plot_num_samples].detach().cpu().numpy(),
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
            }
        else:
            mse = float(np.mean(mse_vals))
            mae = float(np.mean(mae_vals))
            ade = float(np.mean(ade_vals))
            fde = float(np.mean(fde_vals))
            # higher is better for topk manager -> use negative ADE as score
            metrics = {
                "test/mean_score": -ade,
                "eval_action_mse": mse,
                "eval_action_mae": mae,
                "eval_traj_ade_m": ade,
                "eval_traj_fde_m": fde,
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
        if pred.ndim != 3 or gt.ndim != 3:
            return
        n_samples = pred.shape[0]
        t_steps = pred.shape[1]
        dims = pred.shape[2]
        ncols = 3
        nrows = int(math.ceil(dims / ncols))
        x = np.arange(t_steps)

        for s in range(n_samples):
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), sharex=True)
            axes = np.array(axes).reshape(-1)
            for d in range(dims):
                ax = axes[d]
                ax.plot(x, gt[s, :, d], label="gt", linewidth=1.0)
                ax.plot(x, pred[s, :, d], label="pred", linewidth=1.0)
                ax.set_title(f"dim {d}")
                # Use symmetric y-limits around 0 based on data magnitude
                m = max(
                    np.max(np.abs(gt[s, :, d])),
                    np.max(np.abs(pred[s, :, d]))
                )
                if m < 1e-6:
                    m = 1e-3
                ax.set_ylim(-1.2 * m, 1.2 * m)
                ax.set_ylabel("m/step")
                if d == 0:
                    ax.legend(loc="upper right", fontsize=8)
            for d in range(dims, len(axes)):
                axes[d].axis("off")
            fig.suptitle(f"UAV delta-position prediction vs GT (eval {self.eval_count}, sample {s})")
            fig.tight_layout()
            out_path = os.path.join(self.plot_dir, f"action_pred_eval{self.eval_count:04d}_s{s}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            if dims == 3:
                self._save_trajectory3d_plot(
                    start_xyz=start_pos[s],
                    pred_delta=pred[s],
                    gt_delta=gt[s],
                    sample_idx=s,
                )

        # save raw arrays for inspection
        np.savez_compressed(
            os.path.join(self.plot_dir, f"action_pred_eval{self.eval_count:04d}.npz"),
            pred=pred,
            gt=gt,
            start_pos=start_pos,
        )

    def _save_trajectory3d_plot(self, start_xyz, pred_delta, gt_delta, sample_idx):
        start_xyz = np.asarray(start_xyz, dtype=np.float32).reshape(1, 3)
        gt_xyz = start_xyz + np.cumsum(gt_delta, axis=0)
        pred_xyz = start_xyz + np.cumsum(pred_delta, axis=0)
        # Prepend common start point so both curves visibly originate from the same point.
        gt_plot = np.concatenate([start_xyz, gt_xyz], axis=0)
        pred_plot = np.concatenate([start_xyz, pred_xyz], axis=0)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            gt_plot[:, 0], gt_plot[:, 1], gt_plot[:, 2],
            color="tab:blue", linewidth=1.8, label="gt"
        )
        ax.plot(
            pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2],
            color="tab:orange", linewidth=1.8, label="pred"
        )
        ax.scatter(start_xyz[:, 0], start_xyz[:, 1], start_xyz[:, 2], c="k", s=26, label="start")
        ax.scatter(gt_xyz[-1:, 0], gt_xyz[-1:, 1], gt_xyz[-1:, 2], c="tab:blue", s=24, marker="x")
        ax.scatter(pred_xyz[-1:, 0], pred_xyz[-1:, 1], pred_xyz[-1:, 2], c="tab:orange", s=24, marker="x")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(f"3D trajectory (eval {self.eval_count}, sample {sample_idx})")
        ax.legend(loc="upper right", fontsize=8)
        self._set_equal_3d_axes(ax, np.vstack([gt_plot, pred_plot]))
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
