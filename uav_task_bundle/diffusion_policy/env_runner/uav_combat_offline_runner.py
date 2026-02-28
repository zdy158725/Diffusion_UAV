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

                pred_action_m = pred_action * self.action_scale_to_meter
                gt_action_m = gt_action * self.action_scale_to_meter
                plot_pred_m = plot_pred_action * self.action_scale_to_meter
                plot_gt_m = plot_gt_action * self.action_scale_to_meter

                mse = torch.nn.functional.mse_loss(pred_action_m, gt_action_m)
                mae = torch.nn.functional.l1_loss(pred_action_m, gt_action_m)
                mse_vals.append(mse.item())
                mae_vals.append(mae.item())
                if self.save_plots and self.plot_action and (plot_data is None):
                    plot_data = (
                        plot_pred_m[: self.plot_num_samples].detach().cpu().numpy(),
                        plot_gt_m[: self.plot_num_samples].detach().cpu().numpy(),
                    )

                if (self.max_batches is not None) and (batch_idx + 1 >= self.max_batches):
                    break

        self.eval_count += 1
        if len(mse_vals) == 0:
            metrics = {
                "test/mean_score": 0.0,
                "eval_action_mse": float("nan"),
                "eval_action_mae": float("nan"),
            }
        else:
            mse = float(np.mean(mse_vals))
            mae = float(np.mean(mae_vals))
            # higher is better for topk manager -> use negative MSE as score
            metrics = {
                "test/mean_score": -mse,
                "eval_action_mse": mse,
                "eval_action_mae": mae,
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
        pred, gt = plot_data
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

        # save raw arrays for inspection
        np.savez_compressed(
            os.path.join(self.plot_dir, f"action_pred_eval{self.eval_count:04d}.npz"),
            pred=pred,
            gt=gt,
        )

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
                ax.plot(df.loc[s.index, "global_step"], -s, label="test_mse_from_score")

        ax.set_xlabel("global_step")
        ax.set_ylabel("value")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "metrics.png"), dpi=150)
        plt.close(fig)
