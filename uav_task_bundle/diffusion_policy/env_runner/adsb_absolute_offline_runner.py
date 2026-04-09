import numpy as np
import torch

from diffusion_policy.env_runner.uav_combat_offline_runner import UAVCombatOfflineRunner


class ADSBAbsoluteOfflineRunner(UAVCombatOfflineRunner):
    """
    Offline runner for ADS-B relative-to-current future position targets.
    Expects policy outputs and gt actions to be future xyz relative to the
    current observation point, then reconstructs absolute future xyz for eval.
    """

    def _compute_path_stats(self, pred_action_m, gt_action_m, start_pos_m):
        current_pos_m = start_pos_m
        pred_xyz = current_pos_m[:, None, :] + pred_action_m
        gt_xyz = current_pos_m[:, None, :] + gt_action_m
        gt_step = torch.zeros_like(gt_xyz)
        gt_step[:, 0, :] = gt_xyz[:, 0, :] - current_pos_m
        if gt_xyz.shape[1] > 1:
            gt_step[:, 1:, :] = gt_xyz[:, 1:, :] - gt_xyz[:, :-1, :]
        dist = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)
        gt_path_len = torch.linalg.norm(gt_step, dim=-1).sum(dim=1).clamp(min=1e-6)
        return dist, gt_path_len

    def _select_best_of_n(self, pred_action_m_all, gt_action_m, start_pos_m):
        current_pos_m = start_pos_m
        pred_xyz = current_pos_m[None, :, None, :] + pred_action_m_all
        gt_xyz = current_pos_m[None, :, None, :] + gt_action_m[None, :, :, :]
        dist = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)
        gt_step = torch.zeros_like(gt_action_m)
        gt_step[:, 0, :] = gt_xyz[0, :, 0, :] - current_pos_m
        if gt_action_m.shape[1] > 1:
            gt_step[:, 1:, :] = gt_xyz[0, :, 1:, :] - gt_xyz[0, :, :-1, :]
        gt_path_len = torch.linalg.norm(gt_step, dim=-1).sum(dim=1).clamp(min=1e-6)
        path_error_pct = dist.sum(dim=2) / gt_path_len[None, :] * 100.0
        best_idx = torch.argmin(path_error_pct, dim=0)
        batch_idx = torch.arange(gt_action_m.shape[0], device=gt_action_m.device)
        best_pred_action_m = pred_action_m_all[best_idx, batch_idx]
        best_dist = dist[best_idx, batch_idx]
        best_metrics = self._reduce_metrics(best_pred_action_m, gt_action_m, best_dist, gt_path_len)
        return best_metrics, best_pred_action_m, best_idx

    def _save_action_plot(self, plot_data):
        if self.plot_dir is None:
            return
        if plot_data["pred"].ndim != 3 or plot_data["gt"].ndim != 3:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        pred = plot_data["pred"]
        gt = plot_data["gt"]
        obs_pos_full = plot_data.get("obs_pos_full", None)
        n_obs_steps = int(plot_data.get("n_obs_steps", 1))
        variant_tag = plot_data.get("variant_tag", "pred")
        variant_title = plot_data.get("variant_title", "prediction")
        pred_label = plot_data.get("pred_label", "pred_future")
        choice_idx = plot_data.get("choice_idx", None)
        n_samples = pred.shape[0]
        t_steps = pred.shape[1]
        dims = pred.shape[2]
        ncols = 3
        base_rows = int(np.ceil(dims / ncols))
        pred_x = np.arange(t_steps)

        for s in range(n_samples):
            history = obs_pos_full[s, :n_obs_steps]
            current_pos = history[-1]
            future_gt = current_pos[None, :] + gt[s]
            future_pred = current_pos[None, :] + pred[s]
            history_x = np.arange(history.shape[0])
            future_x = np.arange(history.shape[0], history.shape[0] + t_steps)

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

                ax_future = axes[row, col]
                ax_future.plot(pred_x, future_gt[:, d], label="gt", linewidth=1.0)
                ax_future.plot(pred_x, future_pred[:, d], label="pred", linewidth=1.0)
                ax_future.set_title(f"future abs dim {d}")
                ax_future.set_ylabel("m")
                ax_future.grid(True, alpha=0.25)
                if d == 0:
                    ax_future.legend(loc="upper right", fontsize=8)

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
                    future_gt[:, d],
                    color="tab:blue",
                    linewidth=1.4,
                    label="gt_future" if d == 0 else None,
                )
                ax_abs.plot(
                    future_x,
                    future_pred[:, d],
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
                f"ADSB absolute trajectory ({variant_title}, eval {self.eval_count}, sample {s}{choice_text})"
            )
            fig.tight_layout()
            fig.subplots_adjust(top=0.90)
            out_path = f"{self.plot_dir}/action_{variant_tag}_eval{self.eval_count:04d}_s{s}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

    def _save_trajectory3d_plots(self, plot_data):
        pred = plot_data["pred"]
        gt = plot_data["gt"]
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
            history = obs_pos_full[s, :n_obs_steps]
            current_pos = history[-1]
            future_gt = current_pos[None, :] + gt[s]
            future_pred = current_pos[None, :] + pred[s]
            self._save_trajectory3d_plot(
                history_xyz=history,
                gt_future_xyz=future_gt,
                pred_future_xyz=future_pred,
                sample_idx=s,
                variant_tag=variant_tag,
                variant_title=variant_title,
                pred_label=pred_label,
                choice_idx=None if choice_idx is None else int(choice_idx[s]),
            )
