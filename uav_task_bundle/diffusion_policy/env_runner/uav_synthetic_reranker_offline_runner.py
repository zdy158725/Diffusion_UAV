import os

import torch

from diffusion_policy.env_runner.uav_combat_offline_runner import UAVCombatOfflineRunner


class UAVSyntheticRerankerOfflineRunner(UAVCombatOfflineRunner):
    def __init__(
        self,
        *args,
        candidate_selection_mode="top1",
        selection_num_candidates=None,
        reranker_checkpoint=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        valid_modes = {"top1", "oracle_best_of_n", "reranker_best_of_n"}
        if candidate_selection_mode not in valid_modes:
            raise ValueError(
                f"candidate_selection_mode must be one of {sorted(valid_modes)}, "
                f"got {candidate_selection_mode}"
            )
        self.candidate_selection_mode = candidate_selection_mode
        self.selection_num_candidates = (
            None if selection_num_candidates is None else max(int(selection_num_candidates), 1)
        )
        self.reranker_checkpoint = reranker_checkpoint
        self.reranker = None
        extra_curve_keys = [
            "eval_top1_traj_ade_m",
            "eval_top1_traj_fde_m",
            "eval_top1_traj_path_error_pct",
            "eval_reranker_traj_ade_m",
            "eval_reranker_traj_fde_m",
            "eval_reranker_traj_path_error_pct",
            "eval_reranker_gain_traj_path_error_pct",
        ]
        for key in extra_curve_keys:
            if key not in self.curve_keys:
                self.curve_keys.append(key)

    def _resolve_candidate_count(self):
        if self.candidate_selection_mode == "top1":
            return self.multi_sample_eval_samples
        if self.selection_num_candidates is not None:
            return self.selection_num_candidates
        return max(self.multi_sample_eval_samples, 1)

    def _compute_candidate_path_stats(self, pred_action_m_all, gt_action_m, start_pos_m):
        pred_xyz = start_pos_m[None, :, None, :] + torch.cumsum(pred_action_m_all, dim=2)
        gt_xyz = start_pos_m[:, None, :] + torch.cumsum(gt_action_m, dim=1)
        gt_xyz = gt_xyz[None, :, :, :]
        dist = torch.linalg.norm(pred_xyz - gt_xyz, dim=-1)
        gt_path_len = torch.linalg.norm(gt_action_m, dim=-1).sum(dim=1).clamp(min=1e-6)
        return dist, gt_path_len

    @staticmethod
    def _gather_candidates(candidates, choice_idx):
        batch_idx = torch.arange(candidates.shape[1], device=candidates.device)
        return candidates[choice_idx, batch_idx]

    def _reduce_selected_candidate_metrics(
        self,
        pred_action_m_all,
        gt_action_m,
        dist,
        gt_path_len,
        choice_idx,
    ):
        pred_action_m = self._gather_candidates(pred_action_m_all, choice_idx)
        selected_dist = self._gather_candidates(dist, choice_idx)
        metrics = self._reduce_metrics(pred_action_m, gt_action_m, selected_dist, gt_path_len)
        return metrics, pred_action_m

    def _select_best_of_n_from_stats(self, pred_action_m_all, gt_action_m, dist, gt_path_len):
        path_error_pct = dist.sum(dim=2) / gt_path_len[None, :] * 100.0
        best_idx = torch.argmin(path_error_pct, dim=0)
        metrics, pred_action_m = self._reduce_selected_candidate_metrics(
            pred_action_m_all=pred_action_m_all,
            gt_action_m=gt_action_m,
            dist=dist,
            gt_path_len=gt_path_len,
            choice_idx=best_idx,
        )
        return metrics, pred_action_m, best_idx

    def _load_reranker(self, device):
        if self.reranker is None:
            if self.reranker_checkpoint is None:
                raise ValueError("reranker_checkpoint is required for reranker_best_of_n mode.")
            from diffusion_policy.workspace.train_trajectory_reranker_workspace import (
                TrainTrajectoryRerankerWorkspace,
            )

            workspace = TrainTrajectoryRerankerWorkspace.create_from_checkpoint(
                path=os.path.expanduser(self.reranker_checkpoint)
            )
            self.reranker = workspace.model
        self.reranker.to(device)
        self.reranker.eval()
        return self.reranker

    def _select_reranker_best_of_n(self, reranker, obs_hist, pred_action_all):
        cand_action = pred_action_all.transpose(0, 1)
        cand_rel_pos = torch.cumsum(cand_action, dim=2)
        reranker_result = reranker.select_best(
            obs_hist=obs_hist,
            cand_action=cand_action,
            cand_rel_pos=cand_rel_pos,
        )
        return reranker_result["best_idx"]

    def run(self, policy):
        device = self.device if self.device is not None else policy.device
        candidate_count = self._resolve_candidate_count()
        if self.candidate_selection_mode != "top1" and candidate_count <= 1:
            raise ValueError("selection_num_candidates must be > 1 for non-top1 selection modes.")

        mse_vals = []
        mae_vals = []
        ade_vals = []
        fde_vals = []
        path_error_pct_vals = []
        fde_ratio_pct_vals = []

        top1_ade_vals = []
        top1_fde_vals = []
        top1_path_error_pct_vals = []

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

        reranker_mse_vals = []
        reranker_mae_vals = []
        reranker_ade_vals = []
        reranker_fde_vals = []
        reranker_path_error_pct_vals = []
        reranker_fde_ratio_pct_vals = []

        best_of_n_batch_count = 0
        reranker_batch_count = 0
        plot_data_top1 = None
        plot_data_best_of_n = None
        plot_data_reranker = None
        reranker_model = None
        if self.reranker_checkpoint is not None or self.candidate_selection_mode == "reranker_best_of_n":
            reranker_model = self._load_reranker(device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                obs_dict = {"obs": batch["obs"]}
                start, end = policy.get_action_window_indices()
                anchor_idx = policy.get_action_anchor_obs_index()

                top1_result = policy.predict_action(obs_dict)
                top1_pred_action = top1_result["action"]
                plot_gt_action = batch["action"][:, start:end]
                plot_start_pos = batch["obs"][:, anchor_idx, self.position_slice]
                obs_hist = batch["obs"][:, : policy.n_obs_steps]

                plot_gt_m = plot_gt_action * self.action_scale_to_meter
                plot_start_pos_m = plot_start_pos * self.action_scale_to_meter
                plot_pred_m = top1_pred_action * self.action_scale_to_meter
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
                top1_ade_vals.append(top1_metrics["traj_ade_m"])
                top1_fde_vals.append(top1_metrics["traj_fde_m"])
                top1_path_error_pct_vals.append(top1_metrics["traj_path_error_pct"])

                use_candidate_pool = False
                if candidate_count > 1:
                    if self.candidate_selection_mode == "top1":
                        use_candidate_pool = (
                            self.multi_sample_eval_max_batches is None
                            or (batch_idx + 1) <= self.multi_sample_eval_max_batches
                        )
                    else:
                        use_candidate_pool = True

                selected_metrics = top1_metrics
                selected_pred_m = plot_pred_m

                if use_candidate_pool:
                    pred_action_samples = [top1_pred_action]
                    for _ in range(1, candidate_count):
                        sample_result = policy.predict_action(obs_dict)
                        pred_action_samples.append(sample_result["action"])
                    pred_action_all = torch.stack(pred_action_samples, dim=0)
                    pred_action_m_all = pred_action_all * self.action_scale_to_meter
                    dist_all, gt_path_len_all = self._compute_candidate_path_stats(
                        pred_action_m_all=pred_action_m_all,
                        gt_action_m=plot_gt_m,
                        start_pos_m=plot_start_pos_m,
                    )
                    top1_idx = torch.zeros(pred_action_all.shape[1], device=device, dtype=torch.long)
                    top1_subset_metrics, top1_subset_pred_m = self._reduce_selected_candidate_metrics(
                        pred_action_m_all=pred_action_m_all,
                        gt_action_m=plot_gt_m,
                        dist=dist_all,
                        gt_path_len=gt_path_len_all,
                        choice_idx=top1_idx,
                    )
                    top1_subset_ade_vals.append(top1_subset_metrics["traj_ade_m"])
                    top1_subset_fde_vals.append(top1_subset_metrics["traj_fde_m"])
                    top1_subset_path_error_pct_vals.append(top1_subset_metrics["traj_path_error_pct"])
                    top1_subset_fde_ratio_pct_vals.append(top1_subset_metrics["traj_fde_ratio_pct"])

                    best_of_n_metrics, best_pred_action_m, best_idx = self._select_best_of_n_from_stats(
                        pred_action_m_all=pred_action_m_all,
                        gt_action_m=plot_gt_m,
                        dist=dist_all,
                        gt_path_len=gt_path_len_all,
                    )
                    best_of_n_mse_vals.append(best_of_n_metrics["action_mse"])
                    best_of_n_mae_vals.append(best_of_n_metrics["action_mae"])
                    best_of_n_ade_vals.append(best_of_n_metrics["traj_ade_m"])
                    best_of_n_fde_vals.append(best_of_n_metrics["traj_fde_m"])
                    best_of_n_path_error_pct_vals.append(best_of_n_metrics["traj_path_error_pct"])
                    best_of_n_fde_ratio_pct_vals.append(best_of_n_metrics["traj_fde_ratio_pct"])
                    best_of_n_batch_count += 1

                    reranker_metrics = None
                    reranker_pred_action_m = None
                    reranker_idx = None
                    if reranker_model is not None:
                        reranker_idx = self._select_reranker_best_of_n(
                            reranker=reranker_model,
                            obs_hist=obs_hist,
                            pred_action_all=pred_action_all,
                        )
                        reranker_metrics, reranker_pred_action_m = self._reduce_selected_candidate_metrics(
                            pred_action_m_all=pred_action_m_all,
                            gt_action_m=plot_gt_m,
                            dist=dist_all,
                            gt_path_len=gt_path_len_all,
                            choice_idx=reranker_idx,
                        )
                        reranker_mse_vals.append(reranker_metrics["action_mse"])
                        reranker_mae_vals.append(reranker_metrics["action_mae"])
                        reranker_ade_vals.append(reranker_metrics["traj_ade_m"])
                        reranker_fde_vals.append(reranker_metrics["traj_fde_m"])
                        reranker_path_error_pct_vals.append(reranker_metrics["traj_path_error_pct"])
                        reranker_fde_ratio_pct_vals.append(reranker_metrics["traj_fde_ratio_pct"])
                        reranker_batch_count += 1

                    if self.candidate_selection_mode == "oracle_best_of_n":
                        selected_metrics = best_of_n_metrics
                        selected_pred_m = best_pred_action_m
                    elif self.candidate_selection_mode == "reranker_best_of_n":
                        if reranker_metrics is None:
                            raise ValueError("reranker_best_of_n mode requires a valid reranker checkpoint.")
                        selected_metrics = reranker_metrics
                        selected_pred_m = reranker_pred_action_m

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
                            "variant_tag": f"bestof{candidate_count}",
                            "variant_title": f"best-of-{candidate_count}",
                            "pred_label": f"best_of_{candidate_count}",
                            "choice_idx": best_idx[: self.plot_num_samples].detach().cpu().numpy(),
                        }
                    if reranker_pred_action_m is not None and self.save_plots and (self.plot_action or self.plot_trajectory3d) and (plot_data_reranker is None):
                        plot_data_reranker = {
                            "pred": reranker_pred_action_m[: self.plot_num_samples].detach().cpu().numpy(),
                            "gt": plot_gt_m[: self.plot_num_samples].detach().cpu().numpy(),
                            "start_pos": plot_start_pos_m[: self.plot_num_samples].detach().cpu().numpy(),
                            "obs_pos_full": (
                                batch["obs"][: self.plot_num_samples, :, self.position_slice]
                                * self.action_scale_to_meter
                            ).detach().cpu().numpy(),
                            "n_obs_steps": int(policy.n_obs_steps),
                            "variant_tag": "reranker",
                            "variant_title": "reranker-best-of-n",
                            "pred_label": "reranker_best",
                            "choice_idx": reranker_idx[: self.plot_num_samples].detach().cpu().numpy(),
                        }

                mse_vals.append(selected_metrics["action_mse"])
                mae_vals.append(selected_metrics["action_mae"])
                ade_vals.append(selected_metrics["traj_ade_m"])
                fde_vals.append(selected_metrics["traj_fde_m"])
                path_error_pct_vals.append(selected_metrics["traj_path_error_pct"])
                fde_ratio_pct_vals.append(selected_metrics["traj_fde_ratio_pct"])

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
        else:
            ade = float(sum(ade_vals) / len(ade_vals))
            metrics = {
                "test/mean_score": -ade,
                "eval_action_mse": float(sum(mse_vals) / len(mse_vals)),
                "eval_action_mae": float(sum(mae_vals) / len(mae_vals)),
                "eval_traj_ade_m": ade,
                "eval_traj_fde_m": float(sum(fde_vals) / len(fde_vals)),
                "eval_traj_path_error_pct": float(sum(path_error_pct_vals) / len(path_error_pct_vals)),
                "eval_traj_fde_ratio_pct": float(sum(fde_ratio_pct_vals) / len(fde_ratio_pct_vals)),
                "eval_traj_ade_ratio_pct": float(sum(path_error_pct_vals) / len(path_error_pct_vals)),
                "eval_top1_traj_ade_m": float(sum(top1_ade_vals) / len(top1_ade_vals)),
                "eval_top1_traj_fde_m": float(sum(top1_fde_vals) / len(top1_fde_vals)),
                "eval_top1_traj_path_error_pct": float(sum(top1_path_error_pct_vals) / len(top1_path_error_pct_vals)),
            }
            if best_of_n_batch_count > 0:
                top1_subset_path_error_pct = float(sum(top1_subset_path_error_pct_vals) / len(top1_subset_path_error_pct_vals))
                best_of_n_path_error_pct = float(sum(best_of_n_path_error_pct_vals) / len(best_of_n_path_error_pct_vals))
                metrics.update({
                    "eval_best_of_n_action_mse": float(sum(best_of_n_mse_vals) / len(best_of_n_mse_vals)),
                    "eval_best_of_n_action_mae": float(sum(best_of_n_mae_vals) / len(best_of_n_mae_vals)),
                    "eval_best_of_n_traj_ade_m": float(sum(best_of_n_ade_vals) / len(best_of_n_ade_vals)),
                    "eval_best_of_n_traj_fde_m": float(sum(best_of_n_fde_vals) / len(best_of_n_fde_vals)),
                    "eval_best_of_n_traj_path_error_pct": best_of_n_path_error_pct,
                    "eval_best_of_n_traj_fde_ratio_pct": float(sum(best_of_n_fde_ratio_pct_vals) / len(best_of_n_fde_ratio_pct_vals)),
                    "eval_top1_subset_traj_ade_m": float(sum(top1_subset_ade_vals) / len(top1_subset_ade_vals)),
                    "eval_top1_subset_traj_fde_m": float(sum(top1_subset_fde_vals) / len(top1_subset_fde_vals)),
                    "eval_top1_subset_traj_path_error_pct": top1_subset_path_error_pct,
                    "eval_top1_subset_traj_fde_ratio_pct": float(sum(top1_subset_fde_ratio_pct_vals) / len(top1_subset_fde_ratio_pct_vals)),
                    "eval_best_of_n_gain_traj_path_error_pct": top1_subset_path_error_pct - best_of_n_path_error_pct,
                    "eval_best_of_n_samples": float(candidate_count),
                    "eval_best_of_n_batches": float(best_of_n_batch_count),
                })
            if reranker_batch_count > 0:
                reranker_path_error_pct = float(sum(reranker_path_error_pct_vals) / len(reranker_path_error_pct_vals))
                top1_ref = (
                    float(sum(top1_subset_path_error_pct_vals) / len(top1_subset_path_error_pct_vals))
                    if top1_subset_path_error_pct_vals else float(sum(top1_path_error_pct_vals) / len(top1_path_error_pct_vals))
                )
                metrics.update({
                    "eval_reranker_action_mse": float(sum(reranker_mse_vals) / len(reranker_mse_vals)),
                    "eval_reranker_action_mae": float(sum(reranker_mae_vals) / len(reranker_mae_vals)),
                    "eval_reranker_traj_ade_m": float(sum(reranker_ade_vals) / len(reranker_ade_vals)),
                    "eval_reranker_traj_fde_m": float(sum(reranker_fde_vals) / len(reranker_fde_vals)),
                    "eval_reranker_traj_path_error_pct": reranker_path_error_pct,
                    "eval_reranker_traj_fde_ratio_pct": float(sum(reranker_fde_ratio_pct_vals) / len(reranker_fde_ratio_pct_vals)),
                    "eval_reranker_gain_traj_path_error_pct": top1_ref - reranker_path_error_pct,
                    "eval_reranker_samples": float(candidate_count),
                    "eval_reranker_batches": float(reranker_batch_count),
                })

        if self.save_plots and (self.eval_count % self.plot_interval == 0):
            os.makedirs(self.plot_dir, exist_ok=True)
            if plot_data_top1 is not None:
                if self.plot_action:
                    self._save_action_plot(plot_data_top1)
                if self.plot_trajectory3d:
                    self._save_trajectory3d_plots(plot_data_top1)
            if plot_data_reranker is not None:
                if self.plot_action:
                    self._save_action_plot(plot_data_reranker)
                if self.plot_trajectory3d:
                    self._save_trajectory3d_plots(plot_data_reranker)
            if plot_data_best_of_n is not None:
                if self.plot_action:
                    self._save_action_plot(plot_data_best_of_n)
                if self.plot_trajectory3d:
                    self._save_trajectory3d_plots(plot_data_best_of_n)
            if self.plot_curves:
                self._save_curve_plot()

        return metrics
