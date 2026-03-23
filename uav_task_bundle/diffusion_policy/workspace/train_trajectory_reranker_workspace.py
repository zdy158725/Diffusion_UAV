if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import random
import time

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.trajectory_candidate_reranker_dataset import (
    TrajectoryCandidateRerankerDataset,
)
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.policy.trajectory_reranker_policy import TrajectoryRerankerPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainTrajectoryRerankerWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: TrajectoryRerankerPolicy = hydra.utils.instantiate(cfg.policy)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0

    def _evaluate(self, dataloader, device, max_steps=None):
        self.model.eval()
        val_losses = []
        top1_ade_vals = []
        top1_fde_vals = []
        top1_ratio_vals = []
        reranker_ade_vals = []
        reranker_fde_vals = []
        reranker_ratio_vals = []
        oracle_ade_vals = []
        oracle_fde_vals = []
        oracle_ratio_vals = []
        choice_acc_vals = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                pred_error = self.model.predict_scores(
                    obs_hist=batch["obs_hist"],
                    cand_action=batch["cand_action"],
                    cand_rel_pos=batch["cand_rel_pos"],
                ).squeeze(-1)
                target_error = batch["cand_path_error_pct"]
                val_losses.append(
                    torch.nn.functional.smooth_l1_loss(pred_error, target_error).item()
                )

                reranker_idx = torch.argmin(pred_error, dim=1)
                oracle_idx = torch.argmin(target_error, dim=1)
                top1_idx = torch.zeros_like(reranker_idx)

                batch_indices = torch.arange(reranker_idx.shape[0], device=device)
                top1_ade_vals.append(batch["cand_ade_m"][batch_indices, top1_idx].mean().item())
                top1_fde_vals.append(batch["cand_fde_m"][batch_indices, top1_idx].mean().item())
                top1_ratio_vals.append(
                    batch["cand_path_error_pct"][batch_indices, top1_idx].mean().item()
                )
                reranker_ade_vals.append(
                    batch["cand_ade_m"][batch_indices, reranker_idx].mean().item()
                )
                reranker_fde_vals.append(
                    batch["cand_fde_m"][batch_indices, reranker_idx].mean().item()
                )
                reranker_ratio_vals.append(
                    batch["cand_path_error_pct"][batch_indices, reranker_idx].mean().item()
                )
                oracle_ade_vals.append(
                    batch["cand_ade_m"][batch_indices, oracle_idx].mean().item()
                )
                oracle_fde_vals.append(
                    batch["cand_fde_m"][batch_indices, oracle_idx].mean().item()
                )
                oracle_ratio_vals.append(
                    batch["cand_path_error_pct"][batch_indices, oracle_idx].mean().item()
                )
                choice_acc_vals.append(
                    (reranker_idx == batch["best_idx"]).float().mean().item()
                )

                if max_steps is not None and batch_idx >= (int(max_steps) - 1):
                    break

        metrics = {
            "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
            "val_top1_traj_ade_m": float(np.mean(top1_ade_vals)) if top1_ade_vals else float("nan"),
            "val_top1_traj_fde_m": float(np.mean(top1_fde_vals)) if top1_fde_vals else float("nan"),
            "val_top1_traj_path_error_pct": float(np.mean(top1_ratio_vals)) if top1_ratio_vals else float("nan"),
            "val_reranker_traj_ade_m": float(np.mean(reranker_ade_vals)) if reranker_ade_vals else float("nan"),
            "val_reranker_traj_fde_m": float(np.mean(reranker_fde_vals)) if reranker_fde_vals else float("nan"),
            "val_reranker_traj_path_error_pct": float(np.mean(reranker_ratio_vals)) if reranker_ratio_vals else float("nan"),
            "val_oracle_traj_ade_m": float(np.mean(oracle_ade_vals)) if oracle_ade_vals else float("nan"),
            "val_oracle_traj_fde_m": float(np.mean(oracle_fde_vals)) if oracle_fde_vals else float("nan"),
            "val_oracle_traj_path_error_pct": float(np.mean(oracle_ratio_vals)) if oracle_ratio_vals else float("nan"),
            "val_choice_acc": float(np.mean(choice_acc_vals)) if choice_acc_vals else float("nan"),
        }
        if np.isfinite(metrics["val_top1_traj_path_error_pct"]) and np.isfinite(
            metrics["val_reranker_traj_path_error_pct"]
        ):
            metrics["val_reranker_gain_traj_path_error_pct"] = (
                metrics["val_top1_traj_path_error_pct"]
                - metrics["val_reranker_traj_path_error_pct"]
            )
        else:
            metrics["val_reranker_gain_traj_path_error_pct"] = float("nan")
        if np.isfinite(metrics["val_oracle_traj_path_error_pct"]) and np.isfinite(
            metrics["val_reranker_traj_path_error_pct"]
        ):
            metrics["val_reranker_oracle_gap_traj_path_error_pct"] = (
                metrics["val_reranker_traj_path_error_pct"]
                - metrics["val_oracle_traj_path_error_pct"]
            )
        else:
            metrics["val_reranker_oracle_gap_traj_path_error_pct"] = float("nan")
        metrics["test/mean_score"] = -metrics["val_reranker_traj_ade_m"]
        return metrics

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                self.load_checkpoint(path=latest_ckpt_path)

        train_dataset: TrajectoryCandidateRerankerDataset = hydra.utils.instantiate(cfg.dataset)
        val_dataset: TrajectoryCandidateRerankerDataset = hydra.utils.instantiate(cfg.val_dataset)

        train_dataloader = DataLoader(train_dataset, **cfg.dataloader)
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer = train_dataset.get_normalizer(mode=cfg.normalizer.mode)
        self.model.set_normalizer(normalizer)

        effective_train_steps = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            effective_train_steps = min(effective_train_steps, int(cfg.training.max_train_steps))
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                effective_train_steps * int(cfg.training.num_epochs)
            ) // int(cfg.training.gradient_accumulate_every),
            last_epoch=self.global_step - 1,
        )

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.val_every = 1
            cfg.training.checkpoint_every = 1

        train_start_time = time.time()
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with tqdm.tqdm(
            total=cfg.training.num_epochs,
            desc="Training progress",
            unit="epoch",
            mininterval=cfg.training.tqdm_interval_sec,
        ) as epoch_pbar, JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                epoch_start_time = time.time()
                self.model.train()
                train_losses = []
                step_log = {}

                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if cfg.training.max_train_steps is not None and batch_idx >= (
                            int(cfg.training.max_train_steps) - 1
                        ):
                            break

                step_log["train_loss"] = float(np.mean(train_losses)) if train_losses else float("nan")

                if (self.epoch % cfg.training.val_every) == 0:
                    step_log.update(
                        self._evaluate(
                            dataloader=val_dataloader,
                            device=device,
                            max_steps=cfg.training.max_val_steps,
                        )
                    )

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()

                    metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                step_log["epoch_time_sec"] = time.time() - epoch_start_time
                step_log["elapsed_time_sec"] = time.time() - train_start_time
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                epoch_pbar.update(1)
                epoch_pbar.set_postfix(
                    train_loss=step_log.get("train_loss", float("nan")),
                    val_loss=step_log.get("val_loss", float("nan")),
                )
