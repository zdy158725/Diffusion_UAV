if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

from torchinfo import summary

OmegaConf.register_new_resolver("eval", eval, replace=True)

STRUCTURED_POLICY_OBS_KEYS = (
    "obs",
    "agent_obs",
    "agent_team",
    "agent_valid",
    "agent_social_mask",
    "agent_role_id",
)

# %%
class TrainDiffusionTransformerLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(
            dataset,
            collate_fn=dataset.get_collate_fn(),
            **cfg.dataloader,
        )
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(
            val_dataset,
            collate_fn=val_dataset.get_collate_fn(),
            **cfg.val_dataloader,
        )

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        endpoint_ckpt = OmegaConf.select(cfg, "endpoint_pretrained_ckpt", default=None)
        should_load_endpoint_ckpt = (
            endpoint_ckpt is not None and self.global_step == 0 and self.epoch == 0
        )
        if should_load_endpoint_ckpt and hasattr(self.model, "load_endpoint_predictor"):
            self.model.load_endpoint_predictor(endpoint_ckpt)
            if self.ema_model is not None and hasattr(self.ema_model, "load_endpoint_predictor"):
                self.ema_model.load_endpoint_predictor(endpoint_ckpt)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        plot_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        train_loss_history = []
        val_denoise_loss_history = []
        val_epoch_history = []
        eval_ade_history = []
        eval_fde_history = []
        eval_ratio_pct_history = []
        eval_top1_subset_ade_history = []
        eval_top1_subset_fde_history = []
        eval_top1_subset_ratio_pct_history = []
        eval_best_of_n_ade_history = []
        eval_best_of_n_fde_history = []
        eval_best_of_n_ratio_pct_history = []
        eval_epoch_history = []
        best_of_n_label = 'Best-of-N'

        def annotate_best_point(ax, x_vals, y_vals, label, color, offset):
            if len(x_vals) == 0 or len(y_vals) == 0:
                return

            x = np.asarray(x_vals, dtype=np.float32)
            y = np.asarray(y_vals, dtype=np.float32)
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                return

            x = x[valid]
            y = y[valid]
            best_idx = int(np.argmin(y))
            best_x = float(x[best_idx])
            best_y = float(y[best_idx])

            ax.scatter([best_x], [best_y], color=color, s=36, zorder=5)
            ax.annotate(
                f"best {label}: {best_y:.3f}\n(epoch {int(round(best_x))})",
                xy=(best_x, best_y),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

        def save_loss_plot():
            if len(train_loss_history) == 0:
                return
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(len(train_loss_history)), train_loss_history, label='train denoise')
            if len(val_denoise_loss_history) > 0:
                ax.plot(val_epoch_history, val_denoise_loss_history, label='val denoise')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, 'loss_curve.png'), dpi=150)
            plt.close(fig)

        def save_traj_error_plot():
            if len(eval_epoch_history) == 0:
                return

            best_of_n_suffix = best_of_n_label.lower().replace('-', '').replace(' ', '')

            fig, ax = plt.subplots(figsize=(6, 4))
            if len(eval_ade_history) > 0:
                ade_line, = ax.plot(eval_epoch_history, eval_ade_history, label='ADE (m)')
                annotate_best_point(
                    ax, eval_epoch_history, eval_ade_history,
                    label='ADE', color=ade_line.get_color(), offset=(8, -18)
                )
            if len(eval_fde_history) > 0:
                fde_line, = ax.plot(eval_epoch_history, eval_fde_history, label='FDE (m)')
                annotate_best_point(
                    ax, eval_epoch_history, eval_fde_history,
                    label='FDE', color=fde_line.get_color(), offset=(8, 10)
                )
            ax.set_xlabel('epoch')
            ax.set_ylabel('error (m)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, 'traj_error_curve.png'), dpi=150)
            plt.close(fig)

            has_best_of_n_error = (
                len(eval_top1_subset_ade_history) > 0 or
                len(eval_top1_subset_fde_history) > 0 or
                len(eval_best_of_n_ade_history) > 0 or
                len(eval_best_of_n_fde_history) > 0
            )
            if has_best_of_n_error:
                fig, ax = plt.subplots(figsize=(6, 4))
                if len(eval_top1_subset_ade_history) > 0:
                    subset_ade_line, = ax.plot(
                        eval_epoch_history,
                        eval_top1_subset_ade_history,
                        linestyle=':',
                        label='Top-1 subset ADE (m)'
                    )
                    annotate_best_point(
                        ax, eval_epoch_history, eval_top1_subset_ade_history,
                        label='top1 subset ADE',
                        color=subset_ade_line.get_color(),
                        offset=(8, -18)
                    )
                if len(eval_best_of_n_ade_history) > 0:
                    best_ade_line, = ax.plot(
                        eval_epoch_history,
                        eval_best_of_n_ade_history,
                        linestyle='--',
                        label=f'{best_of_n_label} ADE (m, subset)'
                    )
                    annotate_best_point(
                        ax, eval_epoch_history, eval_best_of_n_ade_history,
                        label=f'{best_of_n_label} ADE',
                        color=best_ade_line.get_color(),
                        offset=(8, 12)
                    )
                if len(eval_top1_subset_fde_history) > 0:
                    subset_fde_line, = ax.plot(
                        eval_epoch_history,
                        eval_top1_subset_fde_history,
                        linestyle=':',
                        label='Top-1 subset FDE (m)'
                    )
                    annotate_best_point(
                        ax, eval_epoch_history, eval_top1_subset_fde_history,
                        label='top1 subset FDE',
                        color=subset_fde_line.get_color(),
                        offset=(8, 10)
                    )
                if len(eval_best_of_n_fde_history) > 0:
                    best_fde_line, = ax.plot(
                        eval_epoch_history,
                        eval_best_of_n_fde_history,
                        linestyle='--',
                        label=f'{best_of_n_label} FDE (m, subset)'
                    )
                    annotate_best_point(
                        ax, eval_epoch_history, eval_best_of_n_fde_history,
                        label=f'{best_of_n_label} FDE',
                        color=best_fde_line.get_color(),
                        offset=(8, -22)
                    )
                ax.set_xlabel('epoch')
                ax.set_ylabel('error (m)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, f'traj_error_curve_{best_of_n_suffix}.png'), dpi=150)
                plt.close(fig)

            if len(eval_ratio_pct_history) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ratio_line, = ax.plot(
                    eval_epoch_history,
                    eval_ratio_pct_history,
                    color='tab:green',
                    label='Top-1 path err (%)'
                )
                annotate_best_point(
                    ax, eval_epoch_history, eval_ratio_pct_history,
                    label='top1', color=ratio_line.get_color(), offset=(8, 10)
                )
                ax.set_xlabel('epoch')
                ax.set_ylabel('ratio (%)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, 'traj_error_ratio_curve.png'), dpi=150)
                plt.close(fig)

            has_best_of_n_ratio = (
                len(eval_top1_subset_ratio_pct_history) > 0 or
                len(eval_best_of_n_ratio_pct_history) > 0
            )
            if has_best_of_n_ratio:
                fig, ax = plt.subplots(figsize=(6, 4))
                if len(eval_top1_subset_ratio_pct_history) > 0:
                    subset_ratio_line, = ax.plot(
                        eval_epoch_history,
                        eval_top1_subset_ratio_pct_history,
                        color='0.5',
                        linestyle=':',
                        label='Top-1 subset path err (%)'
                    )
                    annotate_best_point(
                        ax, eval_epoch_history, eval_top1_subset_ratio_pct_history,
                        label='top1 subset',
                        color=subset_ratio_line.get_color(),
                        offset=(8, 10)
                    )
                if len(eval_best_of_n_ratio_pct_history) > 0:
                    best_ratio_line, = ax.plot(
                        eval_epoch_history,
                        eval_best_of_n_ratio_pct_history,
                        color='tab:red',
                        linestyle='--',
                        label=f'{best_of_n_label} subset path err (%)'
                    )
                    annotate_best_point(
                        ax, eval_epoch_history, eval_best_of_n_ratio_pct_history,
                        label=best_of_n_label.lower(),
                        color=best_ratio_line.get_color(),
                        offset=(8, -22)
                    )
                ax.set_xlabel('epoch')
                ax.set_ylabel('ratio (%)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, f'traj_error_ratio_curve_{best_of_n_suffix}.png'), dpi=150)
                plt.close(fig)

        train_start_time = time.time()
        with tqdm.tqdm(
            total=cfg.training.num_epochs,
            desc="Training progress",
            unit="epoch",
            mininterval=cfg.training.tqdm_interval_sec
        ) as epoch_pbar, JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                epoch_start_time = time.time()
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                train_loss_history.append(train_loss)

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)
                    best_of_n_samples = runner_log.get('eval_best_of_n_samples', None)
                    if best_of_n_samples is not None and np.isfinite(best_of_n_samples):
                        best_of_n_label = f"Best-of-{int(best_of_n_samples)}"
                    ade = runner_log.get('eval_traj_ade_m', None)
                    fde = runner_log.get('eval_traj_fde_m', None)
                    ratio_pct = runner_log.get(
                        'eval_traj_path_error_pct',
                        runner_log.get('eval_traj_ade_ratio_pct', None)
                    )
                    top1_subset_ade = runner_log.get('eval_top1_subset_traj_ade_m', None)
                    top1_subset_fde = runner_log.get('eval_top1_subset_traj_fde_m', None)
                    top1_subset_ratio_pct = runner_log.get('eval_top1_subset_traj_path_error_pct', None)
                    best_of_n_ade = runner_log.get('eval_best_of_n_traj_ade_m', None)
                    best_of_n_fde = runner_log.get('eval_best_of_n_traj_fde_m', None)
                    best_of_n_ratio_pct = runner_log.get('eval_best_of_n_traj_path_error_pct', None)
                    has_ade = (ade is not None) and np.isfinite(ade)
                    has_fde = (fde is not None) and np.isfinite(fde)
                    has_ratio = (ratio_pct is not None) and np.isfinite(ratio_pct)
                    if has_ade or has_fde or has_ratio:
                        eval_epoch_history.append(len(train_loss_history)-1)
                        eval_ade_history.append(float(ade) if has_ade else np.nan)
                        eval_fde_history.append(float(fde) if has_fde else np.nan)
                        eval_ratio_pct_history.append(float(ratio_pct) if has_ratio else np.nan)
                        eval_top1_subset_ade_history.append(
                            float(top1_subset_ade)
                            if (top1_subset_ade is not None) and np.isfinite(top1_subset_ade)
                            else np.nan
                        )
                        eval_top1_subset_fde_history.append(
                            float(top1_subset_fde)
                            if (top1_subset_fde is not None) and np.isfinite(top1_subset_fde)
                            else np.nan
                        )
                        eval_top1_subset_ratio_pct_history.append(
                            float(top1_subset_ratio_pct)
                            if (top1_subset_ratio_pct is not None) and np.isfinite(top1_subset_ratio_pct)
                            else np.nan
                        )
                        eval_best_of_n_ade_history.append(
                            float(best_of_n_ade)
                            if (best_of_n_ade is not None) and np.isfinite(best_of_n_ade)
                            else np.nan
                        )
                        eval_best_of_n_fde_history.append(
                            float(best_of_n_fde)
                            if (best_of_n_fde is not None) and np.isfinite(best_of_n_fde)
                            else np.nan
                        )
                        eval_best_of_n_ratio_pct_history.append(
                            float(best_of_n_ratio_pct)
                            if (best_of_n_ratio_pct is not None) and np.isfinite(best_of_n_ratio_pct)
                            else np.nan
                        )

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_denoise_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss. Keep val_loss as a
                            # backward-compatible alias for older analysis scripts.
                            step_log['val_denoise_loss'] = val_denoise_loss
                            step_log['val_loss'] = val_denoise_loss
                            val_denoise_loss_history.append(val_denoise_loss)
                            val_epoch_history.append(len(train_loss_history)-1)
            
                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = {
                            key: batch[key]
                            for key in STRUCTURED_POLICY_OBS_KEYS
                            if key in batch
                        }
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start, end = policy.get_action_window_indices()
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                save_loss_plot()
                save_traj_error_plot()

                epoch_time_sec = time.time() - epoch_start_time
                elapsed_sec = time.time() - train_start_time
                done_epochs = local_epoch_idx + 1
                avg_epoch_time = elapsed_sec / max(done_epochs, 1)
                eta_sec = max(cfg.training.num_epochs - done_epochs, 0) * avg_epoch_time
                step_log['epoch_time_sec'] = epoch_time_sec
                step_log['eta_sec'] = eta_sec

                postfix = {
                    'train_loss': f"{train_loss:.4f}",
                    'eta_min': f"{eta_sec/60.0:.1f}",
                }
                if 'val_denoise_loss' in step_log:
                    postfix['val_denoise'] = f"{step_log['val_denoise_loss']:.4f}"
                elif 'val_loss' in step_log:
                    postfix['val_denoise'] = f"{step_log['val_loss']:.4f}"
                if 'eval_traj_ade_m' in step_log:
                    postfix['ade_m'] = f"{step_log['eval_traj_ade_m']:.2f}"
                path_err_pct = step_log.get(
                    'eval_traj_path_error_pct',
                    step_log.get('eval_traj_ade_ratio_pct')
                )
                if path_err_pct is not None:
                    postfix['path_err_pct'] = f"{path_err_pct:.1f}%"
                epoch_pbar.set_postfix(postfix, refresh=False)
                epoch_pbar.update(1)

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
