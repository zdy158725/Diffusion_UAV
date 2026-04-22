if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random

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
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainStructuredUAVEndpointWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = hydra.utils.instantiate(cfg.policy)
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.global_step = 0
        self.epoch = 0

    def _move_batch(self, batch, device):
        return dict_apply(batch, lambda x: x.to(device, non_blocking=True))

    def _evaluate(self, dataloader, device, max_steps=None):
        self.model.eval()
        loss_values = []
        mse_values = []
        dist_values = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_steps is not None and batch_idx >= max_steps:
                    break
                batch = self._move_batch(batch, device)
                loss = self.model.compute_loss(batch)
                pred = self.model.predict_endpoint(batch)
                target = batch["endpoint_target"]
                mse = torch.mean((pred - target) ** 2, dim=-1)
                dist = torch.linalg.norm(pred - target, dim=-1)
                loss_values.append(float(loss.detach().cpu().item()))
                mse_values.append(mse.detach().cpu())
                dist_values.append(dist.detach().cpu())
        if not loss_values:
            return {
                "val_endpoint_loss": float("inf"),
                "val_endpoint_mse": float("inf"),
                "val_endpoint_ade_m": float("inf"),
                "val_endpoint_fde_m": float("inf"),
                "test/mean_score": float("-inf"),
            }
        mse_tensor = torch.cat(mse_values, dim=0)
        dist_tensor = torch.cat(dist_values, dim=0)
        val_endpoint_loss = float(np.mean(loss_values))
        val_endpoint_mse = float(mse_tensor.mean().item())
        val_endpoint_ade_m = float(dist_tensor.mean().item())
        val_endpoint_fde_m = float(dist_tensor.mean().item())
        return {
            "val_endpoint_loss": val_endpoint_loss,
            "val_endpoint_mse": val_endpoint_mse,
            "val_endpoint_ade_m": val_endpoint_ade_m,
            "val_endpoint_fde_m": val_endpoint_fde_m,
            "test/mean_score": -val_endpoint_fde_m,
        }

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(
            dataset,
            collate_fn=dataset.get_collate_fn(),
            **cfg.dataloader,
        )
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(
            val_dataset,
            collate_fn=val_dataset.get_collate_fn(),
            **cfg.val_dataloader,
        )

        normalizer: LinearNormalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
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
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(self.epoch, cfg.training.num_epochs):
                self.model.train()
                train_losses = []
                epoch_iterator = tqdm.tqdm(
                    train_dataloader,
                    desc=f"Endpoint epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                )
                for batch_idx, batch in enumerate(epoch_iterator):
                    if (
                        cfg.training.max_train_steps is not None
                        and batch_idx >= cfg.training.max_train_steps
                    ):
                        break
                    batch = self._move_batch(batch, device)
                    loss = self.model.compute_loss(batch)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()

                    loss_cpu = float(loss.detach().cpu().item())
                    train_losses.append(loss_cpu)
                    epoch_iterator.set_postfix(
                        loss=loss_cpu,
                        lr=lr_scheduler.get_last_lr()[0],
                    )
                    self.global_step += 1

                log_data = {
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "train_endpoint_loss": (
                        float(np.mean(train_losses)) if train_losses else float("nan")
                    ),
                    "lr": float(lr_scheduler.get_last_lr()[0]),
                }

                if self.epoch % cfg.training.val_every == 0:
                    log_data.update(
                        self._evaluate(
                            val_dataloader,
                            device=device,
                            max_steps=cfg.training.max_val_steps,
                        )
                    )

                if self.epoch % cfg.training.checkpoint_every == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(tag="latest")
                    topk_path = None
                    if "val_endpoint_fde_m" in log_data:
                        topk_path = topk_manager.get_ckpt_path(log_data)
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path)

                wandb_run.log(log_data, step=self.global_step)
                json_logger.log(log_data)
                self.epoch += 1

        wandb_run.finish()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
)
def main(cfg: OmegaConf):
    workspace = TrainStructuredUAVEndpointWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
