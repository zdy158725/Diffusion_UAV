from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            velocity_loss_weight: float=0.0,
            short_horizon_loss_weight: float=0.0,
            short_horizon_focus_steps: int=4,
            short_horizon_focus_gain: float=3.0,
            terminal_pos_loss_weight: float=0.0,
            terminal_pos_loss_power: float=1.0,
            relative_path_loss_weight: float=0.0,
            action_target_type: str='delta_position',
            # parameters passed to step
            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.velocity_loss_weight = float(velocity_loss_weight)
        self.short_horizon_loss_weight = float(short_horizon_loss_weight)
        self.short_horizon_focus_steps = int(short_horizon_focus_steps)
        self.short_horizon_focus_gain = float(short_horizon_focus_gain)
        self.terminal_pos_loss_weight = float(terminal_pos_loss_weight)
        self.terminal_pos_loss_power = float(terminal_pos_loss_power)
        self.relative_path_loss_weight = float(relative_path_loss_weight)
        self.action_target_type = str(action_target_type)
        if self.action_target_type not in {'delta_position', 'relative_future_position'}:
            raise ValueError(
                "Unsupported action_target_type="
                f"{self.action_target_type}. Expected delta_position or relative_future_position."
            )
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def get_action_window_indices(self) -> Tuple[int, int]:
        start = self.n_obs_steps
        end = start + self.n_action_steps
        return start, end

    def get_action_anchor_obs_index(self) -> int:
        return self.n_obs_steps - 1

    def get_action_token_offset(self) -> int:
        if self.pred_action_steps_only:
            start, _ = self.get_action_window_indices()
            return start
        return 0
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            token_offset=0,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond, token_offset=token_offset)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator = None,
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = nobs[:,:To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            generator=generator,
            token_offset=self.get_action_token_offset(),
            **self.kwargs,
        )
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start, end = self.get_action_window_indices()
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            if self.pred_action_steps_only:
                start, end = self.get_action_window_indices()
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)
        
        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps,
            cond,
            token_offset=self.get_action_token_offset()
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
            alpha_t = self.noise_scheduler.alphas_cumprod.to(
                device=trajectory.device, dtype=trajectory.dtype
            )[timesteps]
            alpha_t = alpha_t.view(-1, *([1] * (trajectory.ndim - 1)))
            x0_pred = (noisy_trajectory - torch.sqrt(1.0 - alpha_t) * pred) / torch.sqrt(alpha_t)
        elif pred_type == 'sample':
            target = trajectory
            x0_pred = pred
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        total_loss = loss.mean()

        def get_eval_action_tensors(pred_tensor, target_tensor, mask_tensor=None):
            if self.pred_action_steps_only:
                return pred_tensor, target_tensor, mask_tensor

            T = target_tensor.shape[1]
            start = min(max(self.n_obs_steps, 0), T - 1)
            end = min(start + self.n_action_steps, T)
            focus_pred = pred_tensor[:, start:end, :]
            focus_target = target_tensor[:, start:end, :]
            focus_mask = None
            if mask_tensor is not None:
                focus_mask = mask_tensor[:, start:end, :]
            return focus_pred, focus_target, focus_mask

        if self.short_horizon_loss_weight > 0.0 and trajectory.shape[1] > 0:
            T = trajectory.shape[1]
            if self.pred_action_steps_only:
                focus_start = 0
            else:
                focus_start = min(max(self.n_obs_steps, 0), T - 1)
            focus_steps = max(self.short_horizon_focus_steps, 1)
            focus_end = min(focus_start + focus_steps, T)

            step_weights = torch.ones(
                (1, T, 1), device=trajectory.device, dtype=trajectory.dtype
            )
            if focus_end > focus_start:
                step_weights[:, focus_start:focus_end, :] = self.short_horizon_focus_gain

            recon_err = (x0_pred - trajectory) ** 2
            recon_err = recon_err * loss_mask.type(recon_err.dtype) * step_weights
            recon_denom = (
                loss_mask.type(recon_err.dtype) * step_weights
            ).sum().clamp(min=1.0)
            short_horizon_loss = recon_err.sum() / recon_denom
            total_loss = total_loss + self.short_horizon_loss_weight * short_horizon_loss

        if self.velocity_loss_weight > 0.0 and trajectory.shape[1] > 1:
            vel_pred = x0_pred[:, 1:, :] - x0_pred[:, :-1, :]
            vel_target = trajectory[:, 1:, :] - trajectory[:, :-1, :]
            vel_mask = loss_mask[:, 1:, :] & loss_mask[:, :-1, :]

            vel_err = (vel_pred - vel_target) ** 2
            vel_err = vel_err * vel_mask.type(vel_err.dtype)
            vel_denom = vel_mask.type(vel_err.dtype).sum().clamp(min=1.0)
            vel_loss = vel_err.sum() / vel_denom
            total_loss = total_loss + self.velocity_loss_weight * vel_loss

        if self.terminal_pos_loss_weight > 0.0 and trajectory.shape[1] > 0:
            focus_pred, focus_target, focus_mask = get_eval_action_tensors(
                x0_pred, trajectory, loss_mask
            )

            if focus_pred.shape[1] > 0:
                if self.action_target_type == 'relative_future_position':
                    pred_pos = self.normalizer['action'].unnormalize(focus_pred)
                    target_pos = self.normalizer['action'].unnormalize(focus_target)
                else:
                    pred_pos = torch.cumsum(focus_pred, dim=1)
                    target_pos = torch.cumsum(focus_target, dim=1)
                final_sq_err = (pred_pos[:, -1, :] - target_pos[:, -1, :]) ** 2
                final_mask = focus_mask[:, -1, :].type(final_sq_err.dtype)
                denom = final_mask.sum().clamp(min=1.0)
                terminal_pos_loss = (final_sq_err * final_mask).sum() / denom
                if self.terminal_pos_loss_power != 1.0:
                    terminal_pos_loss = torch.pow(
                        terminal_pos_loss + 1e-8, self.terminal_pos_loss_power
                    )
                total_loss = total_loss + self.terminal_pos_loss_weight * terminal_pos_loss

        if self.relative_path_loss_weight > 0.0 and trajectory.shape[1] > 0:
            focus_pred, focus_target, _ = get_eval_action_tensors(x0_pred, trajectory)
            if focus_pred.shape[1] > 0:
                # Compute the trajectory ratio in physical action space so it matches eval semantics.
                pred_action = self.normalizer['action'].unnormalize(focus_pred)
                target_action = self.normalizer['action'].unnormalize(focus_target)
                if self.action_target_type == 'relative_future_position':
                    current_pos = batch['obs'][
                        :, self.get_action_anchor_obs_index(), :self.action_dim
                    ].to(device=pred_action.device, dtype=pred_action.dtype)
                    pred_pos = current_pos[:, None, :] + pred_action
                    target_pos = current_pos[:, None, :] + target_action
                    gt_step = torch.zeros_like(target_pos)
                    gt_step[:, 0, :] = target_pos[:, 0, :] - current_pos
                    if target_pos.shape[1] > 1:
                        gt_step[:, 1:, :] = target_pos[:, 1:, :] - target_pos[:, :-1, :]
                    gt_path_len = torch.linalg.norm(gt_step, dim=-1).sum(dim=1).clamp(min=1e-6)
                else:
                    pred_pos = torch.cumsum(pred_action, dim=1)
                    target_pos = torch.cumsum(target_action, dim=1)
                    gt_path_len = torch.linalg.norm(target_action, dim=-1).sum(dim=1).clamp(min=1e-6)
                step_err = torch.linalg.norm(pred_pos - target_pos, dim=-1)
                relative_path_loss = (step_err.sum(dim=1) / gt_path_len).mean()
                total_loss = total_loss + self.relative_path_loss_weight * relative_path_loss

        return total_loss
