import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent


class ATR(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, action_dim, num_action_skill, prediction_mlp_layers, projection_mlp_layers):
        super().__init__()

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim*num_action_skill, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.obs_encoder = nn.Sequential(nn.Linear(2 * obs_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        self.apply(utils.weight_init)

        self.action_projection_model = utils.MLP(
            action_dim*num_action_skill,
            skill_dim,
            hidden_dim,
            num_layers=projection_mlp_layers,
            normalization=utils.get_mlp_normalization(hidden_dim),
            weight_standardization=False,
        )

        self.action_prediction_model = utils.MLP(
            skill_dim,
            skill_dim,
            hidden_dim,
            num_layers=prediction_mlp_layers,
            normalization=utils.get_mlp_normalization(hidden_dim),
            weight_standardization=False,
        )

        self.obs_projection_model = utils.MLP(
            obs_dim,
            skill_dim,
            hidden_dim,
            num_layers=projection_mlp_layers,
            normalization= utils.get_mlp_normalization(hidden_dim),
            weight_standardization=False,
        )

        self.obs_prediction_model = utils.MLP(
            skill_dim,
            skill_dim,
            hidden_dim,
            num_layers=prediction_mlp_layers,
            normalization=utils.get_mlp_normalization(hidden_dim),
            weight_standardization=False,
        )

    def forward(self, action_x, obs_x):
        """
        Input:
        im_q: a batch of query images
        im_k: a batch of key images
        Output: logits, targets
        """

        im_q = action_x.contiguous()
        im_k = obs_x.contiguous()

        # compute query features
        emb_q = self.action_encoder(im_q)
        q_projection = self.action_projection_model(emb_q)
        q = self.action_prediction_model(q_projection)  # queries: NxC
        emb_k = self.obs_encoder(im_k)
        k_projection = self.obs_projection_model(emb_k)
        k = self.obs_prediction_model(k_projection)  # queries: NxC

        return emb_q, q, k


class ATRAgent(DDPGAgent):
    def __init__(self, skill_dim, num_action_skill, prediction_mlp_layers, projection_mlp_layers, atr_scale,
                 update_encoder, **kwargs):
        self.skill_dim = skill_dim
        self.num_action_skill = num_action_skill
        self.atr_scale = atr_scale
        self.update_encoder = update_encoder
        self.variance_loss_epsilon= kwargs['variance_loss_epsilon']
        self.invariance_loss_weight= kwargs['invariance_loss_weight']
        self.variance_loss_weight= kwargs['variance_loss_weight']
        self.covariance_loss_weight= kwargs['covariance_loss_weight']

        # create actor and critic
        super().__init__(**kwargs)

        # create atr
        self.atr = ATR(self.obs_dim, self.skill_dim,
                           kwargs['hidden_dim'], self.action_dim, num_action_skill, prediction_mlp_layers,
                       projection_mlp_layers).to(kwargs['device'])
        # loss criterion
        self.atr_criterion = nn.CrossEntropyLoss()
        # optimizers
        self.atr_opt = torch.optim.Adam(self.atr.parameters(), lr=self.lr)
        self.intrinsic_reward_rms = utils.RMS(device=self.device)

        self.atr.train()

    def get_meta_specs(self):
        return (specs.Array([self.action_dim for i in range(self.num_action_skill)], np.float32, 'action_trail'),
                specs.Array([self.obs_dim for i in range(self.num_action_skill)], np.float32, 'obs_trail'))

    def init_meta(self):
        obs_trail = np.zeros((self.num_action_skill, self.obs_dim), dtype=np.float32)
        action_trail = np.zeros((self.num_action_skill, self.action_dim), dtype=np.float32)
        meta = OrderedDict()
        meta['action_trail'] = action_trail
        meta['obs_trail'] = obs_trail
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        if global_step % self.num_action_skill == 0:
            return self.init_meta()
        else:
            meta['action_trail'][:, :-1] = meta['action_trail'][:, 1:]
            meta['action_trail'][:, -1] = torch.from_numpy(time_step.action)
            meta['obs_trail'][:, :-1] = meta['obs_trail'][:, 1:]
            meta['obs_trail'][:, -1] = torch.from_numpy(time_step.observation)
        return meta

    def update_atr(self, action_x, obs_x, step):
        metrics = dict()

        loss, weighted_inv, weighted_var, weighted_cov = self.compute_atr_loss(action_x, obs_x)

        self.atr_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.atr_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['atr_loss'] = loss.item()
            metrics['weighted_inv'] = weighted_inv
            metrics['weighted_var'] = weighted_var
            metrics['weighted_cov'] = weighted_cov


        return metrics



    def compute_intr_reward(self, action_trail, obs_trail, action, next_obs, step):
        prediction_error, _, _, _ = self.atr(action_trail, obs_trail, action, next_obs, step)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = self.atr_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + 1e-8)
        return reward

    def compute_atr_loss(self, action_x, obs_x):
        """
        DF Loss
        """

        _, z_a, z_b = self.atr(action_x, obs_x)
        assert z_a.shape == z_b.shape and len(z_a.shape) == 2

        # invariance loss
        loss_inv = F.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.variance_loss_epsilon)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.variance_loss_epsilon)
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        # covariance loss
        N, D = z_a.shape
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        weighted_inv = loss_inv * self.invariance_loss_weight
        weighted_var = loss_var * self.variance_loss_weight
        weighted_cov = loss_cov * self.covariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov

        return loss, weighted_inv, weighted_var, weighted_cov

    def prepare_unsupervised_data(self, action_trail, obs_trail):
        action_x = torch.cat(action_trail, 1)
        obs_x = torch.cat((obs_trail[:, 0], obs_trail[:, -1]), 1)

        return action_x, obs_x

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, action_trail, obs_trail = utils.to_torch(
            batch, self.device)

        # augment and encode
        action_x, obs_x = self.prepare_unsupervised_data(action_trail, obs_trail)

        if self.reward_free:
            metrics.update(self.update_atr(action_x, obs_x, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(action_x, obs_x, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
