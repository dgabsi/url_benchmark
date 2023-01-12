import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
from functools import partial

import utils
from agent.ddpg import DDPGAgent, Encoder


class ATR_RND(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, action_dim, num_action_skill, prediction_mlp_layers, projection_mlp_layers, device):
        super().__init__()
        self.device=device
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim*num_action_skill, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.obs_encoder = nn.Sequential(nn.Linear(2 * obs_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim))
        self.apply(utils.weight_init)

        self.action_projection_model = utils.MLP(
            hidden_dim,
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
            hidden_dim,
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

        im_a = action_x.contiguous().to(self.device)
        im_o = obs_x.contiguous().to(self.device)

        # compute query features
        emb_a = self.action_encoder(im_a)
        a_projection = self.action_projection_model(emb_a)
        a = self.action_prediction_model(a_projection)  # queries: NxC
        emb_o = self.obs_encoder(im_o)
        o_projection = self.obs_projection_model(emb_o)
        o = self.obs_prediction_model(o_projection)  # queries: NxC

        return emb_a, a, o


class ATR_RNDAgent(DDPGAgent):
    def __init__(self, skill_dim, num_action_skill, prediction_mlp_layers, projection_mlp_layers, atr_scale,
                 update_encoder, variance_loss_epsilon, invariance_loss_weight, variance_loss_weight,
                 covariance_loss_weight, **kwargs):
        self.skill_dim = skill_dim
        self.num_action_skill = num_action_skill
        self.atr_scale = atr_scale
        self.update_encoder = update_encoder
        self.variance_loss_epsilon= variance_loss_epsilon
        self.invariance_loss_weight= invariance_loss_weight
        self.variance_loss_weight= variance_loss_weight
        self.covariance_loss_weight= covariance_loss_weight

        kwargs["meta_dim"] = self.skill_dim
        # create actor and critic
        super().__init__(**kwargs)

        self.atr_encoder = Encoder(self.obs_shape).to(self.device)
        # create atr
        self.atr = ATR_RND(self.obs_dim-self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim'], self.action_dim, num_action_skill, prediction_mlp_layers,
                       projection_mlp_layers, kwargs['device']).to(kwargs['device'])

        # loss criterion
        #self.atr_criterion = nn.CrossEntropyLoss()
        # optimizers
        self.atr_opt = torch.optim.Adam(self.atr.parameters(), lr=self.lr)
        self.intrinsic_reward_rms = utils.RMS(device=self.device)

        self.atr.train()

    def get_meta_specs(self):
        return ( specs.Array((self.skill_dim,), np.float32, 'skill'),
            specs.Array((*[self.action_dim for i in range(self.num_action_skill)],), np.float32, 'action_trail'),
                specs.Array((*[self.obs_dim-self.skill_dim for i in range(self.num_action_skill)],), np.float32, 'obs_trail'))

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        obs_trail = np.zeros((self.num_action_skill, *self.obs_shape), dtype=np.float32)
        action_trail = np.zeros((self.num_action_skill, self.action_dim), dtype=np.float32)
        meta = OrderedDict()
        meta['skill'] = skill
        meta['action_trail'] = action_trail
        meta['obs_trail'] = obs_trail
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        self.atr.eval()
        if global_step % self.num_action_skill == 0:
            return self.init_meta()
        else:
            meta['action_trail'][:-1] = meta['action_trail'][1:]
            meta['action_trail'][-1] = time_step.action
            meta['obs_trail'][:-1] = meta['obs_trail'][1:]
            obs = torch.as_tensor(time_step.observation, device=self.device).unsqueeze(dim=0)
            #obs_h = self.atr_encoder(obs).detach().cpu().numpy() #.squeeze()
            meta['obs_trail'][-1] = obs.detach().cpu().numpy()
            action_x, obs_x = self.prepare_unsupervised_data(meta['action_trail'], meta['obs_trail'])
            _, z_a, _ = self.atr(torch.from_numpy(action_x), torch.from_numpy(obs_x))
            meta['skill'] = z_a.detach().cpu().numpy().squeeze()
        self.atr.train()
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



    def compute_intr_reward(self, action_x, obs_x, step):
        batch_size = action_x.size()[0]
        prediction_error, _, _, _ = self.compute_atr_loss(action_x, obs_x)
        intr_reward_var = prediction_error #self.intrinsic_reward_rms(prediction_error)
        reward = self.atr_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + 1e-8)
        return reward.repeat(batch_size).unsqueeze(dim=1)/batch_size

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
        if action_trail.ndim>2:
            batch_size= action_trail.shape[0]
            action_x = action_trail.reshape(batch_size, -1)
            obs_x = torch.cat((obs_trail[:, 0], obs_trail[:, -1]), dim=1)
        else:
            action_x = np.reshape(action_trail, (1, -1))
            obs_x = np.concatenate((obs_trail[0], obs_trail[-1])).reshape(1, -1)

        return action_x, obs_x

    def update(self, replay_iter, step):
        metrics = dict()
        #self.atr.train()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill, action_trail, obs_trail = utils.to_torch(
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
        obs = torch.as_tensor(obs, device=self.device)
        h_obs = self.encoder(obs)
        next_obs= torch.as_tensor(next_obs, device=self.device)
        h_next_obs = self.encoder(next_obs)
        obs = torch.cat([h_obs, skill], dim=-1)
        next_obs = torch.cat([h_next_obs, skill], dim=-1)

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
        value = meta['skill']
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
