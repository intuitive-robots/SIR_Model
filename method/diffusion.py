import math
import clip
from einops import einops
from omegaconf import DictConfig
import torch
import torch.nn as nn
from functools import partial

from method import utils
from method.base_method import Base_Method

class Diffusion(Base_Method):
    def __init__(self,
                action_generator: DictConfig,
                vision_encoder: DictConfig,
                graph_encoder: DictConfig,
                optimization: DictConfig,
                lr_scheduler: DictConfig,
                device: str,
                act_window_size: int,
                obs_window_size: int,
                action_dim: int,
                sampler_type: str,
                num_sampling_steps: int,
                noise_scheduler: str,
                sigma_data: float,
                sigma_min: float,
                sigma_max: float,
                sigma_sample_density_type: str,
                ):
        super().__init__(action_generator,
                         vision_encoder,
                         graph_encoder,
                         optimization,
                         lr_scheduler,
                         device)
        
        self.act_window_size = act_window_size
        self.obs_window_size = obs_window_size
        self.action_dim = action_dim
        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        
    def preprocess_batch(self, batch):
        
        goal = {}
        state = {}
        action = {}
        
        lang_goal = batch['goal']['lang']
        if isinstance(lang_goal, str):
            lang_goal_tokenized = clip.tokenize(lang_goal).to(self.device)
            lang_goal = self.language_encoder.encode_text(lang_goal_tokenized).to(torch.float32)
        goal['lang'] = lang_goal.to(self.device)
        
        for key in batch['observation']:
            for k in batch["observation"][key]:
                if key == "obs_graph":
                    batch["observation"][key][k] = batch["observation"][key][k].to(self.device)
                else:
                    batch["observation"][key][k] = batch["observation"][key][k].to(self.device).float()
                    batch["observation"][key][k] = einops.rearrange(batch["observation"][key][k], "b w ... -> (b w) ...")
            if key == "obs_prop":
                state[key] = batch["observation"][key].to(self.device).float()
            elif key == "obs_img":
                state[key] = self.vision_encoder(batch["observation"][key])
            elif key == "obs_graph":
                state[key] = self.graph_encoder(batch["observation"][key])
            else:
                raise NotImplementedError(f"Modality {key} not implemented in diffusion method.")
        
            for k in batch["observation"][key]:
                batch["observation"][key][k] = einops.rearrange(batch["observation"][key][k], "(b w) ... -> b w ...", w=self.obs_window_size)
            
        if "action" not in list(batch.keys()):
            action['action'] = None
        else:  
            action['action'] = batch["action"]['eef'].to(self.device).float()
            
        return state, action, goal
    
    def compute_training_loss(self, state, action, goal):
        loss_dict = {}
        
        loss_dict['diffusion_loss'] = self.diffusion_loss(perceptual_emb=state, latent_goal=goal, actions=action)
        
        loss_dict['total_loss'] = loss_dict['diffusion_loss']
        
        return loss_dict
    
    def compute_validation_loss(self, state, action, goal):
        loss_dict = {}
        
        action_pred = self.denoise_actions(perceptual_emb=state, latent_goal=goal)
        
        mse_loss = nn.functional.mse_loss(action_pred, action)
        
        loss_dict['mse_loss'] = mse_loss
        loss_dict['total_loss'] = loss_dict['mse_loss']
        
        return loss_dict
    
    def predict(self, state, goal):
        action_pred = self.denoise_actions(perceptual_emb=state, latent_goal=goal)
            
        return action_pred
    
    def diffusion_loss(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device)
        noise = torch.randn_like(actions).to(self.device)
        loss, _ = self.action_generator.loss(perceptual_emb, actions, latent_goal, noise, sigmas)
        return loss
    
    def denoise_actions(
        self,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
    ):
        """
        Denoise the next sequence of actions 
        """        
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(self.num_sampling_steps, self.noise_scheduler)
        x = torch.randn((latent_goal['lang'].shape[0], self.act_window_size, self.action_dim), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, latent_goal, self.sampler_type)

        return actions
    
    def sample_loop(
        self, 
        sigmas, 
        x_t: torch.Tensor,
        state: torch.Tensor, 
        goal: torch.Tensor, 
        sampler_type: str,
        ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """

        if sampler_type == 'ddim':
            x_0 = utils.sample_ddim(self.action_generator, state, x_t, goal, sigmas, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0
    
    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'exponential':
            return utils.get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        else:
            raise ValueError('Unknown noise schedule type')
    
    def make_sample_density(self):
        if self.sigma_sample_density_type == 'loglogistic':
            loc = math.log(self.sigma_data)
            scale = 0.5
            min_value = self.sigma_min
            max_value = self.sigma_max
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
        else:
            raise NotImplementedError(f"Sigma sample density type {self.sigma_sample_density_type} not implemented.")