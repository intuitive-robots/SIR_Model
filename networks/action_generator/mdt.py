import logging
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from networks.action_generator.utils import SinusoidalPosEmb, append_dims
from networks.transformer.transformer_decoders import *
from networks.transformer.transformer_encoders import *
from networks.transformer.transformer_layers import *

logger = logging.getLogger(__name__)

class MDT(nn.Module):
    """
    A Karras et al. preconditioner for denoising diffusion models.

    Args:
        model: The model used for denoising.
        sigma_data: The data sigma for scalings (default: 1.0).
    """
    def __init__(self,
                sigma_data: float,
                obs_seq_len: int,
                action_dim: int,
                proprio_dim: int,
                img_dim: int,
                graph_dim: int,
                goal_dim: int,
                img_mod: list,
                graph_mod: list,
                use_graph_fusion: bool,
                embed_dim: int,
                embed_pdrob: float,
                attn_pdrop: float,
                resid_pdrop: float,
                mlp_pdrop: float,
                n_dec_layers: int,
                n_enc_layers: int,
                n_heads: int,
                bias: bool = False,
                use_noise_encoder: bool = False,
                ):
        super().__init__()
        self.model = MDT_Transformer(
            obs_seq_len=obs_seq_len,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            img_dim=img_dim,
            graph_dim=graph_dim,
            goal_dim=goal_dim,
            img_mod=img_mod,
            graph_mod=graph_mod,
            use_graph_fusion=use_graph_fusion,
            embed_dim=embed_dim,
            embed_pdrob=embed_pdrob,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            n_dec_layers=n_dec_layers,
            n_enc_layers=n_enc_layers,
            n_heads=n_heads,
            bias=bias,
            use_noise_encoder=use_noise_encoder,
        )
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        """
        Compute the scalings for the denoising process.

        Args:
            sigma: The input sigma.
        Returns:
            The computed scalings for skip connections, output, and input.
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, state, action, goal, noise, sigma):
        """
        Compute the loss for the denoising process.

        Args:
            state: The input state.
            action: The input action.
            goal: The input goal.
            noise: The input noise.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.
        Returns:
            The computed loss.
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        noised_input = action + noise * append_dims(sigma, action.ndim)
        model_output = self.model(state, noised_input * c_in, goal, sigma)
        target = (action - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(), model_output

    def forward(self, state, action, goal, sigma):
        """
        Perform the forward pass of the denoising process.

        Args:
            state: The input state.
            action: The input action.
            goal: The input goal.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass.
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.model(state, action * c_in, goal, sigma) * c_out + action * c_skip

    def get_params(self):
        return self.model.parameters()
    
class MDT_Transformer(nn.Module):
    def __init__(
        self,
        obs_seq_len: int,
        action_dim: int,
        proprio_dim: int,
        img_dim: int,
        graph_dim: int,
        goal_dim: int,
        img_mod: list,
        graph_mod: list,
        use_graph_fusion: bool,
        embed_dim: int,
        embed_pdrob: float,
        attn_pdrop: float,
        resid_pdrop: float,
        mlp_pdrop: float,
        n_dec_layers: int,
        n_enc_layers: int,
        n_heads: int,
        bias: bool = False,
        use_noise_encoder: bool = False,
    ):
        super().__init__()
        
        self.use_proprioceptive = True if proprio_dim > 0 else False
        self.proprio_dim = proprio_dim
        
        self.use_image = True if len(img_mod) > 0 else False
        self.image_modalities = img_mod
        
        self.use_graph = True if len(graph_mod) > 0 else False
        self.graph_modalities = graph_mod
        
        seq_size = self.get_seq_length(obs_seq_len)
        
        if self.use_proprioceptive: # Bigger MLP, because it functions as embedding network
            self.proprio_emb = nn.Sequential(
                nn.Linear(self.proprio_dim, embed_dim * 2),
                nn.Mish(),
                nn.Linear(embed_dim * 2, embed_dim * 2),
                nn.Mish(),
                nn.Linear(embed_dim * 2, embed_dim * 2),
                nn.Mish(),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        
        # image token ebmedders
        if self.use_image:
            self.img_emb_dict = nn.ModuleDict()
            for m in img_mod:
                self.img_emb_dict[m] = nn.Linear(img_dim, embed_dim)
            
        # graph token embedders
        if self.use_graph:
            self.graph_emb_dict = nn.ModuleDict()
            for m in graph_mod:
                self.graph_emb_dict[m] = nn.Linear(graph_dim, embed_dim)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        
        self.goal_emb = nn.Sequential(
            nn.Linear(goal_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Encoder takes all observation tokens and creates one context embedding
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_enc_layers,
            bias=bias,
            mlp_pdrop=mlp_pdrop,
        )

        # Decoder takes the context embedding and the noise value to produce a action prediction for the diffusion process
        self.decoder = TransformerFiLMDecoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            n_layers=n_dec_layers,
            film_cond_dim=embed_dim,
            bias=bias,
            mlp_pdrop=mlp_pdrop,
            use_cross_attention=True,
            use_noise_encoder=use_noise_encoder,
        )

        self.sigma_emb = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.action_emb = nn.Linear(action_dim, embed_dim)
        self.action_pred = nn.Linear(embed_dim, action_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, MDT_Transformer):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def forward(self, states, actions, goals, sigma):
        context = self.enc_only_forward(states, goals)
        pred_actions = self.dec_only_forward(context, actions, sigma)
        return pred_actions

    def enc_only_forward(self, states, goals):
        obs_embeddings = self.process_state_embeddings(states)
        goal_embed = self.goal_emb(torch.unsqueeze(goals['lang'], dim=1))
                
        goal_x, obs_x = self.apply_position_embeddings(goal_embed, obs_embeddings)
        
        input_seq = torch.concatenate((goal_x, obs_x), dim=1)
        
        context = self.encoder(input_seq)
        
        return context

    def dec_only_forward(self, context, actions, sigma):
        emb_t = self.process_sigma_embeddings(sigma)
        action_embed = self.action_emb(actions)
        action_x = self.drop(action_embed)

        x = self.decoder(action_x, emb_t, context)

        pred_actions = self.action_pred(x)
        return pred_actions

    def get_params(self):
        return self.parameters()

    def get_seq_length(self, obs_length):
        seq_len = 1 # the one is necessary, because of the goal
        for _ in self.image_modalities + self.graph_modalities:
            seq_len += obs_length
        if self.use_proprioceptive:
            seq_len += obs_length
            
        return seq_len

    def process_sigma_embeddings(self, sigma):
        sigmas = sigma.log() / 4
        sigmas = rearrange(sigmas, 'b -> b 1')
        emb_t = self.sigma_emb(sigmas)
        if len(emb_t.shape) == 2:
            emb_t = rearrange(emb_t, 'b d -> b 1 d')
        return emb_t

    def process_state_embeddings(self, states):
        processed_states = []
        
        if self.use_proprioceptive:
            proprio_embed = self.proprio_emb(states['obs_prop'])
            processed_states.append(proprio_embed)
        
        if self.use_image:
            for m in self.image_modalities:
                img_embed = self.img_emb_dict[m](states['obs_img'][m])
                processed_states.append(img_embed)
                
        if self.use_graph:
            for m in self.graph_modalities:
                graph_embed = self.graph_emb_dict[m](states['obs_graph'][m])
                processed_states.append(graph_embed)
                
        state_embeddings = torch.concatenate(processed_states, dim=1)
        
        return state_embeddings

    def apply_position_embeddings(self, goal_embed, state_embed):
        goal_x = self.drop(goal_embed + self.pos_emb[:, :1, :])
        
        obs_x = self.drop(state_embed + self.pos_emb[:, 1:state_embed.shape[1]+1, :])
        
        return goal_x, obs_x