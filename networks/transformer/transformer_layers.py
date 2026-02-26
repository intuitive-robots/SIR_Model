import logging
import math 
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
import einops

from .position_embedding import *
from .utils import RMSNorm, LayerNorm, SwishGLU


class Attention(nn.Module):
    def __init__(
        self, 
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        causal: bool = False,
        bias=False,
        use_rot_embed: bool = False,
        use_relative_pos: bool = False,
        rotary_xpos: bool = False,
        rotary_emb_dim=None,
        rotary_xpos_scale_base=512,
        rotary_interpolation_factor=1.,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash and causal:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
        
        self.use_relative_pos = use_relative_pos
        self.use_rot_embed = use_rot_embed
        
        assert not (use_relative_pos and use_rot_embed), "Can't use both relative position and rotary embedding"
        if self.use_relative_pos:
            self.pos_emb = RelativePositionBias(n_embd, n_head)
        if self.use_rot_embed:
            rotary_emb_dim = max(rotary_emb_dim or self.n_head // 2, 32)
            self.rotary_pos_emb = RotaryEmbedding(
                rotary_emb_dim, 
                freqs_for='constant',
                use_xpos=rotary_xpos, 
                xpos_scale_base=rotary_xpos_scale_base, 
                interpolate_factor=rotary_interpolation_factor, 
            ) 

    def forward(self, x, context=None, custom_attn_mask=None):
        B, T, C = x.size()

        if context is not None:
            k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.use_rot_embed:
            q = self.rotary_pos_emb.rotate_queries_or_keys(q)
            k = self.rotary_pos_emb.rotate_queries_or_keys(k)

        if self.flash:
            # boolean or 1 and 0 attention mask
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=custom_attn_mask, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                if custom_attn_mask is not None:
                    att = att.masked_fill(custom_attn_mask == 0, float('-inf'))
                else:
                    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y



class MLP(nn.Module):

    def __init__(
            self, 
            n_embd: int,
            bias: bool,
            use_swish: bool = True,
            use_relus: bool = False,
            dropout: float = 0
        ):
        super().__init__()
        layers = []        
        if use_swish:
            layers.append(SwishGLU(n_embd, 4 * n_embd))
        else:
            layers.append(nn.Linear(n_embd, 4 * n_embd, bias=bias))
            if use_relus:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4 * n_embd, n_embd, bias=bias))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    

class Block(nn.Module):

    def __init__(
            self, 
            n_embd: int, 
            n_heads: int, 
            attn_pdrop: float, 
            resid_pdrop: float, 
            mlp_pdrop: float,
            causal: bool,
            use_rms_norm: bool = False,
            use_cross_attention: bool = False,
            use_relative_pos: bool = False,
            use_rot_embed: bool = False,
            rotary_xpos: bool = False,
            bias: bool = False, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        ):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd) if use_rms_norm else LayerNorm(n_embd, bias=bias)
        self.attn = Attention(n_embd, n_heads, attn_pdrop, resid_pdrop, causal, bias, use_rot_embed, rotary_xpos)
        self.use_cross_attention = use_cross_attention
        # cross attention is only used in the decoder blocks
        if self.use_cross_attention:
            self.cross_att = Attention(
                n_embd, 
                n_heads, 
                attn_pdrop, 
                resid_pdrop, 
                causal, 
                bias, 
                use_rot_embed, 
                use_relative_pos,
                rotary_xpos
            )
            self.ln_3 = RMSNorm(n_embd) if use_rms_norm else LayerNorm(n_embd, bias=bias)
        self.ln_2 = RMSNorm(n_embd) if use_rms_norm else LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, mlp_pdrop)

    def forward(self, x, context=None, custom_attn_mask=None):
        x = x + self.attn(self.ln_1(x), custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln_3(x), context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero modulation for conditioning.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Initialize weights and biases to zero
        # nn.init.zeros_(self.modulation[1].weight)
        # nn.init.zeros_(self.modulation[1].bias)

    def forward(self, c):
        return self.modulation(c).chunk(6, dim=-1)

def modulate(x, shift, scale):
    return shift + (x * (scale))





class ConditionedBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop, 
            causal, 
            film_cond_dim,
            use_cross_attention=False, 
            use_rot_embed=False, 
            use_relative_pos=False,
            rotary_xpos=False, 
            bias=False # and any other arguments from the Block class
        ):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, causal,
                         use_cross_attention=use_cross_attention, 
                         use_rot_embed=use_rot_embed, 
                         use_relative_pos=use_relative_pos,
                         rotary_xpos=rotary_xpos, 
                         bias=bias)
        self.adaLN_zero = AdaLNZero(film_cond_dim)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_zero(c)
        
        # Attention with modulation
        x_attn = self.ln_1(x)
        x_attn = modulate(x_attn, shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_attn, custom_attn_mask=custom_attn_mask)
        
        # Cross attention if used
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln_3(x), context, custom_attn_mask=custom_attn_mask)
        
        # MLP with modulation
        x_mlp = self.ln_2(x)
        x_mlp = modulate(x_mlp, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mlp)
        
        return x

class NoiseBlock(Block):
    """
    Block with AdaLN-Zero conditioning.
    """
    def __init__(
            self, 
            n_embd, 
            n_heads, 
            attn_pdrop, 
            resid_pdrop, 
            mlp_pdrop, 
            causal, 
            use_rms_norm: bool=False,
            use_cross_attention=False, 
            use_rot_embed=False, 
            use_relative_pos=False,
            rotary_xpos=False, 
            bias=False # and any other arguments from the Block class
        ):
        super().__init__(n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_pdrop, causal,
                         use_cross_attention=use_cross_attention, 
                         use_rot_embed=use_rot_embed, use_rms_norm=use_rms_norm,
                         use_relative_pos=use_relative_pos,
                         rotary_xpos=rotary_xpos, 
                         bias=bias)

    def forward(self, x, c, context=None, custom_attn_mask=None):
        
        x = x + self.attn(self.ln_1(x) + c, custom_attn_mask=custom_attn_mask)
        if self.use_cross_attention and context is not None:
            x = x + self.cross_att(self.ln3(x) + c, context, custom_attn_mask=custom_attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    


# As defined in Set Transformers () -- basically the above, additionally taking in
# a set of $k$ learned "seed vectors" that are used to "pool" information.
class MAPAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int) -> None:
        """Multi-Input Multi-Headed Attention Operation"""
        super().__init__()
        assert embed_dim % n_heads == 0, "`embed_dim` must be divisible by `n_heads`!"
        self.n_heads, self.scale = n_heads, (embed_dim // n_heads) ** -0.5

        # Projections (no bias) --> separate for Q (seed vector), and KV ("pool" inputs)
        self.q, self.kv = nn.Linear(embed_dim, embed_dim, bias=False), nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, seed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        (B_s, K, C_s), (B_x, N, C_x) = seed.shape, x.shape
        assert C_s == C_x, "Seed vectors and pool inputs must have the same embedding dimensionality!"

        # Project Seed Vectors to `queries`
        q = self.q(seed).reshape(B_s, K, self.n_heads, C_s // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B_x, N, 2, self.n_heads, C_x // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention --> compute weighted sum over values!
        scores = q @ (k.transpose(-2, -1) * self.scale)
        attn = scores.softmax(dim=-1)
        vals = (attn @ v).transpose(1, 2).reshape(B_s, K, C_s)

        # Project back to `embed_dim`
        return self.proj(vals)


class MAPBlock(nn.Module):
    def __init__(
        self,
        n_latents: int,
        embed_dim: int,
        n_heads: int,
        output_dim: None,
        mlp_ratio: float = 4.0,
        do_rms_norm: bool = True,
        do_swish_glu: bool = True,
    ) -> None:
        """Multiheaded Attention Pooling Block -- note that for MAP, we adopt earlier post-norm conventions."""
        super().__init__()
        self.n_latents, self.embed_dim, self.n_heads = n_latents, embed_dim, 2 * n_heads

        self.embed_dim = output_dim
        # Projection Operator
        self.projection = nn.Linear(embed_dim, self.embed_dim)

        # Initialize Latents
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.embed_dim))
        nn.init.normal_(self.latents, std=0.02)

        # Custom MAP Attention (seed, encoder outputs) -> seed
        self.attn_norm = RMSNorm(self.embed_dim) if do_rms_norm else LayerNorm(self.embed_dim, eps=1e-6)
        self.attn = MAPAttention(self.embed_dim, n_heads=self.n_heads)
        if output_dim is None:
            output_dim = self.embed_dim
        # Position-wise Feed-Forward Components
        self.mlp_norm = RMSNorm(self.embed_dim) if do_rms_norm else LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            # Handle SwishGLU vs. GELU MLP...
            (
                SwishGLU(self.embed_dim, int(mlp_ratio * self.embed_dim))
                if do_swish_glu
                else nn.Sequential(nn.Linear(self.embed_dim, int(mlp_ratio * self.embed_dim)), nn.GELU())
            ),
            nn.Linear(int(mlp_ratio * self.embed_dim), self.embed_dim),
        )

    def forward(self, x: torch.Tensor, batch=None) -> torch.Tensor:
        latents = repeat(self.latents, "n_latents d -> bsz n_latents d", bsz=x.shape[0])
        latents = self.attn_norm(latents + self.attn(latents, self.projection(x)))
        latents = self.mlp_norm(latents + self.mlp(latents))
        return latents.squeeze(dim=1)