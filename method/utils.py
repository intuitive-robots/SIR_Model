import math
import torch
from torch.optim.lr_scheduler import LinearLR, StepLR, SequentialLR
from tqdm.auto import trange
import copy

class WarmupSequentialLRScheduler(SequentialLR):
    def __init__(
        self, 
        optimizer, 
        epochs: int, 
        warmup_ratio: float = 0.05, 
        step_ratio: float = 0.1, 
        gamma: float = 0.5, 
        start_factor: float = 0.1
    ):
        # 1. Calculate dynamic milestones based on total epochs
        warmup_epochs = int(epochs * warmup_ratio)
        main_step_size = int(epochs * step_ratio)
        
        if warmup_epochs < 1:
            warmup_epochs = 1
        if main_step_size < 1:
            main_step_size = 1
        
        # 2. Instantiate the sub-schedulers using the passed optimizer
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=start_factor, 
            total_iters=warmup_epochs
        )
        
        main_scheduler = StepLR(
            optimizer, 
            step_size=main_step_size, 
            gamma=gamma
        )

        # 3. Initialize the parent SequentialLR
        super().__init__(
            optimizer=optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[warmup_epochs]
        )
        
def rand_log_logistic(shape, loc=0., scale=1., min_value=0., max_value=float('inf'), device='cpu', dtype=torch.float32):
    """Draws samples from an optionally truncated log-logistic distribution."""
    min_value = torch.as_tensor(min_value, device=device, dtype=torch.float64)
    max_value = torch.as_tensor(max_value, device=device, dtype=torch.float64)
    min_cdf = min_value.log().sub(loc).div(scale).sigmoid()
    max_cdf = max_value.log().sub(loc).div(scale).sigmoid()
    u = torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf) + min_cdf
    return u.logit().mul(scale).add(loc).exp().to(dtype)

def sample_ddim(
    model, 
    state, 
    action, 
    goal, 
    sigmas,
    disable,
):
    """
    DPM-Solver
    """
    s_in = action.new_ones([action.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        # predict the next action
        denoised = model(state, action, goal, sigmas[i] * s_in)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        action = (sigma_fn(t_next) / sigma_fn(t)) * action - (-h).expm1() * denoised
    return action

def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)

def append_zero(action):
    return torch.cat([action, action.new_zeros([1])])

class EMA:
    def __init__(self, models, decay=0.999):
        """
        Args:
            models: A single nn.Module or a list/dict of nn.Modules to track.
            decay: The decay factor (usually 0.999 or 0.9999).
        """
        self.decay = decay
        self.models = models if isinstance(models, (list, tuple)) else [models]
        
        # Shadow dictionary to store the averaged weights
        # Key: (model_idx, param_name), Value: shadow_param
        self.shadow = {} 
        self.backup = {} # To store original weights during evaluation

        # Initialize shadow weights
        self.register()

    def register(self):
        """Initialize shadow weights as a copy of the current model weights."""
        for i, model in enumerate(self.models):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Clone and detach to ensure it's a separate tensor
                    self.shadow[(i, name)] = param.data.clone().detach().to(param.device)

    def update(self):
        """Update shadow weights: shadow = decay * shadow + (1 - decay) * new_param"""
        for i, model in enumerate(self.models):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    key = (i, name)
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[key]
                    self.shadow[key] = new_average.clone()

    def apply_shadow(self):
        """
        Backup current weights and load EMA weights into the model.
        Call this before Validation/Testing.
        """
        for i, model in enumerate(self.models):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    key = (i, name)
                    self.backup[key] = param.data.clone()
                    param.data = self.shadow[key].to(param.device)

    def restore(self):
        """
        Restore original weights from backup.
        Call this after Validation/Testing to resume training.
        """
        for i, model in enumerate(self.models):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    key = (i, name)
                    param.data = self.backup[key]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict