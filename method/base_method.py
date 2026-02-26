import abc
import hydra
from omegaconf import DictConfig
import clip
import torch

import logging

log = logging.getLogger(__name__)

class Base_Method(abc.ABC):
    def __init__(self,
                action_generator: DictConfig,
                vision_encoder: DictConfig,
                graph_encoder: DictConfig,
                optimization: DictConfig,
                lr_scheduler: DictConfig,
                device: str,
                ):
        super().__init__()
        
        self.models = []
        self.model_names = []
        
        self.device = device
        
        self.action_generator = hydra.utils.instantiate(action_generator).to(self.device)
        self.models.append(self.action_generator)
        self.model_names.append('action_generator')
        
        self.language_encoder, _ = clip.load("ViT-B/32", device=device)
        
        self.vision_encoder_config = vision_encoder
        self.graph_encoder_config = graph_encoder
        
        self.optimization_config = optimization
        self.lr_scheduler_config = lr_scheduler
    
    def instantiate_proprioceptive(self):
        pass
    
    def instantiate_vision_encoder(self):
        self.vision_encoder = hydra.utils.instantiate(self.vision_encoder_config).to(self.device)
        self.models.append(self.vision_encoder)
        self.model_names.append('vision_encoder')
    
    def instantiate_graph_encoder(self):
        self.graph_encoder = hydra.utils.instantiate(self.graph_encoder_config).to(self.device)
        self.models.append(self.graph_encoder)
        self.model_names.append('graph_encoder')
        
    def instantiate_optimizer(self):
        params = []
        total_param_count = 0
        for i in range(len(self.models)):
            params += list(self.models[i].parameters())
            param_count = sum(p.numel() for p in self.models[i].parameters())
            total_param_count += param_count
            log.info("number of parameters " + self.model_names[i] + ": %e", param_count)
        log.info("number of parameters: %e", total_param_count)
        self.optimizer = hydra.utils.instantiate(self.optimization_config, params=params)
        self.lr_scheduler = hydra.utils.instantiate(self.lr_scheduler_config, optimizer=self.optimizer)
    
    def train(self):
        for model in self.models:
            model.train()
    
    def eval(self):
        for model in self.models:
            model.eval()
    
    def save_models(self, dir, add_name):
        for model, name in zip(self.models, self.model_names):
            torch.save(model.state_dict(), f"{dir}/{name}{add_name}.pth")
            log.info(f"Model saved to {dir}/{name}_{add_name}.pth")
    
    def load_models(self, dir, add_name):
        for model, name in zip(self.models, self.model_names):
            model.load_state_dict(torch.load(f"{dir}/{name}{add_name}.pth", map_location=self.device))
            log.info(f"Model loaded from {dir}/{name}{add_name}.pth")

    def clip_action(self, y, y_bounds):
        """
        Clips the input tensor `y` based on the defined action bounds.
        """
        y_bounds = y_bounds.to(self.device)
        return torch.clamp(y, y_bounds[0, :]*1.1, y_bounds[1, :]*1.1).to(self.device).to(torch.float32)
    
    @abc.abstractmethod
    def preprocess_batch(self, batch):
        """
        Method to preprocess the batch before feeding it to the model
        """
        pass
    
    @abc.abstractmethod
    @torch.no_grad()
    def compute_validation_loss(self, state, action, goal):
        """
        Method to compute the validation loss
        """
        pass
    
    @abc.abstractmethod
    def compute_training_loss(self, state, action, goal):
        """
        Method to compute the training loss
        """
        pass
    
    @abc.abstractmethod
    @torch.no_grad()
    def predict(self, state, goal):
        """
        Method to predict the action given the state and goal
        """
        pass