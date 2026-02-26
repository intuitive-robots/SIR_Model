import logging
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import random
from tqdm import tqdm
import wandb
import os

from method.utils import EMA

log = logging.getLogger(__name__)

class Trainer():
    def __init__(self, manager, method, config):
        self.seed = config.seed
        self.device = config.device
        self.epochs = config.epochs
        self.train_bool = config.train_bool
        self.test_during_training_bool = config.test_during_training_bool
        self.test_bool = config.test_bool
        self.log_dir = config.log_dir
        self.model_path = config.model_path
        self.eval_n_times = config.eval_n_times
        self.store_videos = config.store_videos
        self.use_ema = config.use_ema
        
        # set seed for random stuff
        # Note: GNNs are non-deterministic by nature when executing them on the GPU -> https://github.com/pyg-team/pytorch_geometric/issues/92
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

        log.info(f'Used seed: {self.seed}')
        
        manager, method = self.check_if_model_is_loaded(manager, method)
        
        self.manager = hydra.utils.instantiate(manager)
        method = self.set_runtime_parameter(method)
        self.method = hydra.utils.instantiate(method)
        
        self.working_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    
    def check_if_model_is_loaded(self, manager, method):
        new_task_names = manager.task_names
        new_datapath = manager.data_path
        self.load_model_bool = not self.train_bool
        if self.load_model_bool:
            if os.path.exists(self.model_path + "/.hydra/config.yaml"):
                cfg_reloaded = OmegaConf.load(self.model_path + "/.hydra/config.yaml")
            else:
                raise FileNotFoundError("Could not find config file to reload model parameters.")
            manager = cfg_reloaded.manager
            manager.load_dataset = False
            manager.task_names = new_task_names
            manager.data_path = new_datapath
            method = cfg_reloaded.method
            method.device = self.device
            
            self.use_ema = cfg_reloaded.trainer.use_ema
        
        return manager, method
    
    def start(self):
        self.instantiate_models()
        
        if self.train_bool:
            log.info("Starting training...")
            self.train()
            log.info("Training done.")
        
        if self.test_bool:
            if self.load_model_bool:
                self.load_model(epoch=None)
            log.info("Starting testing...")
            self.test()
            log.info("Testing done.")
    
    def train(self):
        for epoch in tqdm(range(self.epochs)):
            log.info(f"Epoch {epoch+1}/{self.epochs}")
            
            if self.use_ema:
                self.ema.apply_shadow()
            
            val_loss_accum = {}
            for val_batch in tqdm(self.manager.valid_dataloader):
                val_loss_dic = self.val_step(val_batch)
                for k, v in val_loss_dic.items():
                    if k not in val_loss_accum:
                        val_loss_accum[k] = 0.0
                    val_loss_accum[k] += v.item()
            
            if self.use_ema:
                self.ema.restore()
            
            val_log_dict = {
                f"val_{k}": (v / len(self.manager.valid_dataloader)) 
                for k, v in val_loss_accum.items()
            }
            
            wandb.log(val_log_dict, step=epoch)
            for k, v in val_log_dict.items():
                log.info(f"{k}: {v:.6f}")
            
            train_loss_accum = {}
            for batch in tqdm(self.manager.train_dataloader):
                loss_dic = self.train_step(batch)
                for k, v in loss_dic.items():
                    if k not in train_loss_accum:
                        train_loss_accum[k] = 0.0
                    train_loss_accum[k] += v.item()
            
            train_log_dict = {
                f"train_{k}": (v / len(self.manager.train_dataloader)) 
                for k, v in train_loss_accum.items()
            }
            
            wandb.log(train_log_dict, step=epoch)
            for k, v in train_log_dict.items():
                log.info(f"{k}: {v:.6f}")
            
            self.save_model()
            
            if self.test_during_training_bool and epoch % 5 == 0 and epoch != 0:
                self.test(20, epoch)
                self.save_model(epoch)
            
            wandb.log({"learn_rate": self.method.lr_scheduler.get_last_lr()[0]}, step=epoch)
            
            self.method.lr_scheduler.step()
    
    def val_step(self, batch):
        self.method.eval()
        state, action, goal = self.method.preprocess_batch(batch)
        
        with torch.no_grad():
            loss_dic = self.method.compute_validation_loss(state, action['action'], goal)
        
        return loss_dic
    
    def train_step(self, batch):
        self.method.train()
        state, action, goal = self.method.preprocess_batch(batch)
        
        loss_dic = self.method.compute_training_loss(state, action['action'], goal)
        loss = loss_dic['total_loss']
        
        self.method.optimizer.zero_grad()
        loss.backward()
        self.method.optimizer.step()
        
        if self.use_ema:
            self.ema.update()
        
        return loss_dic
    
    def test(self, eval_n_times=None, epoch=None):
        self.method.eval()
        
        if self.use_ema:
            self.ema.apply_shadow()
        
        during_training = False
        
        if eval_n_times is not None:
            eval_n_times = eval_n_times
            during_training = True
        else:
            eval_n_times = self.eval_n_times
        
        test_results = self.manager.test_method(self.method, self.store_videos, eval_n_times, self.working_dir, during_training, epoch)
        
        if self.use_ema:
            self.ema.restore()
        
        if epoch is not None:
            wandb.log({f"test_during_training_{k}": v for k, v in test_results.items()}, step=epoch)
        else:
            wandb.log(test_results)
    
    def instantiate_models(self):
        if self.manager.use_proprioceptive:
            self.method.instantiate_proprioceptive()
        if self.manager.use_image:
            self.method.instantiate_vision_encoder()
        if self.manager.use_graph:
            self.method.instantiate_graph_encoder()
            
        self.method.instantiate_optimizer()
        
        if self.use_ema:
            ema_modules = []
            
            # Check standard attributes (adjust names to match your Method class)
            if hasattr(self.method, 'action_generator'): # The MDT model
                ema_modules.append(self.method.action_generator)
            if hasattr(self.method, 'vision_encoder') and self.method.vision_encoder:
                ema_modules.append(self.method.vision_encoder)
            if hasattr(self.method, 'graph_encoder') and self.method.graph_encoder:
                ema_modules.append(self.method.graph_encoder)
            
            # Initialize EMA if we found modules
            if len(ema_modules) > 0:
                log.info(f"Initializing EMA for {len(ema_modules)} modules.")
                self.ema = EMA(ema_modules, decay=0.999)
        
    def set_runtime_parameter(self, method):
        method.action_generator.proprio_dim = self.manager.prop_dim
        method.action_generator.graph_mod = self.manager.adapted_graph_modalities
        method.graph_encoder.input_dim = self.manager.graph_dim
        method.graph_encoder.edge_dim = self.manager.graph_edge_dim
        method.graph_encoder.modalities = self.manager.adapted_graph_modalities
        
        return method
    
    def save_model(self, epoch=None):
        if epoch is None:
            name = '_final'
        else:
            name = '_epoch_' + str(epoch)
        self.method.save_models(self.working_dir, name)
        if self.use_ema:
            torch.save(self.ema.state_dict(), os.path.join(self.working_dir, "ema_weights" + name + ".pth"))
    
    def load_model(self, epoch=None):
        if epoch is None:
            name = '_final'
        else:
            name = '_epoch_' + str(epoch)
        self.method.load_models(self.model_path, name)
        if self.use_ema:
            self.ema.load_state_dict(torch.load(os.path.join(self.model_path, "ema_weights" + name + ".pth")))