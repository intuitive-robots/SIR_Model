import abc

import torch

class Base_Manager(abc.ABC):
    def __init__(
        self,
        data_path: str,
        load_dataset: bool,
        batch_size: int,
        train_per: float,
        num_workers: int,
        obs_window: int,
        action_window: int,
        action_dim: int,
        task_names: list,
        prop_modalities: list,
        image_modalities: list,
        graph_modalities: list,
        pretrained_img_encoder_name: str,
    ):
        self.data_path = data_path
        self.load_dataset = load_dataset
        self.batch_size = batch_size
        self.train_per = train_per
        self.num_workers = num_workers
        self.obs_window = obs_window
        self.action_window = action_window
        self.action_dim = action_dim
        
        self.task_names = task_names
        self.use_proprioceptive = True if len(prop_modalities) > 0 else False
        self.prop_modalities = prop_modalities
        self.use_image = True if len(image_modalities) > 0 else False
        self.image_modalities = image_modalities
        self.use_graph = True if len(graph_modalities) > 0 else False
        self.graph_modalities = graph_modalities
        self.pretrained_img_encoder_name = pretrained_img_encoder_name
            
    @abc.abstractmethod      
    def init_data(self):
        """
        Method to initate data and models used for training or preprocessing data
        """
        pass
    
    @abc.abstractmethod
    def test_method(self, method):
        """
        Method to test the given method on the test dataset
        """
        pass
    
    @abc.abstractmethod
    def calculate_proprioceptive_dim(self, modalities):
        """
        Method to calculate the proprioceptive dimension based on the given modalities
        """
        pass
    
    @abc.abstractmethod
    def calculate_graph_dim(self, modalities):
        """
        Method to calculate the graph dimension based on the given modalities
        """
        pass