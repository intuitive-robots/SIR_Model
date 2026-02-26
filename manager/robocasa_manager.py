import os

import torch
from dataloader.dataloader import dataset_split_robocasa
from envs.robocasa.kitchen import RoboCasaKitchenTester
from envs.robocasa.utils import TASK_LIST
from manager.base_manager import Base_Manager
from networks.vision_encoder.utils import load_pretrained_image_encoder
from utils.generate_graph_dataset_robocasa import OBJECT_NAMES_IMAGES, create_graphs_and_save

import logging

log = logging.getLogger(__name__)

class RoboCasa_Manager(Base_Manager):
    def __init__(
        self,
        data_path: str,
        load_dataset: bool,
        batch_size: int,
        train_per: float,
        num_workers: int,
        obs_window: int,
        act_window: int,
        action_dim: int,
        times_repeat: int,
        task_names: list = None,
        prop_modalities: list = None,
        image_modalities: list = None,
        graph_modalities: list = None,
        pretrained_img_encoder_name: str = None,
        use_graph_fusion: bool = False,
        use_splitted_modalities: bool = False,
    ):
        super().__init__(
            data_path,
            load_dataset,
            batch_size,
            train_per,
            num_workers,
            obs_window,
            act_window,
            action_dim,
            task_names,
            prop_modalities,
            image_modalities,
            graph_modalities,
            pretrained_img_encoder_name,
        )
        
        self.times_repeat = times_repeat
        
        self.use_graph_fusion = use_graph_fusion
        self.use_splitted_modalities = use_splitted_modalities
        self.adapted_graph_modalities = graph_modalities
        
        self.is_cropped_fusion = False
        
        self.cropped_img_dim = 0
        self.cropped_image_feature_encoder = None
        
        self.prop_dim = self.calculate_proprioceptive_dim(prop_modalities)
        self.graph_dim, self.graph_edge_dim = self.calculate_graph_dim() if self.use_graph else (0, 0)

        self.init_data()
        
    def init_data(self):
        if self.use_graph:
            self.check_and_create_graph_dataset()
        
        if self.load_dataset:
            self.train_dataloader, self.valid_dataloader = dataset_split_robocasa(
                data_directory=self.data_path,
                task_names=self.task_names,
                split=self.train_per,
                obs_window=self.obs_window,
                act_window=self.action_window,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                prop_mod=self.prop_modalities,
                img_mod=self.image_modalities,
                graph_mod=self.graph_modalities,
                use_graph_fusion=self.use_graph_fusion,
                use_splitted_modalities=self.use_splitted_modalities,
                cropped_img_dim=self.cropped_img_dim,
                cropped_model_name=self.pretrained_img_encoder_name,
            )

        self.y_bounds = torch.tensor(
                    [[-1.0000, -1.0000, -1.0000, -1.0000, -0.9629, -0.9857, -1.0000,  0.0000,
                    -0.0000, -0.0286, -0.2071, -1.0000],
                    [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  0.8971,  1.0000,  0.1214,
                    0.6929,  0.0000,  0.0000, -1.0000]])
    
    def check_and_create_graph_dataset(self):
        # Check if the requested dataset is there and if not create it
        datasets_to_create = []

        for mod in self.graph_modalities:
            if "cropped_image_feature" == mod:
                crop_str = "_" + str(self.cropped_img_dim)
                if self.is_cropped_fusion:
                    crop_str = "_fusion" + crop_str
            else:
                crop_str = ""
            
            if self.task_names[0] == "ALL":
                task_names = TASK_LIST
            else:
                task_names = self.task_names
            for task_name in task_names:
                right_graph = not os.path.isfile(os.path.join(self.data_path, task_name, mod + crop_str + "_right_image.pth"))
                left_graph = not os.path.isfile(os.path.join(self.data_path, task_name, mod + crop_str + "_left_image.pth"))
                if right_graph or left_graph:
                    datasets_to_create.append((task_name, mod))
        
        if "cropped_image_feature" in self.graph_modalities:
            self.cropped_image_feature_encoder = load_pretrained_image_encoder(self.pretrained_img_encoder_name)
        
        if datasets_to_create == []:
            log.info("Requested datasets are available")
        else:
            for task, mod in datasets_to_create:
                log.info(f"Creating dataset for task: {task} and modality: {mod}")
                create_graphs_and_save(
                    dataset_path=self.data_path,
                    task_name=task,
                    graph_modality=mod,
                    encoder_model=self.cropped_image_feature_encoder,
                )
    
    def test_method(self, method, store_videos, eval_n_times, working_dir, during_training=False, epoch=None):
        times_repeat = self.times_repeat #if during_training else self.times_repeat
        results_all = {}
        
        task_names = ["CloseDrawer"] if during_training else self.task_names
        if task_names[0] == "ALL":
            task_names = TASK_LIST
        
        for i in range(times_repeat):
            tester = RoboCasaKitchenTester(
                dataset_path=self.data_path,
                task_list=task_names,
                use_depth=False,
                cropped_image_feature_encoder=self.cropped_image_feature_encoder,
            )
            
            results = tester.test(
                method=method,
                manager = self,
                store_videos=store_videos,
                eval_n_times=eval_n_times,
                working_dir=working_dir,
                test_count=i,
                epoch=epoch
            )
            
            # add results over all repeated times
            for key in results.keys():
                if key not in list(results_all.keys()):
                    results_all[key] = results[key]
                else:
                    results_all[key] += results[key]
                    
        avg_test_reward = 0 # average over all tasks

        # divide by the number of repeated times
        for key in results_all.keys():
            results_all[key] = results_all[key]/times_repeat
            avg_test_reward += results_all[key]
        
        num_tasks = len(task_names) if task_names else 1
        results_all['Avg_Test_Reward'] = avg_test_reward / num_tasks
        
        return results_all
    
    def calculate_proprioceptive_dim(self, modalities):
        dim = 0
        for mod in modalities:
            if mod == 'robot0_gripper_qpos':
                dim += 4
            elif mod == 'robot0_joint_pos':
                dim += 3
            elif mod == 'robot0_eef_pos':
                dim += 3
            elif mod == 'robot0_eef_quat':
                dim += 4
            elif mod == 'robot0_base_pos':
                dim += 3
            elif mod == 'robot0_base_quat':
                dim += 4
            elif mod == 'object':
                dim += 52
            else:
                raise ValueError(f'Unknown proprioceptive modality: {mod}')
        return dim
    
    def calculate_graph_dim(self):
        factor = 2
        if not self.use_splitted_modalities:
            self.adapted_graph_modalities = ["_".join(self.graph_modalities)]
        if not self.use_graph_fusion:
            factor = 1
            temp_graph_mods = self.adapted_graph_modalities
            self.adapted_graph_modalities = []
            for key in temp_graph_mods:
                self.adapted_graph_modalities.append(key + "_left")
                self.adapted_graph_modalities.append(key + "_right")
                
        dim = {}
        for mod in self.adapted_graph_modalities:
            dim[mod] = 0
        
        if "256" in self.pretrained_img_encoder_name:
            self.cropped_img_dim = 256
        elif "37" in self.pretrained_img_encoder_name:
            self.cropped_img_dim = 37
        else:
            raise ValueError(f"Unknown embed_dim for pretrained image encoder: {self.pretrained_img_encoder_name}")
        
        graph_edge_dim = 1
        for mod in self.adapted_graph_modalities:
            if "one_hot_labels" in mod:
                dim[mod] += len(OBJECT_NAMES_IMAGES)
            if "bb_coordinates" in mod:
                dim[mod] += 10 * factor
                # graph_edge_dim = 4
            if "cropped_image_feature" in mod:
                if "fusion" in self.pretrained_img_encoder_name:
                    dim[mod] += self.cropped_img_dim
                    self.is_cropped_fusion = True
                else:
                    dim[mod] += self.cropped_img_dim * factor
        return dim, graph_edge_dim