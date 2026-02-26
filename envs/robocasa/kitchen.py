import os
import cv2
import imageio
import numpy as np
import torch
from torchvision import transforms

from torchvision.ops import masks_to_boxes
from dataloader.utils import combine_graph_modalities, fuse_graphs
from envs.robocasa.utils import create_kitchen_env

from tqdm import tqdm
import networkx as nx

import logging

from networks.vision_encoder.cnn import SimpleImageEncoder
from networks.vision_encoder.utils import crop_and_resize_to_64, crop_and_resize_to_64_for_fusion
from utils.create_graphs import create_graph_datapoint
from utils.generate_graph_dataset_robocasa import OBJECT_NAMES_IMAGES, get_bb_pos

log = logging.getLogger(__name__)

class RoboCasaKitchenTester():
    def __init__(self,
                 dataset_path: str,
                 task_list: list[str],
                 use_depth: bool,
                 cropped_image_feature_encoder: torch.nn.Module = None,
                 ):
        
        self.env_list, self.task_list = create_kitchen_env(dataset_path, task_list, use_depth, 42)
        
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
            ])
        
        self.transform_cropped_feature = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((128, 128)),
            ])
        
        self.cropped_image_feature_encoder = cropped_image_feature_encoder

    def test(self, method, manager, store_videos: bool, eval_n_times: int, working_dir: str, test_count: int, epoch: int):
        result_dict = {}
        
        for env, task_name in zip(self.env_list, self.task_list):
            total_reward = 0
            log.info("Evaluation of: " + task_name)
            
            for i in range(eval_n_times):
                
                frames = []

                obs_all = env.reset()
                
                self.cls_list = {cls: i for i, cls in enumerate(list(env.model.classes_to_ids.keys()))}
                self.cls_numbers = []
                for class_name in list(self.cls_list.keys()):
                    self.cls_numbers.append(self.cls_list[class_name] + 1)
                self.cls_numbers = torch.tensor(self.cls_numbers)
                
                obs = {}
                
                left_graph = {}
                right_graph = {}
                for mod in manager.graph_modalities:
                    left_graph[mod] = nx.DiGraph()
                    right_graph[mod] = nx.DiGraph()

                if manager.use_proprioceptive:
                    prop_obs = self.create_prop_obs(obs_all)
                    obs['obs_prop'] = prop_obs
                if manager.use_image:
                    img_obs = self.create_image_obs(obs_all, manager)
                    obs['obs_img'] = img_obs
                if manager.use_graph:
                    graph_obs = self.create_graph_obs(obs_all, left_graph, right_graph, manager)
                    obs['obs_graph'] = graph_obs
                             
                lang_goal = env.get_ep_meta()['lang']
                print("Goal: ", lang_goal)
                
                success = 0
                step_counter = 0
                pbar = tqdm(total=env.horizon)
                
                while step_counter < env.horizon:
                    batch = {'observation': obs,
                             'goal': {'lang': lang_goal}
                             }

                    state, _, goal = method.preprocess_batch(batch)
                    pred_action = method.predict(state, goal)
                    for a in range(pred_action.shape[1]):
                        if store_videos:
                            img = obs_all['robot0_agentview_left_image']
                            frame = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
                            frames.append(frame)
                        action = method.clip_action(pred_action[0][a], manager.y_bounds).reshape(-1).detach().cpu().numpy()
                        obs_all, _, _, _ = env.step(action)
                        step_counter += 1
                        pbar.update(1)

                    obs = {}
                    if manager.use_proprioceptive:
                        prop_obs = self.create_prop_obs(obs_all)
                        obs['obs_prop'] = prop_obs
                    if manager.use_image:
                        img_obs = self.create_image_obs(obs_all, manager)
                        obs['obs_img'] = img_obs
                    if manager.use_graph:
                        graph_obs = self.create_graph_obs(obs_all, left_graph, right_graph, manager)
                        obs['obs_graph'] = graph_obs
                    
                    if env._check_success():
                        log.info(f"Success in {i}")
                        success = 1
                        total_reward += 1
                        break

                if store_videos:
                    add_str = ''
                    if epoch is not None:
                        add_str = f"epoch_{epoch}_"
                    video_filename = add_str + f"env_{task_name}_{test_count}_rollout_{i}_success_{success}.mp4"
                    video_filepath = os.path.join(working_dir, video_filename)
                    # Save the frames as a video using imageio
                    imageio.mimsave(video_filepath, frames, fps=30)

            log.info(f"Total reward in {task_name}: {total_reward}")
            result_dict["Avg_Test_Reward_" + task_name] = total_reward / eval_n_times

            env.close()
        return result_dict
    
    def create_prop_obs(self, obs_all):
        pass # TODO not yet needed (could be interesting in future)
    
    def create_image_obs(self, obs_all, manager):
        obs = {}
        for mod in manager.image_modalities:
            if mod == 'left':
                img = obs_all['robot0_agentview_left_image']
            elif mod == 'right':
                img = obs_all['robot0_agentview_right_image']
            elif mod == 'inhand':
                img = obs_all['robot0_eye_in_hand_image']
            else:
                raise ValueError(f'Unknown image modality: {mod}')
            
            img = self.transform(img).unsqueeze(0).unsqueeze(0)  # add batch and time dimensions
            
            obs[mod] = img
        return obs
    
    def create_graph_obs(self, obs_all, left_graph, right_graph, manager):
        left_data_point = {}
        right_data_point = {}
        
        for mod in manager.graph_modalities:
            left_object_names, right_object_names, left_objects, right_objects = self.get_data_from_img(obs_all, mod, self.cls_numbers, self.cls_list)
            left_data_point[mod] = create_graph_datapoint(left_graph[mod], left_object_names, left_objects)
            right_data_point[mod] = create_graph_datapoint(right_graph[mod], right_object_names, right_objects)

        local_graph_mods = manager.graph_modalities
        
        if not manager.use_splitted_modalities and len(manager.graph_modalities) > 1:
            merged_graph_modalities = "_".join(manager.graph_modalities)
            graph_data_left = {merged_graph_modalities: combine_graph_modalities(left_data_point, manager.graph_modalities)}
            graph_data_right = {merged_graph_modalities: combine_graph_modalities(right_data_point, manager.graph_modalities)}
            local_graph_mods = [merged_graph_modalities]
        else:
            graph_data_left = left_data_point
            graph_data_right = right_data_point

        if manager.use_graph_fusion:
            graph_obs = {}
            for mod in local_graph_mods:
                fused_graph = fuse_graphs(graph_data_left, graph_data_right, mod, is_cropped_fusion=manager.is_cropped_fusion)
                graph_obs[mod] = fused_graph
        else:
            graph_obs = {}
            for mod in local_graph_mods:
                graph_obs[mod + '_left'] = graph_data_left[mod]
                graph_obs[mod + '_right'] = graph_data_right[mod]
        return graph_obs
                
    def get_data_from_img(self, obs, mod, cls_numbers: list, cls_list: list):
        
        left_object_names, left_boxes, left_morph_masks, left_img = self.helper_function(obs, 'agentview_left', cls_numbers, cls_list)
        right_object_names, right_boxes, right_morph_masks, right_img = self.helper_function(obs, 'agentview_right', cls_numbers, cls_list)
        
        if mod == "one_hot_labels":
            objects = []
            for name in left_object_names:
                one_hot = torch.zeros(len(OBJECT_NAMES_IMAGES))
                index = OBJECT_NAMES_IMAGES.index(name)
                one_hot[index] = 1.0
                objects.append(one_hot)
            left_objects = torch.stack(objects)
        elif mod == "bb_coordinates":
            objects = []
            for i in range(left_boxes.shape[0]):
                objects.append(get_bb_pos(left_boxes[i]) / 127)
            left_objects = torch.stack(objects)
        elif mod == "cropped_image_feature":
            object_img = []
            for i in range(left_morph_masks.shape[0]):
                if isinstance(self.cropped_image_feature_encoder, SimpleImageEncoder):
                    if left_object_names[i] in right_object_names:
                        r_obj_idx = right_object_names.index(left_object_names[i])
                        obj_r = right_morph_masks[r_obj_idx]
                    else:
                        obj_r = None
                    object_img.append(crop_and_resize_to_64_for_fusion(left_img, right_img, left_morph_masks[i], obj_r))
                else:
                    object_img.append(crop_and_resize_to_64(left_img, left_morph_masks[i]))
            left_objects = self.cropped_image_feature_encoder(torch.stack(object_img)).squeeze(0)
            
        if mod == "one_hot_labels":
            objects = []
            for name in right_object_names:
                one_hot = torch.zeros(len(OBJECT_NAMES_IMAGES))
                index = OBJECT_NAMES_IMAGES.index(name)
                one_hot[index] = 1.0
                objects.append(one_hot)
            right_objects = torch.stack(objects)
        elif mod == "bb_coordinates":
            objects = []
            for i in range(right_boxes.shape[0]):
                objects.append(get_bb_pos(right_boxes[i]) / 127)
            right_objects = torch.stack(objects)
        elif mod == "cropped_image_feature":
            object_img = []
            for i in range(right_morph_masks.shape[0]):
                if isinstance(self.cropped_image_feature_encoder, SimpleImageEncoder):
                    if right_object_names[i] in left_object_names:
                        l_obj_idx = left_object_names.index(right_object_names[i])
                        obj_l = left_morph_masks[l_obj_idx]
                    else:
                        obj_l = None
                    object_img.append(crop_and_resize_to_64_for_fusion(left_img, right_img, obj_l, right_morph_masks[i]))
                else:
                    object_img.append(crop_and_resize_to_64(right_img, right_morph_masks[i]))
            right_objects = self.cropped_image_feature_encoder(torch.stack(object_img)).squeeze(0)
        
        return left_object_names, right_object_names, left_objects, right_objects
    
    def helper_function(self, obs, view, cls_numbers, cls_list):
        img = self.transform(obs['robot0_' + view + '_image'])
        mask = torch.tensor(obs['robot0_' + view + '_segmentation_class'][:, :, 0])
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        img = img*std + mean

        obj_ids_unique = torch.unique(mask)
        obj_ids = []
        for id in cls_numbers:
            if id in obj_ids_unique:
                obj_ids.append(id)
        obj_ids = torch.tensor(obj_ids)

        masks = mask == obj_ids[:, None, None]

        new_obj_ids = []
        morph_masks = []

        for r, mask in enumerate(masks):
            new_mask = cv2.morphologyEx(mask.numpy().astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2)))
            if np.sum(new_mask) == 0:
                pass
            else:
                morph_masks.append(new_mask)
                new_obj_ids.append(obj_ids[r].item())

        if new_obj_ids == []:
            return [], torch.tensor([])
        
        morph_masks = torch.tensor(np.array(morph_masks))

        boxes = masks_to_boxes(morph_masks)

        object_names = []

        for i, id in enumerate(new_obj_ids):
            obj_name = list(cls_list.keys())[id - 1]
            object_names.append(obj_name)
            
        return object_names, boxes, morph_masks, img