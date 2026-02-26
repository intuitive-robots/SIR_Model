import json
import clip
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import einops
import logging

from dataloader.utils import combine_graph_modalities, fuse_graphs, pad_sequence, convert_weight_to_dim_distance

log = logging.getLogger(__name__)

class RoboCasaDataset(Dataset):
    def __init__(self,
                 data: any,
                 keys: list,
                 offset: int,
                 data_directory: str,
                 obs_window: int,
                 action_window: int,
                 prop_mod: list,
                 img_mod: list,
                 graph_mod: list,
                 use_graph_fusion: bool,
                 use_splitted_modalities: bool,
                 cropped_img_dim: int,
                 model_name: str = None
    ):
        self.data_directory = data_directory
        
        self.use_prop = True if len(prop_mod) > 0 else False
        self.use_img = True if len(img_mod) > 0 else False
        self.use_graph = True if len(graph_mod) > 0 else False
        
        self.use_graph_fusion = use_graph_fusion
        
        self.obs_window = obs_window
        self.action_window = action_window
        self.prop_mod = prop_mod
        self.img_mod = img_mod
        self.graph_mod = graph_mod
        
        self.prop_obs = []
        self.img_obs = []
        if use_graph_fusion:
            self.graph_obs = []
        else:
            self.graph_obs_left = []
            self.graph_obs_right = []
        
        self.actions = []
        self.lang_goals = []
        
        self.dataset_length = 0
        self.demo_lengths = []
        
        language_encoder, _ = clip.load("ViT-B/32", device='cpu')
        
        self.is_cropped_fusion = False

        if self.use_graph:
            self.graph_data_left = {}
            self.graph_data_right = {}
            for mod in self.graph_mod:
                if mod == "cropped_image_feature":
                    crop_str = "_" + str(cropped_img_dim)
                    if "fusion" in model_name:
                        crop_str = "_fusion" + crop_str
                        self.is_cropped_fusion = True
                else:
                    crop_str = ""
                left_graph_path = self.data_directory + "/" + mod + crop_str + "_left_image.pth"
                right_graph_path = self.data_directory + "/" + mod + crop_str + "_right_image.pth"
                log.info(f"Loading graph modality '{mod}' from:\n Left: {left_graph_path}\n Right: {right_graph_path}")
                self.graph_data_left[mod] = (torch.load(left_graph_path, weights_only=False))
                self.graph_data_right[mod] = (torch.load(right_graph_path, weights_only=False))
            if not use_splitted_modalities and len(self.graph_mod) > 1:
                self.merged_mod_name = "_".join(self.graph_mod)
        
        for key in tqdm(keys):
            
            dl = len(data['data'][key]['actions'][()])
            self.dataset_length += dl
            self.demo_lengths.append(dl)
            
            idx = int(key.split('_')[-1]) - offset
            
            if self.use_prop:
                single_prop_obs = self.get_proprioceptive_obs(data['data'][key]['obs'], prop_mod)
                self.prop_obs.append(single_prop_obs)
            
            if self.use_img:
                single_img_obs = self.get_image_obs(img_mod, idx)
                self.img_obs.append(single_img_obs)
                
            if self.use_graph:
                if use_graph_fusion:
                    self.graph_obs.append(self.get_graph_obs(idx, use_splitted_modalities))
                else:
                    left, right = self.get_graph_obs(idx, use_splitted_modalities)
                    self.graph_obs_left.append(left)
                    self.graph_obs_right.append(right)
                    
            action = torch.tensor(data['data'][key]['actions'][()])
            self.actions.append(action)
                        
            self.lang_goals.append(json.loads(data['data'][key].attrs.get("ep_meta", None))['lang'])

        with torch.no_grad():
            lang_goal_tokenized = clip.tokenize(self.lang_goals)
            self.lang_goals = language_encoder.encode_text(lang_goal_tokenized).to(torch.float32)
            
        if self.use_graph and not use_splitted_modalities and len(self.graph_mod) > 1:
            # Update the class attribute so __getitem__ knows the new key
            self.graph_mod = [self.merged_mod_name]

    def get_proprioceptive_obs(demo_data, mod):
        modalities = []
        for m in mod:
            modalities.append(demo_data[m][()])
            if m == 'object':
                temp_objects = torch.tensor(demo_data[m][()])
                objects = torch.zeros((temp_objects.shape[0], 56))
                objects[:,:temp_objects.shape[1]] = temp_objects
                modalities.append(objects)
        
        observation = torch.concatenate((modalities), dim=-1).type(torch.float32)
        return observation
    
    def get_image_obs(self, mod, idx):
        img_tensor = torch.load(self.data_directory + "/img_tensor_demo_" + str(idx) + ".pth")
        img_tensor = einops.rearrange(img_tensor, "step (num c h w) -> step num c h w", num=3, c=3, h=128, w=128)
        
        img_obs = {}
        
        for m in mod:
            if m == 'inhand':
                img_obs['inhand'] = img_tensor[:,0]
            elif m == 'left':
                img_obs['left'] = img_tensor[:,1]
            elif m == 'right':
                img_obs['right'] = img_tensor[:,2]
        
        return img_obs
    
    def get_graph_obs(self, idx, use_splitted_modalities):
        local_graph_mod = self.graph_mod
        
        # if "bb_coordinates" in self.graph_mod:
        #     self.graph_data_left = convert_weight_to_dim_distance(self.graph_data_left, self.graph_mod)
        #     self.graph_data_right = convert_weight_to_dim_distance(self.graph_data_right, self.graph_mod)
        
        # Handle merging modalities if required
        if not use_splitted_modalities and len(self.graph_mod) > 1:
            new_data_left = []
            new_data_right = []
                
            num_steps = len(self.graph_data_left[self.graph_mod[0]][idx])
            
            for j in range(num_steps):
                new_data_left.append(combine_graph_modalities(self.graph_data_left, self.graph_mod, idx, j))
                new_data_right.append(combine_graph_modalities(self.graph_data_right, self.graph_mod, idx, j))
            
            graph_data_left = {self.merged_mod_name: new_data_left}
            graph_data_right = {self.merged_mod_name: new_data_right}
            
            local_graph_mod = [self.merged_mod_name]
        else:
            graph_data_left = {}
            graph_data_right = {}
            for mod in self.graph_mod:
                graph_data_left[mod] = self.graph_data_left[mod][idx]
                graph_data_right[mod] = self.graph_data_right[mod][idx]
                        
        # Handle merging graphs if required
        if self.use_graph_fusion:
            fused_graph_data = {}
            for mod in local_graph_mod:
                fused_graph_data[mod] = []
                
                for step_idx in range(len(graph_data_left[mod])):
                    fused_graph_data[mod].append(fuse_graphs(graph_data_left, graph_data_right, mod, step_idx, self.is_cropped_fusion))                    
            return fused_graph_data
        else:
            return graph_data_left, graph_data_right
    
    def get_indices(self, idx, window):
        """
        Resolves a global dataset index into a specific demonstration index
        and the start/end indices within that demonstration.
        Ensures indices do not exceed the length of the specific demonstration.
        """
        demo_index = 0
        inside_start_index = 0
        
        # 1. Find which demo this index belongs to
        for i in range(len(self.demo_lengths)):
            temp = idx - self.demo_lengths[i]
            if temp < 0:
                demo_index = i
                inside_start_index = idx
                break
            else:
                idx = temp
        
        # 2. Calculate the end index based on the window size
        # We clamp the end index to the max length of the current demo
        demo_len = self.demo_lengths[demo_index]
        inside_end_index = min(inside_start_index + window, demo_len)
        
        return demo_index, inside_start_index, inside_end_index
    
    def __getitem__(self, idx):
        item = {
            'observation': {},
            'action': {},
            'goal': {}
        }
        
        # --- OBSERVATIONS ---
        # Get indices for Observation Window
        demo_idx, obs_start, obs_end = self.get_indices(idx, self.obs_window)
        
        if self.use_prop:
            # Slicing
            raw_prop = self.prop_obs[demo_idx][obs_start:obs_end]
            # Padding if out of bounds (handled by helper)
            item['observation']['obs_prop'] = pad_sequence(raw_prop, self.obs_window)
            
        if self.use_img:
            item['observation']['obs_img'] = {}
            for m in self.img_mod:
                raw_img = self.img_obs[demo_idx][m][obs_start:obs_end]
                item['observation']['obs_img'][m] = pad_sequence(raw_img, self.obs_window)
                
        if self.use_graph:
            item['observation']['obs_graph'] = {}
            for m in self.graph_mod:
                if not self.use_graph_fusion:
                    raw_graph_left = self.graph_obs_left[demo_idx][m][obs_start:obs_end]
                    raw_graph_right = self.graph_obs_right[demo_idx][m][obs_start:obs_end]
                    # TODO padding
                    item['observation']['obs_graph'][m + '_left'] = raw_graph_left[0]
                    item['observation']['obs_graph'][m + '_right'] = raw_graph_right[0]
                else:
                    raw_graph = self.graph_obs[demo_idx][m][obs_start:obs_end]
                    # Graph padding usually implies repeating the last graph object TODO
                    if len(raw_graph) < self.obs_window:
                        last = raw_graph[-1]
                        raw_graph.extend([last] * (self.obs_window - len(raw_graph)))
                    item['observation']['obs_graph'][m] = raw_graph[0]

        # --- ACTIONS ---
        # Get indices for Action Window
        _, act_start, act_end = self.get_indices(idx, self.action_window)
        
        raw_action = self.actions[demo_idx][act_start:act_end]
        item['action']['eef'] = pad_sequence(raw_action, self.action_window)

        # --- GOALS ---
        # Language goal is static per demonstration
        item['goal']['lang'] = self.lang_goals[demo_idx]
        
        return item
            
    def __len__(self):
        return self.dataset_length