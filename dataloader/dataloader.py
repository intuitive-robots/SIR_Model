import h5py
import numpy as np
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from dataloader.datasets import RoboCasaDataset
from envs.robocasa.utils import TASK_LIST

def dataset_split_robocasa(
    data_directory: str,
    task_names: list,
    split: float = 0.95,
    obs_window: int = 1,
    act_window: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
    prop_mod: list = None,
    img_mod: list = None,
    graph_mod: list = None,
    use_graph_fusion: bool = False,
    use_splitted_modalities: bool = False,
    cropped_img_dim: int = 256,
    cropped_model_name: str = None,
):
    train_dataset_list = []
    val_dataset_list = []
    
    if task_names[0] == "ALL":
        task_names = TASK_LIST
    
    for task_name in task_names:
        data = h5py.File(data_directory + task_name + "/demo_gentex_im128_randcams.hdf5", "r")

        # sorting demonstrations by number
        raw_list = np.array(data['data'])
        permutation = np.argsort(np.array([int(i.split('_', 1)[-1]) for i in raw_list]))
        sorted_keys = raw_list[permutation]
        
        train_split = int(len(raw_list) * split)
        
        offset = int(sorted_keys[0].split('_')[-1])
        
        train_dataset_list.append(RoboCasaDataset(data,
                                                sorted_keys[:train_split],
                                                offset,
                                                data_directory + task_name,
                                                obs_window, 
                                                act_window,
                                                prop_mod, 
                                                img_mod, 
                                                graph_mod, 
                                                use_graph_fusion,
                                                use_splitted_modalities,
                                                cropped_img_dim,
                                                cropped_model_name))
        val_dataset_list.append(RoboCasaDataset(data,
                                        sorted_keys[train_split:],
                                        offset,
                                        data_directory + task_name,
                                        obs_window, 
                                        act_window,
                                        prop_mod, 
                                        img_mod, 
                                        graph_mod, 
                                        use_graph_fusion,
                                        use_splitted_modalities,
                                        cropped_img_dim,
                                        cropped_model_name))
    
    # ConcatDataset creates a single dataset from the list of datasets
    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)
    
    print("Train dataset size: ", len(train_dataset))
    print("Validation dataset size: ", len(val_dataset))
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, val_dataloader