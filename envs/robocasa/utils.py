import os
import robocasa
import robosuite
import h5py
import json

TASK_LIST = [
    'CloseDrawer', 'OpenDrawer',
    'CloseDoubleDoor', 'CloseSingleDoor', 'OpenDoubleDoor', 'OpenSingleDoor',
    'TurnOffMicrowave', 'TurnOnMicrowave',
    'TurnOnSinkFaucet', 'TurnOffSinkFaucet', 'TurnSinkSpout',
    'TurnOnStove', 'TurnOffStove',
    'CoffeePressButton', 'CoffeeServeMug', 'CoffeeSetupMug',
    'PnPCabToCounter', 'PnPCounterToCab', 'PnPCounterToMicrowave', 'PnPCounterToSink',
    'PnPCounterToStove', 'PnPMicrowaveToCounter', 'PnPSinkToCounter', 'PnPStoveToCounter'
]

HORIZON_LIST = [
                500, 500,
                700, 500, 1000, 500,
                500, 500,
                500, 500, 500,
                500, 500,
                300, 600, 600,
                500, 500, 600, 700,
                500, 500, 500, 500
            ]

def create_kitchen_env(dataset_path: str, task_list: list[str], use_depth: bool, seed: int):
    if task_list[0] == "ALL":
        dataset_path = os.path.dirname(dataset_path)
        data_folder_name_list = TASK_LIST
        task_list = TASK_LIST
        print("Task list: ", task_list)
        env_meta_list = []
        for name in data_folder_name_list:
            env_meta_list.append(get_env_metadata_from_dataset(dataset_path=os.path.join(dataset_path, name, "demo_gentex_im128_randcams.hdf5")))
    else:
        env_meta_list = []
        for task in task_list:
            env_meta_list.append(get_env_metadata_from_dataset(dataset_path=os.path.join(dataset_path, task, "demo_gentex_im128_randcams.hdf5")))

    env_list = []
    for env_meta, env_n in zip(env_meta_list, task_list):
        config = env_meta['env_kwargs']

        config['camera_heights'] = 128
        config['camera_widths'] = 128
        config['horizon'] = HORIZON_LIST[TASK_LIST.index(env_n)]
        config['seed'] = seed
        config['camera_depths'] = use_depth

        print("Initializing environment...")

        env = robosuite.make(
            **config,
            env_name=env_n
        )

        env_list.append(env)

    return env_list, task_list

def get_env_metadata_from_dataset(dataset_path):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_meta = json.loads(f["data"].attrs["env_args"])
    f.close()
    return env_meta