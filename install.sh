#!/bin/bash

############ GENERAL ENV SETUP ############
echo New Environment Name:
read envname

echo Creating new conda environment $envname 
conda create -n $envname python=3.10 -y -q

eval "$(conda shell.bash hook)"
conda activate $envname

echo
echo Activating $envname
if [[ "$CONDA_DEFAULT_ENV" != "$envname" ]]
then
    echo Failed to activate conda environment.
    exit 1
fi

conda activate $envname

pip install torch torchvision
pip install wandb
pip install einops
pip install matplotlib
pip install hydra-core
pip install torch_geometric
pip install h5py
pip install imageio-ffmpeg
pip install imageio
pip install opencv-python
pip install einops_exts
pip install git+https://github.com/openai/CLIP.git

echo Done installing all necessary packages. Please follow the next steps mentioned on the readme

echo
echo
echo Successfully installed.
echo
echo To activate your environment call:
echo conda activate $envname
exit 0