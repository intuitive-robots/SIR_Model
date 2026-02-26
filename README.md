# SIR: Structured Image Representations for Explainable Robot Learning
[Paper](https://paulmattes.github.io/publication/conference-paper/), [Project Page](https://intuitive-robots.github.io/SIR_website/), [CVPR 2026]()

[Paul Mattes](https://paulmattes.github.io/)<sup>1</sup>,
Jan Schwab,
Jens Bosch,
Maximilian Li,
Nils Blank,
Minh-Trung Tang,
Moritz Haberland and
[Rudolf Lioutikov](http://rudolf.intuitive-robots.net/)<sup>1</sup>

<sup>1</sup>Intuitive Robots Lab, Karlsruhe Institute of Technology

This is the official code repository for the paper [SIR: Structured Image Representations for Explainable Robot Learning](https://paulmattes.github.io/publication/conference-paper/).

## Installation

1. Start installation using the install.sh
```
cd sir
sh install.sh
```

Following instructions are taken from here: https://github.com/robocasa/robocasa

All installations should NOT be done in the SIR folder. RoboCasa and SIR, should be in one folder.

2. Copy robosuite repo and install it using

```
cd ..
git clone -b robocasa_v0.1 https://github.com/ARISE-Initiative/robosuite
cd robosuite
pip install -e .
python robosuite/scripts/setup_macros.py
```

For Windows user: https://robosuite.ai/docs/installation.html

3. Copy robocasa repo and install it. Afterwards download kitchen assets and setup macro
```
cd ..
git clone https://github.com/robocasa/robocasa
cd robocasa
git reset --hard 370f986aa3934be6c134ecb978952423df9a1ed0
pip install -e .
python robocasa/scripts/download_kitchen_assets.py
python robocasa/scripts/setup_macros.py
```

# File Changes

Following files need to be changed in the robosuite and robocasa repos

### Robosuite

Change in robosuite/macros_private.py IMAGE_CONVENTION from opgengl to opencv

Also add the following code in `robosuite/robosuite/models/tasks/task.py` and define `self.count = 0` in the init method. 

Add after line 112: 

```
if cls == "MJCFObject":
    cls = model.name
self.count += 1
if self.count > 3:
    for geom in model.contact_geoms:
        if geom not in sim.model.geom_names:
            print("removed: ", geom)
            geom_name = geom.split("_")[-1]
            model._contact_geoms.remove(geom_name)
    for geom in model.visual_geoms:
        if geom not in sim.model.geom_names:
            print("removed: ", geom)
            geom_name = geom.split("_")[-1]
            model._visual_geoms.remove(geom_name)
```

### RoboCasa

#### robocasa/environments/kitchen/kitchen.py

All line-numbers refer to the original code, without the changes made previously in the files, respectively. 

Add below line 230 (right at the start of the init function)

```
np.random.seed(seed)
random.seed(seed)
```

Add to the end of the super().init() function after line 313:

```
camera_segmentations="class",
```

Additionally, you need to add the following code below every `self.model.merge_objects([model])` starting from line 478 (3 times: after 507, 483, 474):

```
self.model.mujoco_objects.append(model)
```

# Datasets (TBD)

The graph datasets for RoboCasa will be uploaded to HuggingFace in the upcoming days.
https://huggingface.co/datasets/MrLayen/SIR_robocasa