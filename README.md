# OpenVLA-OFT_withUR5

## Introduction

This repository contains code and resources for training and deploying an OpenVLA-OFT model on a UR5 robotic arm. 

---

## File Overview

| File / Folder             | Description |
|---------------------------|-------------|
| `ur5_haochen.py`         | Script to collect demonstrations using a joystick. Saves UR5 TCP poses, associated actions, images(3rd view and wrist view) and timestamps |
| `raw_to_preprocessed.py`          | Filters and converts pose data into delta XYZ+RPY format(as action), associates it with synchronized third-person and wrist images, and saves structured episodes with actions and language instructions into .npy files  |
| `rlds_dataset_builder/UR5`         | The folder for UR5 dataset builder, refer to [here](https://github.com/moojink/rlds_dataset_builder/tree/main) to build the dataset |
| `openvla-oft/vla-scripts/finetune.py`         | The file to do finetuning |
| `openvla-oft/vla-scripts/deploy.py`         | The file to run on the server side when deployment |
| `ur5_deploy.py`         | The file to run on the client side when deployment |


---

## Usage

1. **Install dependencies**
   ```bash
   conda env create -f environment.yml

2. **Above is my environment. If some module is not working or needs to be built, I suggest referring to the original repos of [OpenVLA-OFT](https://github.com/moojink/openvla-oft)**
