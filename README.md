
# JYC Object Detection

Object Detection repository for our Junior Year Competition Project


## Roadmap

- Installing Dependencies
- Importing our Dataset
- Choosing our Model
- Start Training
- F1 & Precision Recall Curves
- Inferencing
- Configuration Parameters
- Webcame Setup


## Installing Dependencies
Run the following commands in your CMD terminal or VSCode terminal to install all the required dependencies. 
Remember these commands are for windows CMD and won't run on LINUX

#### Selecting a drive to store the YOLOv7 repository
This is a really good way to keep all your packages and dependencies in one place rather than them being scattered all over the place and in turn colliding with other pre-installed packages.

for e.g., If I am placing the repository in my **F: Drive** and further making a folder named **"YOLORepository"** I'll do this in my cmd terminal

```bash
F:
mkdir YOLORepository
cd YOLORepository
```

#### Clone the YOLOv7 Repository and install all the dependencies in the folder

```bash
git clone https://github.com/augmentedstartups/yolov7.git
cd yolov7
pip install -r requirements.txt
pip install roboflow
```


## Importing Roboflow Dataset
#### Startup VisualStudio Code & Open the VSCode Terminal
Start VisualStudio Code and open the YOLORepository folder by going into File > Open Folder > F > YOLORepository.
Start the VSCode Terminal (if not already open). Go into View > Terminal.

You will already be in the YOLORepository folder. Just type in the following code into the terminal.
```bash
cd yolov7
```
#### Roboflow Dataset import code
Run the following python script (make sure you have python installed in your laptop and that your VSCode python is set up). Don't give out the code below, it is unique to our own dataset.

or

Just simply run the ImportDatasets.py file in the AllPythonScripts folder from this repository. 

After running the script, you will have a new folder by the name of "xardari-yc-3". Move this folder into the "yolov7" folder
```bash
from roboflow import Roboflow
rf = Roboflow(api_key="HahXqnKfPHiO1vdUK7Lb")
project = rf.workspace("xardariii").project("xardarii-yc")
dataset = project.version(3).download("yolov7")
```
## Choosing Model

For this you have to have the GitBash dependency installed in your computer.
The best way to check is if you open your windows explorer or any folder and right click to check if you can see the following:

un-comment any of the following model you want to choose. I have already selected the best model so just run "bash", make sure you're in the yolov7 folder if not run the cd yolov7 command and then the first wget from the codes below.

Make sure to double check the path in the code below after -P. In my case, I have placed the yolov7 foler in the following path "F:/YOLORepository/yolov7".

```bash
bash
cd yolov7

wget -P F:/YOLORepository/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# wget -P F:/YOLORepository/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt

# wget -P F:/YOLORepository/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt

# wget -P F:/YOLORepository/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt

# wget -P F:/YOLORepository/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt

# wget -P F:/YOLORepository/yolov7 https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
```
## Training Custom Model

**THIS PART IS ALREADY PERFORMED BY ME IN COLLAB, NO NEED TO RUN IT AGAIN**

Full list of training arguments (https://github.com/WongKinYiu/yolov7/blob/main/train.py)

Some important arguments to know
- **configuration:** In the main yolov7 folder go to cfg/training folder and select the path of appropriate configuration file. Give the relative path to the file in --cfg argument
- **data:** the path to data folder, it will be automatically catered
- **weights:** path to pretrained weights given by --weights argument

Note for resuming training from checkpoint
By default, the checkpoints for the epoch are stored in folder, yolov7/runs/train, give the relative path to last epoch checkpoints

Run the "cd" command if not already in the yolov7 folder.
Change the "device 0" to "decide "cpu"" in case you are using your cpu for training the model. If you have a CUDA (NVIDIA) GPU, let it stay as it is.
We will be using COLAB for training only so you might not have to run this training custom model code. I will provide you with the trained model in the repository.

*Note: Type in "exit" in the terminal in case you want to get out of the bash command.*

```bash
python train.py --batch 16 --cfg cfg/training/yolov7.yaml --epochs 600 --data xardarii-yc-3/data.yaml --weights 'yolov7.pt' --device 0 
```
### Some important parameters
- **workers (int):** how many subprocesses to parallelize during training
- **img (int):** the resolution of our images. For this project, the images were resized to 1280 x 720
- **batch_size (int):** determines the number of samples processed before the model update is created
- **device:** the sort of device you are using. GPU or CPU. GPUs have numberings, if you are using single GPU then set it to 0 or else the 1,2,3,4 (if you have more than one GPU) etc depending on the GPU you want to use. Set it to CPU if you dont have a GPU.
- **epochs:** An epoch is the number of passes a training dataset takes around an algorithm.

## F1 & Precision-Recall Curves

The F1 graph determines the accuracy of our model.
It is a curve that combines precision (PPV) and Recall (TPR) in a single visualization. For every threshold, you calculate PPV and TPR and plot it. The higher on y-axis your curve is the better your model performance.
Accuracy is calculated as follows:

The Precision-Recall curve shows the tradeoff between precision and recall for different threshold. They are defined as follows:
- Precision: Precision refers to the number of true positives divided by the total number of positive predictions (i.e., the number of true positives plus the number of false positives).
- Recall: the number of true positives divided by the sum of true posititves and false negatives.

#### Run the "F1PrecisionRecall.py" script
Run the script below or the "F1PrecisionRecall.py" script in the AllPythonScripts folder for the curves.

```bash
from IPython.display import Image
display(Image("F:/YOLORepository/yolov7/runs/train/exp3/F1_curve.png", width=600, height=600))
display(Image("F:/YOLORepository/yolov7/runs/train/exp3/PR_curve.png", width=600, height=600))
display(Image("F:/YOLORepository/yolov7/runs/train/exp3/confusion_matrix.png", width=600, height=600))
```


## Inferencing

Referred to as "putting an ML model into production". 

An ML lifecycle can be broken up into two main, distinct parts. The first is the training phase, in which an ML model is created or “trained” by running a specified subset of data into the model. ML inference is the second phase, in which the model is put into action on live data to produce actionable output. The data processing by the ML model is often referred to as “scoring,” so one can say that the ML model scores the data, and the output is a score.

#### Run the following python code for Inferencing
```bash
import os
import sys
sys.path.append('F:/YOLORepository/yolov7')

import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

```
## Configuration Parameters

Simply run the ConfigurationParam.py script or the code below
Change the path of both weights and yaml file. It is already done by me, you don't have to change the paths. Just run the following python code snippets

- weights will be in yolov7 main folder -> runs -> train and then select the appropriate weight
- yaml yolov7 main folder -> Trash-5, there you will find yaml file

```bash
classes_to_filter = None  # You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]

opt  = {
    
    "weights": "F:/YOLORepository/yolov7/runs/train/exp3/weights/best.pt",        # Path to weights file default weights are for nano model
    "yaml"   : "xardarii-yc-3/data.yaml",
    "img-size": 640,        # default image size
    "conf-thres": 0.25,     # confidence threshold for inference.
    "iou-thres" : 0.45,     # NMS IoU threshold for inference.
    "device" : '0',         # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None
    }
```

## Webcam Setup
Once you are done with the training, inference, configuration parameters. It is time to test our model.
Use the following command in the terminal to startup the webcam.

```bash
python detect.py --weights runs/train/exp3/weights/best.pt --device cpu --source 0         #replace cpu with 0,1,2,3 or 4 if you are using an NVIDIA CUDA GPU.
```

**TADAAA!! You're done detecting!! <3**

![image](https://user-images.githubusercontent.com/108223404/232883997-0b90773a-f83e-4e5f-ac17-7bdc8c21e8ad.png)

