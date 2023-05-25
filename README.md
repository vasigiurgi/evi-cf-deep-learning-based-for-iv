# Evidential deep learning-based multi-modal environment perception for intelligent vehicles
KITTI Road, KITTI Semantic, Road Detection, Semantic Segmentation, Evidence Theory

!!! Code to be reviewed and updated accordingly



# Overview 
In this work, evidence theory is combined with a camera-lidar-based deep learning fusion architecture. The coupling is based on generating basic belief functions using distance to prototypes. It also uses a distance-based decision rule.
The project is an extended work of the cross-fusion reduction repository:
https://github.com/geletumn/cf_reduction
The new benchmark introduces evidence theory for the decision-making part. 

# Architecture:
![arch_1-1](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/a20c2b0f-ea65-46e2-a73c-ba08f767c261)


# Installation 
Python version and TF framework
**python 3.7**

**tensorflow==2.8.0**

# Getting Started
Before running the jupyter, the user is advised to import the corresponding libraries and activation functions related to evidential formulation part.
In the _utils.py_ file, simplification of the classes is intransigent. Two architectures can be found in model_arch and their corresponding weights 
 
**ds_layer_p2p**

**DS1_activate**

# Decision Making
The belief theory approach Distance to prototypes using Interval Dominance is adapted to road segmentation as well as multi-class segmentation 

# Dataset

Two datasets both from KITTI Benchmark were used in this work: KITTI road and KITTI semantic pixel-wise. 
https://www.cvlibs.net/datasets/kitti/
The second dataset used for semantic segmentation contains 127 frames (lidar and camera) from the KITTI raw dataset. 
The number of classes is simplified to 3 classes: road, vehicle and background. 

# Results
The evidential formulation introduces a extra class called _ignorance_ to treat the uncertaintes. More deeper explanations are treated in the paper that is in the process of being published. 

Road Detection: Bird Eye View

Evidential Prediction

![uu_road_000028_evi_mix_old](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/206a2be5-b604-4f5c-82d1-3d2ed5d932b0)

Probabilistic Prediction

![uu_road_000028_prob_mix_old](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/d22f19b9-3162-4185-a60c-fe37df7c20c4)


Semantic Segmentation

Far-end points are classified as ignorance, rather than making a wrong prediction.

Predicted frame (4 classes, including ignorance)

![pred_test](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/dce440c8-3b16-4f9f-bc39-419bf700fc56)

Groundtruth frame, simplified with 3 classes. 

![gt_test](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/ef87a3fd-db12-4e63-b5a6-8c749951ffcb)

Predicted frame (4 classes, including ignorance)

![pred_000162_10](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/343d6507-1da5-40d4-b96a-9c1fde72b040)

Groundtruth frame, simplified with 3 classes. 

![gt_000162_10](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/431767af-6fba-4649-afc7-9af154649142)


# Citing 
**Evidential deep learning-based multi-modal environment perception for intelligent vehicles**
[to be updated]



