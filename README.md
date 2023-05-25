# Code to be UPDATED

# Evidential deep learning-based multi-modal environment perception for intelligent vehicles
KITTI Road, KITTI Semantic, Road Detection, Semantic Segmentation, Evidence Theory


# Overview 
In this work, evidence theory is combined with a camera-lidar-based deep learning fusion architecture. The coupling is based on generating basic belief functions using distance to prototypes. It also uses a distance-based decision rule.
The project is an extended work of the cross-fusion reduction repository:
https://github.com/geletumn/cf_reduction

Architecture:
![arch_1-1](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/a20c2b0f-ea65-46e2-a73c-ba08f767c261)


# Installation 
Python version and TF framework
**python 3.7**

**tensorflow==2.8.0**

# Getting Started
Before running the jupyter, the user is advised to import the corresponding libraries and activation functions related to evidential formulation part:

**ds_layer_p2p**

**DS1_activate**

# Interval Dominance
The belief theory approach is adapted to road segmentation as well as multi-class segmentation 

# Dataset

Two datasets both from KITTI Benchmark were used in this work: KITTI road and KITTI semantic pixel-wise. 
https://www.cvlibs.net/datasets/kitti/
The second dataset used for semantic segmentation contains 127 frames (lidar and camera) from the KITTI raw dataset. 
The number of classes is simplified to 3 classes: road, vehicle and background. 

# Results
Road Detection

Semantic Segmentation
The evidential formulation introduces a extra class called _ignorance_ to treat the uncertaintes. 
Far-end points are classified as ignorance, rather than making a wrong prediction.
![pred_test](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/dce440c8-3b16-4f9f-bc39-419bf700fc56)
![gt_test](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/ef87a3fd-db12-4e63-b5a6-8c749951ffcb)
![pred_000162_10](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/343d6507-1da5-40d4-b96a-9c1fde72b040)
![gt_000162_10](https://github.com/vasigiurgi/evi-cf-deep-learning-based-for-iv/assets/49117053/431767af-6fba-4649-afc7-9af154649142)


# Citing 
**Evidential deep learning-based multi-modal environment perception for intelligent vehicles**
[to be updated]



