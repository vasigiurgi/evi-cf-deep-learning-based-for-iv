# Evidential deep learning-based multi-modal environment perception for intelligent vehicles
KITTI Road, KITTI Semantic, Road Detection, Semantic Segmentation, Evidence Theory

# Overview 
In this work, evidence theory is combined with a camera-lidar-based deep learning fusion architecture. The coupling is based on generating basic belief functions using distance to prototypes. It also uses a distance-based decision rule.

# Installation 

_python 3.7_
_tensorflow _

# Getting Started
Before running the _jupyter_, the user is advised to import the corresponding libraries and activation functions related to evidential formulation part:
_ds_layer_p2p_
_DS1_activate_

# Interval Dominance
The belief theory approach is adapted to road segmentation as well as multi-class segmentation 

# Dataset

Two datasets both from KITTI Benchmark were used in this work: KITTI road and KITTI semantic pixel-wise. 
https://www.cvlibs.net/datasets/kitti/
The second dataset used for semantic segmentation contains 127 frames (lidar and camera) from the KITTI raw dataset. 
The number of classes is simplified to 3 classes: road, vehicle and background. 

# Results

The evidential formulation introduces a extra class called _ignorance_ to treat the uncertaintes. 
Far-end points are classified as ignorance, rather than making a wrong prediction
