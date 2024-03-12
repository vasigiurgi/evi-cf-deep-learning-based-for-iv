# Evidential deep learning-based multi-modal environment perception for intelligent vehicles
KITTI Road, KITTI Semantic, Road Detection, Semantic Segmentation, Evidence Theory

Adjustements to decision-making and the number of prototypes used have to be considered, and links between the organization of the files. 

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
The belief theory approach Distance to prototypes using Interval Dominance and Decision Based Interval adapted to road segmentation as well as multi-class segmentation.  
The E-CNN-classifier is referred to Tong Zheng's respository (@tongzheng1992) and its inspired by the evidential classifer of Prof. Denouex. 
# Dataset

Two datasets both from KITTI Benchmark were used in this work: KITTI road and KITTI semantic pixel-wise. 
https://www.cvlibs.net/datasets/kitti/
The second dataset used for semantic segmentation contains 127 frames (lidar and camera) from the KITTI raw dataset. 
The number of classes is simplified to 3 classes: road, vehicle and background. 

	The lidar-camera dataset The semantic KITTI dataset has originally only 200 camera images. The dataset is similar to KITTI Stereo and KITTI Flow 2012/2015 datasets. Since the KITTI semantic has no LiDAR frames (like the road dataset for instance), the corresponding 3D point-cloud points of the existing camera frames have to be identified in the big original KITTI raw dataset, which contains the data for all tasks. Hence, for 127 out of the 200 camera images, LiDAR frames have been successfully projected and up-sampled to create dense depth images. A 3D LiDAR point x is mapped into a point y in the camera plane according to the KITTI projection P, rectification R and translation T matrices

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

BibTex:

@INPROCEEDINGS{10186581,
  author={Geletu, Mihreteab Negash and Giurgi, Dănuţ-Vasile and Josso-Laurain, Thomas and Devanne, Maxime and Wogari, Mengesha Mamo and Lauffenburger, Jean-Philippe},
  booktitle={2023 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={Evidential deep learning-based multi-modal environment perception for intelligent vehicles}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/IV55152.2023.10186581}}

Plain Text: 

M. N. Geletu, D. -V. Giurgi, T. Josso-Laurain, M. Devanne, M. M. Wogari and J. -P. Lauffenburger, "Evidential deep learning-based multi-modal environment perception for intelligent vehicles," 2023 IEEE Intelligent Vehicles Symposium (IV), Anchorage, AK, USA, 2023, pp. 1-6, doi: 10.1109/IV55152.2023.10186581.



