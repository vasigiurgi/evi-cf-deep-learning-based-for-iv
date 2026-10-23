#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from skimage.transform import rotate


def img_path (dir, ext):
    # Path to files
    img_path=sorted(
    [
        os.path.join(dir, fname)
        for fname in os.listdir(dir)
        if fname.endswith(ext)
    ]
)
    return img_path

# category label: a single class for now, so the stratified split is a plain k-fold
def cat_label (paths_list):
    return [0] * len(paths_list)

# stratified 10-fold cross-validation
def train_val_split(input_dir_rgb, input_dir_velo, target_dir):
    """
    Parameters
    ----------
    
    input_dir_rgb : string
        DESCRIPTION. path to the rgb camera image
    input_dir_velo : string
        DESCRIPTION. path to the projected velodyne 
    target_dir : string
        DESCRIPTION. path to the ground truth image

    Returns
    -------
    split_dic : dictionary
        DESCRIPTION. dictionary of 10 train/val split for rgb, velo, target
    """
    split_dic={}
    # Keep only frames present in all three folders, so the splits stay aligned
    common= set(os.listdir(input_dir_rgb)) & set(os.listdir(input_dir_velo)) & set(os.listdir(target_dir))
    keep= lambda paths: [p for p in paths if os.path.basename(p) in common]
    # Prepare the image paths
    #RGB
    img_paths_rgb= keep(img_path(input_dir_rgb, ".png"))
    img_paths_rgb= np.array(img_paths_rgb)
    # Velo
    img_paths_velo= keep(img_path(input_dir_velo, ".png"))
    img_paths_velo= np.array(img_paths_velo)
    # Target (ground truth)
    target_img_paths= keep(img_path(target_dir, ".png"))
    target_img_paths= np.array(target_img_paths)
    # Path label
    label= cat_label(img_paths_rgb)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    # camera split
    i=1
    for train_index, val_index in skf.split(img_paths_rgb, label):
        train_name= "train_cam_" + str(i)
        val_name= "val_cam_" + str(i)
        split_dic[train_name]=list(shuffle(img_paths_rgb[train_index], random_state=0))
        split_dic[val_name]= list(shuffle(img_paths_rgb[val_index], random_state=0))
        i+=1
    # velo split
    i=1
    for train_index, val_index in skf.split(img_paths_velo, label):
        train_name= "train_velo_" + str(i)
        val_name= "val_velo_" + str(i)
        split_dic[train_name]= list(shuffle(img_paths_velo[train_index], random_state=0))
        split_dic[val_name]= list(shuffle(img_paths_velo[val_index], random_state=0))
        i+=1
    # target(groud truth) split
    i=1
    for train_index, val_index in skf.split(target_img_paths, label):
        train_name= "train_target_" + str(i)
        val_name= "val_target_" + str(i)
        split_dic[train_name]= list(shuffle(target_img_paths[train_index], random_state=0))
        split_dic[val_name]= list(shuffle(target_img_paths[val_index], random_state=0))
        i+=1
        
    return split_dic

def split_check(paths_list):
    #split a path in to file keys for preparing split checking
    
    file_names=[]
    for item in  paths_list:
        file_key=os.path.split(item)[1].split('.')[0]
        file_names.append(file_key)
    return file_names



def write_path(out_dir,paths_list,file_name):
    # Write the validation paths to text file
    path_name=os.path.join(out_dir,file_name)
    with open(path_name, 'w') as f:
      # write elements of list
      for items in paths_list:
            f.write('%s\n' %items)
    return 0

#Prepare Sequence class to load & vectorize batches of data: individual random aug, not at epch end
class kittiroad(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths_rgb,input_img_paths_velo, target_img_paths,val=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths_rgb = input_img_paths_rgb
        self.input_img_paths_velo = input_img_paths_velo
        self.target_img_paths = target_img_paths
        self.val=val

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths_rgb = self.input_img_paths_rgb[i : i + self.batch_size]
        batch_input_img_paths_velo=self.input_img_paths_velo[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x1 = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        angle=random.uniform(-20, 20)
        for j, path in enumerate(batch_input_img_paths_rgb):
            img = load_img(path)
            w,h=img.size

            dh=int((self.img_size[0]-h)/2)
            dw=int((self.img_size[1]-w)/2)
            zero_padded_img=np.zeros(self.img_size+(3,), dtype="float32")
            zero_padded_img[dh:dh+h,dw:dw+w,:]=img
            zero_padded_img/=255.0
            if self.val==False: # training
                zero_padded_img=rotate(zero_padded_img,angle)
            x1[j] = zero_padded_img
            
        x2 = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")   
        for j, path in enumerate(batch_input_img_paths_velo):
            img = load_img(path)
            w,h=img.size

            dh=int((self.img_size[0]-h)/2)
            dw=int((self.img_size[1]-w)/2)
            zero_padded_img=np.zeros(self.img_size+(3,), dtype="float32")
            zero_padded_img[dh:dh+h,dw:dw+w,:]=img
            zero_padded_img/=255.0
            if self.val==False: # training 
                zero_padded_img=rotate(zero_padded_img,angle)
            x2[j] = zero_padded_img
            
        x1_x2=[x1,x2]   
        
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        sw = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            
            mask = load_img(path)
            mask = np.asanyarray(mask)

            img_height = mask.shape[0]
            img_width = mask.shape[1]

           # Roads 
            road_region= np.zeros((img_height, img_width), dtype="uint8")
            road_region[np.where(mask[:,:,0]==128.0)[0], np.where(mask[:,:,0]==128.0)[1]]=1

            assert (road_region==1).any(), 'INVALID INPUT:1 no road_region mask'

            # Cars: car + truck + bus 
            car_region = np.zeros((img_height, img_width), dtype="uint8")
            car_region[np.where(mask[:,:,2]==142.0)[0], np.where(mask[:,:,2]==142.0)[1]]=1  
            car_region[np.where(mask[:,:,2]==70.0)[0], np.where(mask[:,:,2]==70.0)[1]]=1 
            car_region[np.where(mask[:,:,1]==70.0)[0], np.where(mask[:,:,1]==70.0)[1]]=0  
            car_region[np.where(mask[:,:,2]==100.0)[0], np.where(mask[:,:,2]==100.0)[1]]=1  
            car_region[np.where(mask[:,:,1]==100.0)[0], np.where(mask[:,:,1]==100.0)[1]]=0  
            car_region[np.where(mask[:,:,1]==80.0)[0], np.where(mask[:,:,1]==80.0)[1]]=0  

            c_everythingelse_region = road_region+car_region
            everythingelse_region = np.zeros((img_height, img_width), dtype="uint8")
            everythingelse_region[np.where(c_everythingelse_region!=1)[0], np.where(c_everythingelse_region!=1)[1]]=1 
            
            assert (everythingelse_region==1).any(), 'INVALID INPUT:1 no everythinselse_region mask'

            gt_mask = np.stack((road_region, car_region, everythingelse_region), axis=-1)
            img = gt_mask
            h,w,c=img.shape
            dh=int((self.img_size[0]-h)/2)
            dw=int((self.img_size[1]-w)/2)
            zero_padded_img=np.zeros(self.img_size+(3,), dtype="uint8")
            zero_padded_img[dh:dh+h,dw:dw+w,:]=img
            if self.val==False: # training
                zero_padded_img=rotate(zero_padded_img,angle)
            zero_padded_img=zero_padded_img>0    
            assert (zero_padded_img[:,:,0]==1).any(), 'INVALID INPUT:2 no road_region mask'
            assert (zero_padded_img[:,:,2]==1).any(), 'INVALID INPUT:2 no everythinselse_region mask'
            
            y[j]=zero_padded_img 
            zero_padded_img=zero_padded_img*np.array([1.2,2.5,1],dtype='float32')
            zero_padded_img=zero_padded_img[:,:,0]+zero_padded_img[:,:,1]+zero_padded_img[:,:,2]
            zero_padded_img=np.expand_dims(zero_padded_img,axis=-1)
            sw[j]=zero_padded_img
        return x1_x2, y, sw
