

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:48:01 2022

@author: Mihreteab
"""

from tensorflow import keras
from tensorflow.keras import layers
from libs import cf_m2m
from libs import ds_layer_p2p
from libs import mass_layer_p2p


# cf_uni_ds model architecture
def get_model(img_size, ldr_size, prototypes, singleton_num):
    
    inputs_img=keras.Input(shape=img_size+(3,), name='rgb')
    inputs_ldr=keras.Input(shape=ldr_size+(3,), name='lidar')
    
    #x: image processing brach
    #y: lidar processing brach
    
    # Encoder
    #B1: Block 1
    x=layers.ZeroPadding2D(padding=1, name='Block1_rgb_zp')(inputs_img)
    x=layers.Conv2D(32,4,strides=2,activation='elu', name='Block1_rgb_conv')(x)
    y=layers.ZeroPadding2D(padding=1, name='Block1_lidar_zp')(inputs_ldr)
    y=layers.Conv2D(32,4,strides=2,activation='elu', name='Block1_lidar_conv')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block1_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block1_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block1_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block1_lidar_cf')([y_fsn,y])
    #B2: Block 2
    x=layers.ZeroPadding2D(padding=1,name='Block2_rgb_zp')(x_fsn)
    x=layers.Conv2D(32,3,strides=1,activation='elu',name='Block2_rgb_conv')(x) 
    y=layers.ZeroPadding2D(padding=1,name='Block2_lidar_zp')(y_fsn)
    y=layers.Conv2D(32,3,strides=1,activation='elu', name='Block2_lidar_conv')(y) 
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block2_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block2_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block2_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block2_lidar_cf')([y_fsn,y])
    #B3: Block 3
    x=layers.ZeroPadding2D(padding=1,name='Block3_rgb_zp')(x_fsn)
    x=layers.Conv2D(64,4,strides=2,activation='elu', name='Block3_rgb_conv')(x)  
    y=layers.ZeroPadding2D(padding=1, name='Block3_lidar_zp')(y_fsn)
    y=layers.Conv2D(64,4,strides=2,activation='elu', name='Block3_lidar_conv')(y)   
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block3_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block3_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block3_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block3_lidar_cf')([y_fsn,y])
    #B4: Block 4
    x=layers.ZeroPadding2D(padding=1, name='Block4_rgb_zp')(x_fsn)
    x=layers.Conv2D(64,3,strides=1,activation='elu', name='Block4_rgb_conv')(x) 
    y=layers.ZeroPadding2D(padding=1, name='Block4_lidar_zp')(y_fsn)
    y=layers.Conv2D(64,3,strides=1,activation='elu', name='Block4_lidar_conv')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block4_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block4_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block4_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block4_lidar_cf')([y_fsn,y])
    #B5: Block 5
    x=layers.ZeroPadding2D(padding=1,name='Block5_rgb_zp')(x_fsn)
    x=layers.Conv2D(128,4,strides=2,activation='elu', name='Block5_rgb_conv')(x) 
    y=layers.ZeroPadding2D(padding=1, name='Block5_lidar_zp')(y_fsn)
    y=layers.Conv2D(128,4,strides=2,activation='elu', name='Block5_lidar_conv')(y)  
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block5_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block5_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block5_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block5_lidar_cf')([y_fsn,y])
    
    # Context module
    #B6: Block 6
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block6_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block6_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block6_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block6_lidar_dp')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block6_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block6_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block6_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block6_lidar_cf')([y_fsn,y])
    #B7: Block 7
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block7_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block7_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block7_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block7_lidar_dp')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block7_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block7_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block7_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block7_lidar_cf')([y_fsn,y])
    #B8: Block 8
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(1,2),activation='elu', name='Block8_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block8_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(1,2),activation='elu', name='Block8_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block8_lidar_dp')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block8_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block8_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block8_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block8_lidar_cf')([y_fsn,y])
    #B9: Block 9
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(2,4),activation='elu', name='Block9_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block9_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(2,4),activation='elu', name='Block9_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block9_lidar_dp')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block9_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block9_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block9_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block9_lidar_cf')([y_fsn,y])
    #B10: Block 10
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(4,8),activation='elu', name='Block10_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block10_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(4,8),activation='elu', name='Block10_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block10_lidar_dp')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block10_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block10_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block10_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block10_lidar_cf')([y_fsn,y])
    #B11: Block 11
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(8,16),activation='elu', name='Block11_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block11_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(8,16),activation='elu', name='Block11_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block11_liar_dp')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block11_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block11_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block11_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block11_lidar_cf')([y_fsn,y])
    #B12: Block 12
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(16,32),activation='elu', name='Block12_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block12_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(16,32),activation='elu', name='Block12_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block12_lidar_dp')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block12_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block12_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block12_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block12_lidar_cf')([y_fsn,y])
    #B13: Block 13
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block13_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block13_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block13_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block13_lidar_dp')(y)
    # Fusion
    x_fsn=cf_m2m.c_fusion_wt(name='Block13_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block13_rgb_cf')([x_fsn,x])
    y_fsn=cf_m2m.c_fusion_wt(name='Block13_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block13_lidar_cf')([y_fsn,y])
    #B14: Block 14
    x=layers.Conv2D(128,1,padding="same",activation='elu', name='Block14_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block14_rgb_dp')(x)
    y=layers.Conv2D(128,1,padding="same",activation='elu', name='Block14_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block14_lidar_dp')(y)
    # Fusion
    y_wtd=cf_m2m.c_fusion_wt(name='Block14_lidar_cfw')(y)
    x_wtd=cf_m2m.c_fusion_wt(name='Block14_rgb_cfw')(x)
    ctx_out=layers.Add(name='context_output')([y_wtd,x_wtd])
    
    # Decoder
    #B15: Block 15
    x=layers.Conv2DTranspose(64,4,strides=2,activation='elu', padding='same', name='Block15_convtp')(ctx_out)
    #B16: Block 16
    x=layers.ZeroPadding2D(padding=1,name='Block16_zp')(x)
    x=layers.Conv2D(64,3,strides=1,activation='elu', name='Block16_conv')(x)
    #B17: Block 17
    x=layers.Conv2DTranspose(32,4,strides=2,activation='elu', padding='same', name='Block17_convtp')(x)
    #B18: Block 18
    x=layers.ZeroPadding2D(padding=1, name='Block18_zp')(x)
    x=layers.Conv2D(32,3,strides=1,activation='elu', name='Block18_conv')(x)
    #B19: Block 19
    feature=layers.Conv2DTranspose(8,4,strides=2, activation='elu', padding='same', name='Block19_convtp')(x)
   
    # Evidential formulation
    x = ds_layer_p2p.DS1(prototypes, name='distance_prototype')(feature)
    x = ds_layer_p2p.DS1_activate(name='prototype_activation')(x)
    x = ds_layer_p2p.DS2(singleton_num, name='prototype_singleton_mass')(x)
    x = ds_layer_p2p.DS2_omega(name='prototype_singleton_omega_mass')(x)
    x = ds_layer_p2p.DS3_Dempster(name='unorm_combined_mass')(x)
    x = ds_layer_p2p.DS3_normalize(name='norm_combined_mass')(x)
    x = mass_layer_p2p.SelectSingleton(name='singleton_mass')(x)
    
    model=keras.Model(inputs=[inputs_img, inputs_ldr], outputs=x)
    return model


img_size = (384,1248) # camera image size
ldr_size= (384,1248) # projected lidar image size
prototypes = 6 # number of prototypes
singleton_num = 2 # number of singletons

#Build model
model = get_model(img_size, ldr_size, prototypes, singleton_num)
model.summary()

# save model
model.save('model_cf_uni_ds',save_format='tf')





