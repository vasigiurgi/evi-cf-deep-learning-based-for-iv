# -*- coding: utf-8 -*-

#Cross fusion layer: multiplies a layer (lidar or rgb) 
#with a scalar (the fusion weight)

from tensorflow.keras import layers

# Fusion: weighting
class c_fusion_wt (layers.Layer):
    def __init__(self,**kwargs):
        super(c_fusion_wt, self).__init__(**kwargs)
    
    def build (self, input_shape):
        self.w=self.add_weight(shape=(1,), initializer="zero", trainable=True,name='weight') 
        
    def call(self, layer_1):
        return layer_1*self.w 

    def get_config(self):
        config = super(c_fusion_wt, self).get_config()
        return config
