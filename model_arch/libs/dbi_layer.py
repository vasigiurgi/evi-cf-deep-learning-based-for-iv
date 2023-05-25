# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:40:18 2022

@author: Mihreteab
"""
import tensorflow as tf

# Belief-Plausibility joint calculation of BBAs, m(.)s   
# m(.)s have only singleton plus omega as focal elements 
class BeliefPlausibility(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BeliefPlausibility, self).__init__(**kwargs)
        
    
    def build(self, input_shape):
        self.cardinal_fod = input_shape[-1]-1  # cardinality of frame of discernment (fod)
        
    @tf.function(autograph=True)         
    def call(self, inputs):
        #inputs: bba from the evidential neural network
        zero_value = tf.zeros_like(inputs)[:,:,:,1]
        zero_value = tf.expand_dims(zero_value, axis=-1)
        unity_value = tf.ones_like(zero_value)
        
        singleton_index = tf.range(0, self.cardinal_fod)
        singleton_index = tf.math.pow(2,singleton_index)
        
        bel = tf.zeros_like(unity_value)
        
        for i in range (pow(2,self.cardinal_fod)-2):
            
            tf.autograph.experimental.set_loop_options(
              shape_invariants=[(bel, tf.TensorShape([None,384,1248,None]))]
               )
            if i == 0:                            
                index = tf.bitwise.bitwise_and(i+1, singleton_index)
                index = tf.cast(index,tf.float32)
                index = tf.math.divide (tf.math.log(index), tf.math.log(2.))
                mask =  ~tf.math.is_inf(index)
                index = tf.boolean_mask(index, mask)
                index = tf.cast(index, tf.int32)
                bel_i = tf.gather(inputs,index,axis=-1)
                bel = tf.reduce_sum(bel_i, axis=-1, keepdims=True)

            if i >= 1:
                index = tf.bitwise.bitwise_and(i+1, singleton_index)
                index = tf.cast(index,tf.float32)
                index = tf.math.divide (tf.math.log(index), tf.math.log(2.))
                mask =  ~tf.math.is_inf(index)
                index = tf.boolean_mask(index, mask)
                index = tf.cast(index, tf.int32)
                bel_i = tf.gather(inputs,index,axis=-1)
                bel_i = tf.reduce_sum(bel_i, axis=-1, keepdims=True)
                bel = tf.concat([bel,bel_i],axis=-1)

        pl = bel        
        bel = tf.concat([zero_value, bel, unity_value],-1)
        mass_omega = tf.expand_dims(inputs[:,:,:,-1], axis=-1)
        pl = tf.math.add(pl,mass_omega)
        pl = tf.concat([zero_value, pl, unity_value], axis=-1)
            
        return [bel,pl]
        
	
    def get_config(self):
        config = super(BeliefPlausibility, self).get_config()
        return config


# Belief-Plausibility joint calculation of focused bba/ logical bba m_x(.)s    
# The focused bba is focused on a non empty subset of Omega, fod
class BeliefPlausibilityFocused(tf.keras.layers.Layer):
    def __init__(self, focal, **kwargs):
        super(BeliefPlausibilityFocused, self).__init__(**kwargs)
        self.focal = focal
    
    def build(self, input_shape):
        self.cardinal_fod = input_shape[-1]-1  # cardinality of frame of discernment (fod)
    
    @tf.function(autograph=True)          
    def call(self, inputs):
        #inputs: bba from the evidential neural network 
        bel = tf.zeros_like(inputs)[:,:,:,1]
        bel = tf.expand_dims(bel, axis=-1)
        pl = tf.zeros_like(bel)
        
        zero_value = tf.zeros_like(bel)
        unity_value = tf.ones_like(bel)
        
        for i in range (pow(2,self.cardinal_fod)-1):
            
            tf.autograph.experimental.set_loop_options(
              shape_invariants=[(bel, tf.TensorShape([None,384,1248,None])),(pl, tf.TensorShape([None,384,1248,None]))]
               )
         
            index_contain = tf.bitwise.bitwise_and(i+1, self.focal)
            
            bel = tf.cond(index_contain == self.focal, 
                          lambda: tf.concat([bel,unity_value], -1), 
                          lambda: tf.concat([bel,zero_value], -1)
                          )
            
            pl = tf.cond(index_contain >0, 
                          lambda: tf.concat([pl,unity_value], -1), 
                          lambda: tf.concat([pl,zero_value], -1)
                          )
            
        return [bel,pl]
	
    def get_config(self):
        config = super(BeliefPlausibilityFocused, self).get_config()
        return config


# Wassertein's distance square
class Wassertein(tf.keras.layers.Layer):
    def __init__(self, focal, **kwargs):
        # focal: the categorical bba m_x(.) is focused on focal, obtained from binary encoding
        super(Wassertein, self).__init__(**kwargs)
        self.focal =  focal
        self.belief_plausibility_bba = BeliefPlausibility()     
        self.belief_plausibility_focused = BeliefPlausibilityFocused(self.focal)
    def call(self, inputs):
        bel_pl_bba = self.belief_plausibility_bba(inputs)
        bel_pl_x = self.belief_plausibility_focused(inputs)
        sum_bba = tf.math.add (bel_pl_bba[0] , bel_pl_bba[1])*0.5
        dif_bba = tf.math.subtract(bel_pl_bba[1], bel_pl_bba[0])*0.5
        sum_x = tf.math.add(bel_pl_x[0], bel_pl_x[1])*0.5
        dif_x = tf.math.subtract(bel_pl_x[1],bel_pl_x[0])*0.5
        
        # difference of sum
        dif_sum = tf.math.subtract(sum_bba,sum_x)
        dif_sum = tf.math.pow(dif_sum,2) # for debugging the power was changed to absolute value
        #dif_sum = tf.math.abs(dif_sum) # debugging insertion
        
        # difference of difference
        dif_dif = tf.math.subtract(dif_bba,dif_x)
        dif_dif = tf.math.pow(dif_dif,2) # for debugging the power was changed to absolute value
        #dif_dif = tf.math.abs(dif_dif) # debugging insertion
        dif_dif = tf.math.divide(dif_dif,3)
        
        distance_wassertein = dif_sum + dif_dif # Wassertein's distance square
        
        return distance_wassertein
	
    def get_config(self):
        config = super(Wassertein, self).get_config()
        return config
  

# belief interval distance
class BeliefIntervalDistance(tf.keras.layers.Layer):
    def __init__(self, space, **kwargs):
        # space: list of elements in decision (i.e., list of integers representing corresponding binary encodings)
        super(BeliefIntervalDistance, self).__init__(**kwargs)
        self.space = space
    
    def build(self, input_shape):
        self.cardinal_fod = input_shape[-1]-1  # cardinality of frame of discernment (fod)
             
    def call(self, inputs):
        #inputs: bba from the evidential neural network
        norm_const = tf.math.pow(2,self.cardinal_fod-1)
        norm_const = tf.cast(norm_const, tf.float32)
        for i in self.space:
            dw = Wassertein(i)(inputs)
            if i == self.space[0]:
                dBI = tf.math.reduce_sum(dw, axis=-1)
                dBI = tf.math.divide(dBI,norm_const)
                #dBI = tf.math.exp(dBI) # debugging insertion
                #dBI = tf.math.subtract(dBI,1) # debugging insertion
                dBI = tf.math.sqrt(dBI) # for debuging the square root part was replaced by other forms
                dBI = tf.expand_dims(dBI,axis=-1)
            else:
                dBI_i = tf.math.reduce_sum(dw, axis=-1)
                dBI_i = tf.math.divide(dBI_i,norm_const)
                #dBI = tf.math.exp(dBI) # debugging insertion
                #dBI = tf.math.subtract(dBI,1) # debugging insertion
                dBI_i = tf.math.sqrt(dBI_i) # for debuging the square root part was replaced by other forms
                dBI_i = tf.expand_dims(dBI_i,axis=-1)
                dBI = tf.concat([dBI,dBI_i],-1)
        #dBI = np.round(dBI.numpy(),7) # to be used on evaluation: to fix no. of decimal points (all results will have same precision)
        #dBI = tf.convert_to_tensor(dBI,dtype=tf.float32)  # to be used on evaluation
        return dBI
	
    def get_config(self):
        config = super(BeliefIntervalDistance, self).get_config()
        return config
    
    
