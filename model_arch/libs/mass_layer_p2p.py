import tensorflow as tf

#This takes the mass of the singletons as outputs

class SelectSingleton(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelectSingleton, self).__init__(**kwargs)
        
    def call(self, inputs):
        
        mass_class = inputs[:,:,:,0:-1]
        
        return mass_class
    
    def get_config(self):
        config = super(SelectSingleton, self).get_config()
        return config