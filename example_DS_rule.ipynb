import numpy as np
import tensorflow as tf

class DS3_Dempster(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(DS3_Dempster, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim=input_shape[-2]

    def call(self, inputs):
        m1=inputs[:,:,:,0,:]
        omega1=tf.expand_dims(inputs[:,:,:,0,-1],-1)
        for i in range (self.input_dim-1):
            m2=inputs[:,:,:,(i+1),:]
            omega2=tf.expand_dims(inputs[:,:,:,(i+1),-1], -1)
            combine1=tf.multiply(m1, m2, name=None)
            combine2=tf.multiply(m1, omega2, name=None)
            combine3=tf.multiply(omega1, m2, name=None)
            combine1_2=tf.add(combine1, combine2, name=None)
            combine2_3=tf.add(combine1_2, combine3, name=None) # the omega has three times its actual mass
            combine2_3_omega=tf.divide(combine2_3[:,:,:,-1], 3) # correction
            combine2_3_omega=tf.expand_dims(combine2_3_omega,-1) # correction
            combine2_3=tf.concat([combine2_3[:,:,:,0:-1],combine2_3_omega],-1) #correction
            m1=combine2_3
            omega1=tf.expand_dims(combine2_3[:,:,:,-1], -1)
        return m1

    def get_config(self):
        config = super(DS3_Dempster, self).get_config()
        return config

class DS3_normalize(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(DS3_normalize, self).__init__(**kwargs)

    def call(self, inputs):
        mass_combine_normalize = inputs / tf.reduce_sum(inputs, axis = -1, keepdims=True)
        return mass_combine_normalize

    def get_config(self):
        config = super(DS3_normalize, self).get_config()
        return config

# Define mass functions
m1 = np.array([0.6, 0.3, 0.1])  # m1(A), m1(B), m1(A ∪ B)
m2 = np.array([0.2, 0.3, 0.5])  # m2(A), m2(B), m2(A ∪ B)


# Create a 5D input tensor for the fusion layers
batch_size, height, width, input_dim, mass_dim = 1, 1, 1, 2, 3
inputs = np.zeros((batch_size, height, width, input_dim, mass_dim))

# Assign m1 and m2 to the input tensor
inputs[0, 0, 0, 0, :] = m1  # First mass function
inputs[0, 0, 0, 1, :] = m2  # Second mass function

# Convert to TensorFlow tensor
inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)

# Initialize Dempster fusion layer
dempster_layer = DS3_Dempster()

output_dempster_1 = dempster_layer(inputs_tf)
output_dempster_2 = DS3_normalize()(output_dempster_1)
print("Conjuctive fusion output:", output_dempster_1.numpy())
print("Dempster fusion output:", output_dempster_2.numpy())
