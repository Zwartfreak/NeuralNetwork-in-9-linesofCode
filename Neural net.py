#!/usr/bin/env python
# coding: utf-8

# # The Hello World of Deep Learning with Neural Networks

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow import keras


# In[ ]:


model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])


# In[ ]:


model.compile(optimizer='sgd', loss='mean_squared_error')


# In[ ]:


xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# In[ ]:


model.fit(xs, ys, epochs=500)


# In[ ]:


print(model.predict([10.0]))

