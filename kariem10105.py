#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

X = [1, 2, 3]
Wf, Wnf, bf = 0.5, 0.1, 0
Wi, Wni, bi = 0.6, 0.2, 0
Wc, Wnc, bc = 0.7, 0.3, 0
Wo, Wno, bo = 0.8, 0.4, 0

h_prev, c_prev = 0, 0

for x_t in X:
    f_t = sigmoid(Wf * x_t + Wnf * h_prev + bf)
    i_t = sigmoid(Wi * x_t + Wni * h_prev + bi)
    c_tilde = tanh(Wc * x_t + Wnc * h_prev + bc)
    c_t = f_t * c_prev + i_t * c_tilde
    o_t = sigmoid(Wo * x_t + Wno * h_prev + bo)
    h_t = o_t * tanh(c_t)
    
    h_prev, c_prev = h_t, c_t

Wy, by = 4, 0
y_pred = Wy * h_prev + by

print(f"Predicted X4: {y_pred:.3f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




