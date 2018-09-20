
# coding: utf-8

# # Importing packages

# In[1]:


import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()
print(digits.DESCR)


# # finding the Principal components

# In[2]:


dim_req = 64
x_digits = np.asmatrix(digits.data)
print(x_digits.shape)

# mean centering
x_digits = x_digits - np.mean(x_digits , axis = 0)

# finding covariance matrix
co_vaiance = (x_digits.T * x_digits)/x_digits.shape[0]
print(co_vaiance.shape)

# finding eigen values and eigen vectors
x_eig_values , x_eig_vectors = np.linalg.eig(co_vaiance)
print(x_eig_values.shape)
print(x_eig_vectors.shape)

# finding the transformed matrix
y_digits = x_digits * x_eig_vectors[:,0:dim_req]
print(y_digits.shape)


# # Ploting the proportian of variance contained in each principle component

# In[3]:


prop_variance = x_eig_values/np.sum(x_eig_values)

plt.plot(np.arange(64),prop_variance , label = "individual")
plt.plot(np.arange(64),np.cumsum(prop_variance) , label = "cummulative")
plt.legend()
plt.xlabel("principal components")
plt.ylabel("proportion of variance")
plt.show()

