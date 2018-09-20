
# coding: utf-8

# In[82]:


import numpy as np
import matplotlib.pyplot as plt


# In[83]:


poly_order = 4
# Number of training samples
N = 10
# Generate equispaced floats in the interval [0, 2*pi]
x_train = np.linspace(0, 2*np.pi, N)
# Generate noise
mean = 0
std = 0.05
# Generate some numbers from the sine function
y = np.sin(x_train)
# Add noise
y += np.random.normal(mean, std, N)
#defining it as a matrix
y_train = np.asmatrix(y.reshape(N,1))


# # adding the bias and higher order terms to x

# In[84]:


x = np.append(np.ones((N,1)),x_train.reshape((N,1)),axis = 1)
for i in range(0,poly_order-1):
	x = np.append(x,(x_train.reshape((N,1)))**(i+2),axis = 1)
x = np.asmatrix(x)
print(x.shape)
print(x)


# # finding the optimum weights

# In[85]:


w = (x.T*x).I*x.T*y_train
print(w)


# # generating test samples

# In[86]:


M = 100
x_test = np.linspace(0, 2*np.pi, M)
x_t = np.asmatrix(np.append(np.ones((M,1)),x_test.reshape(M,1),axis = 1))
for i in range(0,poly_order-1):
	x_t = np.append(x_t,(x_test.reshape((M,1)))**(i+2),axis = 1)
x_t = np.asmatrix(x_t)
print(x.shape)


# # predicting the outputs for the test sample

# In[87]:


y_test = x_t*w


# # Error ( Cost)

# In[88]:


y_fin = x * w

print("error:- ",np.asmatrix(y_train-y_fin).T*np.asmatrix(y_train-y_fin))


# # ploting the results

# In[89]:


plt.plot(x_train,y_train,'o',label = 'training data')
plt.plot(x_test,y_test,'.',label = 'testing data')
plt.legend()
plt.grid()
plt.title("Linear regression using a polynomial basis function.")
plt.show()


# # Observations

# - Model is approximated by a polinomial function
# - Noise is added to the training data labels
# 
# Polynomial order - Errors (10 training samples)
# - 9  -> 6.63505983e-06
# - 10 -> 11.34967298
# - 4  -> 0.06597847
# 
# Clearly as the number of parameters crosses the number of training points, the model is performing very poorly
