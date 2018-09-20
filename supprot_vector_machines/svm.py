
# coding: utf-8

# # Importing the libraries

# In[1]:


import numpy as np
import cvxpy as cp


# # Loading the dataset

# In[2]:


x = np.genfromtxt('Xsvm.csv' , delimiter = ",")
y = np.genfromtxt('ysvm.csv' , delimiter = ",")
print(x.shape,y.shape)


# In[3]:


z = np.append(y.reshape(500,1),y.reshape(500,1),axis = 1)
print(z.shape)
xy = x*z ##for sigma_over_n(y*x_bar)
# print(x[1:10],y[1:10],xy[1:10])


# # Finding the optimal alphas

# In[4]:


alpha = cp.Variable(500)
a = cp.vstack([alpha , alpha])
#equation to be maximixed ()cost
objective = cp.Maximize(sum(alpha) - 0.5 * cp.sum_squares(cp.sum(cp.multiply(a.T , xy) , axis = 0)))
constraints = [ 0 <= alpha , np.transpose(y)*alpha == 0]
problem = cp.Problem(objective,constraints)
result = problem.solve()  #maximizing the equation
print(result) #optimal cost function
print(alpha.value) #values of alphas


# # Finding the values of W

# In[5]:


v = np.append(alpha.value.reshape(500,1),alpha.value.reshape(500,1),axis = 1)
print(v.shape)
w = np.sum(v * xy , axis = 0)
print(w)


# # Finding Wo

# In[6]:


for i in range(0,500):
    if alpha.value[i] > 0.1:
        w0 = y[i] - np.matmul(w,np.transpose(x[i]))
        print(w0)


# # Testing the Model

# In[7]:


x_test = np.array([[2,0.5],[0.8,0.7],[1.58,1.33], [0.008,0.001]])
# print(x_test.shape)
print("Predictions:")
for i in range(0,4):
    y_test = w0 + np.matmul(w , np.transpose(x_test[i]))
    if(y_test >= 0):
        print("1")
    else:
        print("0")


# # Visualization

# In[9]:


import matplotlib.pyplot as plt

x_train_1 = []
x_train_0 = []
for i in range(0,500):
    if y[i] == 1:
        x_train_1.append(x[i])
    else:
        x_train_0.append(x[i])
x_train_1 = np.asarray(x_train_1)
x_train_0 = np.asarray(x_train_0)
print(x_train_1.shape)
print(x_train_0.shape)

plt.plot(x_train_1[:,0],x_train_1[:,1],'o',label = 'label = 1')
plt.plot(x_train_0[:,0],x_train_0[:,1],'o', label = 'Label = -1')
plt.plot(x_test[:,0],x_test[:,1],'ro',label = 'Test Sample')
plt.legend()
plt.show()

