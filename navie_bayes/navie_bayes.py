
# coding: utf-8

# ## Importing packages  

# In[1]:


import numpy as np


# ## Parameters

# In[2]:


x_test = [-1,-1]


# ## Loading the dataset

# In[3]:


x_train = np.transpose(np.genfromtxt('X.csv',delimiter=','))    # Loading training dataset
y_train = np.genfromtxt('Y.csv',delimiter=',')                  #loading the correspoding labels
print(x_train)


# ## Sperating the training set based on the labels 

# In[4]:


x_train_1 = []
x_train_0 = []
for i in range(0,1000):
    if y_train[i] == 1:
        x_train_1.append(x_train[i])     # appending into new array for labels = 1
    else:
        x_train_0.append(x_train[i])     # appending into new array for labels = -1
x_train_1 = np.asarray(x_train_1)        #converting the lists to arrays
x_train_0 = np.asarray(x_train_0)
print(x_train_1.shape)
print(x_train_0.shape)


# ## Fitting the distributions(finding mean and variances)

# In[5]:


mean_1 = np.mean(x_train_1,axis = 0)
mean_0 = np.mean(x_train_0,axis = 0)
print(mean_1)
print(mean_0)

var_1 = np.var(x_train_1,axis = 0)
var_0 = np.var(x_train_0,axis = 0)
print(var_1)
print(var_0)


# # Testing

# ### Finding the probabilities

# In[6]:


p_1 = np.exp(-((x_test-mean_1)**2)/(2*var_1))/np.sqrt(2*np.pi*var_1)
p_0 = np.exp(-((x_test-mean_0)**2)/(2*var_0))/np.sqrt(2*np.pi*var_0)

print(p_1)
print(p_0)


# ## Estimating the label

# In[7]:


if np.prod(p_1)*(x_train_1.shape[0]/1000) > np.prod(p_0)*(x_train_0.shape[0]/1000):
    print("label = ",1)
else:
    print("label = ",-1)


# # Observations

# - Navie bayes classifier has been implemented
# - The training set(each dimension) is assumed to be guassian
# 
# The predictions on the test data given is as follows:
# - for [1,-1] the estimated label is "1"
# - for [1,1] the estimated label is "1"
# - for [-1,-1] the estimated label is "-1"
# - for [-1,1] the estimated label is "-1"
