
# coding: utf-8

# # Importing packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # Defining functions

# In[2]:


def multi_normal_pdf(x , mean , co_var):
    ex = -0.5 * ((x - mean.T).T * co_var.I * (x - mean.T))
    den = np.sqrt(((2*np.pi)**mean.shape[1]) * np.linalg.det(co_var))
    return np.exp(ex)/den


# In[3]:


def gaama(x , mean , co_var , weights):
    pdfs = np.asmatrix(np.zeros((len(weights), x.shape[1])))
    for i in range(0,x.shape[1]):
        for j in range(0,len(weights)):
            pdfs[j , i] = weights[j] * multi_normal_pdf(x[:,i] , mean[j] , np.asmatrix(co_var[j]))
    pdfs_mean = np.asmatrix(np.mean(pdfs , axis = 0))
    pdfs_mean = np.repeat(pdfs_mean , pdfs.shape[0] , axis = 0)
    gama = pdfs / pdfs_mean
    return np.asmatrix(gama)


# In[4]:


def log_likelihood(x , mean , co_var , weights):
    likelihood = 0
    for i in range(0,x.shape[1]):
        temp_pdf = 0
        for j in range(0,len(weights)):
            temp_pdf = temp_pdf + weights[j] * multi_normal_pdf(x[:,i] , mean[j] , np.asmatrix(co_var[j]))
        likelihood = likelihood +  np.log(temp_pdf)
    return likelihood


# # Parameters

# In[86]:


# no of mixtures
# mix_size = 2
mix_size = int(input("no of mixtures: "))
# dimensions of samples
# dim_sample = 5
dim_sample = int(input("dimensions of samples: "))
# no of samples
# samples = 100
samples = int(input("number of samples: "))


# # Generating Training samples

# In[79]:


data = np.zeros((dim_sample , samples))
for i in range(0,mix_size):
#     mean = np.random.randint(5,size = dim_sample)
    mean = np.linspace(i,i+1,dim_sample)
    temp = np.random.randint(5,size = (dim_sample,dim_sample))
    co_var = np.matmul(temp,np.transpose(temp)) 
    data = data + np.transpose(np.random.multivariate_normal(mean , co_var , samples))
data = np.asmatrix(data/mix_size)
# print(data.shape , data[:,0].shape)


# # Initializing parameters

# In[83]:


mean = np.asmatrix(np.random.rand(mix_size,dim_sample))
# co_var = np.eye((mix_size,dim_sample , dim_sample))
co_var = []
for i in range(0,mix_size):
    co_var.append(np.eye((dim_sample)))
co_var = np.array(co_var)
weights = np.ones(mix_size)/mix_size
diff = 1
# print(co_var.shape , mean.shape, weights)


# # Finding the optimal parameters

# In[85]:


while(diff > 0):
#     finding posterior
    gama = gaama(data , mean , co_var , weights)
    init = log_likelihood(data , mean , co_var , weights)
#     print("before: " , mean[0,0] , "   " , co_var[0,0,0] , '   ' , weights[0])
    n_k = np.sum(gama , axis = 1)
    co_var = []
#     updating mean , variance
    for i in range(0,mix_size):
        mean[i] = (np.sum(np.multiply(data, np.asmatrix(np.repeat(gama[i] , dim_sample , axis = 0))),axis = 1).T)/n_k[i]
        temp = np.asmatrix(np.zeros((dim_sample,dim_sample)))
        for j in range(0,samples): 
            temp = temp + (data[:,j]-mean[i].T) * (data[:,j]-mean[i].T).T 
        co_var.append(temp)
    co_var = np.asarray(co_var)
#     updating weights
    weights = n_k/samples
    final = log_likelihood(data , mean , co_var , weights)
    diff = final - init
#     print(diff)
# print("mean:\n" , mean , "\n weights: \n" , weights , "\n variance: \n" , co_var)
print("mean: ")
print(mean)
print("weights:")
print(weights)
print("co-variance")
print(co_var)

