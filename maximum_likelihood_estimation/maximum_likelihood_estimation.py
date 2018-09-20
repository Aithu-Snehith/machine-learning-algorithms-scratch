
# coding: utf-8

# # Importing packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # Parameters

# In[2]:


samples = 10000
n_binomial = 10


# # Generating test data

# In[3]:


x = np.random.chisquare(5,samples)  #some random distribution other than the below
plt.hist(x,bins = 100)
plt.show()


# # Estimating the parameters

# In[4]:


p_binomial = np.mean(x)/n_binomial
x_binomial = np.random.binomial(n_binomial , p_binomial , samples)

beta_exponential = np.mean(x)
x_exponential = np.random.exponential(beta_exponential , samples)

lambda_poisson = np.mean(x)
x_poisson = np.random.poisson(lambda_poisson , samples)

mean_guass = np.mean(x)
variance_guass = np.mean(np.square(x - mean_guass))
x_guassian = np.random.normal(mean_guass , variance_guass , samples)

mu_laplacian = np.median(x)
lambda_laplacian = np.mean(abs(x - mu_laplacian))
x_laplace = np.random.laplace(mu_laplacian , lambda_laplacian , samples)


# # Ploting the histograms of the samples drawn from distributions

# In[5]:


plt.figure(figsize = (25,20))
plt.subplot(3,2,1)
plt.hist(x,bins = 100)
plt.title("ground truth")

plt.subplot(3,2,2)
plt.hist(x_binomial,bins = 100)
plt.title("Binomial estimation")

plt.subplot(3,2,3)
plt.hist(x_exponential,bins = 100)
plt.title("Expontntial estimation")

plt.subplot(3,2,4)
plt.hist(x_poisson,bins = 100)
plt.title("Poisson estimation")

plt.subplot(3,2,5)
plt.hist(x_guassian,bins = 100)
plt.title("Guassian estimation")

plt.subplot(3,2,6)
plt.hist(x_laplace,bins = 100)
plt.title("laplacian estimation")

plt.show()

