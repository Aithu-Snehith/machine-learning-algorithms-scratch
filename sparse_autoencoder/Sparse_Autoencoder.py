
# coding: utf-8

# # Importing the packages

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread , imresize
import os
import time


# # Parameters

# In[4]:


hidden_nodes = 1024
lr = 1e-3
lr_kl = 0.1
epochs = 300
sparsity = 0.2


# # Declaring functions

# In[5]:


# Sigmoid

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#derivative of sigmoid
def deriv_sigmoid(x):
    return np.exp(-x)/((1+ np.multiply(np.exp(-x),np.exp(-x))))

# softmax

def softmax(x):
    out = np.zeros(x.shape)
    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            out[i,j] = np.exp(x[i,j])/np.sum(np.exp(x[i]))
    return out

# sum of Squared error

def squared_error(y_train, y_predicted):
    return np.sum(np.multiply(y_train - y_predicted , y_train - y_predicted))


# In[6]:


def load_flattened_images(Loc):
    Images = []
    for root, dirs, files in os.walk(Loc):
        for file in files:
            Image = imread(os.path.join(root, file))
#             Image = imresize(Image, (14,14))
            Image = np.round(Image / 255.0)
#             Image = Image/255.0
            Images.append(Image.flatten())
                
    Images = np.asmatrix(Images)
    print(Images.shape)

    return Images


# In[7]:


## fitting the model

def net_fit(x_train , y_train , epochs = 100 , hidden_nodes = 2 , lr = 1e-3 , lr_kl = 1e-2):
    input_dim = x_train.shape[1]
    training_samples = x_train.shape[0]
    output_dim = y_train.shape[1]
    costs = []
    z_means = []
    x_train = np.hstack((np.ones((training_samples , 1)), x_train))
    #initializig the parameters
    alpha = np.asmatrix(np.random.normal(0,1,(input_dim + 1 , hidden_nodes)))
    beta = np.asmatrix(np.random.normal(0,1,(hidden_nodes+1 , output_dim)))
    
    #looping for number of itretions
    for epoch in range(0,epochs):
        #finding z matrix
        z_raw = x_train * alpha 
        z = sigmoid(z_raw)
        z_biased = np.asmatrix(np.hstack((np.ones((training_samples,1)),z)))
        
        #finding y matrix
        y_raw = z_biased * beta
        y_predicted = sigmoid(y_raw)
        
        ##finding the cost
        cost = squared_error(y_train , y_predicted) + lr_kl * np.sum((sparsity*np.log(sparsity/np.mean(z , axis = 0))) + ((1-sparsity)*np.log((1-sparsity)/(1-np.mean(z , axis = 0)))))
        costs.append(cost)
        z_means.append(np.mean(z))
        #finding gradient w.r.t beta
        delta = np.multiply((y_predicted - y_train), deriv_sigmoid(y_raw))
        d_beta = z_biased.T * delta
       
        temp_beta = beta[1:,:]
        
        #finding gradient w.r.t alpha
        ss = np.multiply((delta * temp_beta.T),deriv_sigmoid(z_raw))
        d_alpha_raw = x_train.T * ss
#         kl divergence derivative
        d_kl = t = np.repeat(((-sparsity/np.mean(z , axis = 0)) + ((1-sparsity)/(1-np.mean(z , axis = 0)))).reshape(1,hidden_nodes) , training_samples , axis = 0)
        d_alpha_kl = (1/training_samples)*(x_train.T * d_kl)
        d_alpha = d_alpha_raw + lr_kl*d_alpha_kl
#         print(np.max(d_alpha) ,np.max(d_alpha_raw) ,np.max(d_kl) , np.max(d_beta) )
        
        #updating the weights
        beta = beta - lr * d_beta
        alpha = alpha - lr*d_alpha
#         print(np.max(alpha) , np.max(beta) , np.min(alpha) , np.min(beta))
        print("\nEpoch: " + str(epoch+1) + "   cost : " + str(cost))
    return alpha , beta , costs , z_means


# In[8]:


#prediction

def net_predict(x_test , alpha , beta ):
    testing_samples = x_test.shape[0]
    #adding bias
    x_test = np.hstack((np.ones((testing_samples , 1)), x_test))
    
    #finding z matrix
    z_raw = x_test * alpha
    z = sigmoid(z_raw)
    z_biased = np.asmatrix(np.hstack((np.ones((testing_samples,1)),z)))
    
    #finding Y matrix (predicting the outputs)
    y_raw = z_biased * beta
    y_predicted = sigmoid(y_raw)
    y_predicted = np.round(y_predicted)   ##comment it if solving for regression
    return y_predicted


# # Generating training data

# In[9]:


x_train = np.load("train_set_10000.npy")
print(x_train.shape)


# # Training the Model

# In[8]:


# alpha , beta , losses = net_fit(x_Train , y_train_and , hidden_nodes = hidden_nodes , epochs = epochs ,lr = lr)
tic = time.time()
alpha , beta , losses , means = net_fit(x_train , x_train , hidden_nodes = hidden_nodes , epochs = epochs ,lr = lr , lr_kl = lr_kl)
print("time taken: "+ str(time.time() - tic) + "sec")
# print("\nalpha:\n",alpha ,"\nbeta:\n", beta,"\n" ,"\nloss:\n", losses[epochs-1])
np.save("alpha_weights_sae_v3.npy" , alpha)
np.save("beta_weights_sae_v3.npy" , beta)


# ### Predicting

# In[20]:


#testing samples
alpha = np.load("alpha_weights_sae_v3.npy")
beta = np.load("beta_weights_sae_v3.npy")
# x_test = load_flattened_images("/home/snehith/Documents/machine learning/datasets/mnist/mnistasjpg/testSample/")
# np.save("test_set_350.npy" , x_test)
x_test = np.asmatrix(np.load("test_set_350.npy"))
#predicting the output
res = net_predict(x_test , alpha , beta)
# print(losses.shape)
plt.figure(figsize = (20,15))
for i in range(0,4):
    plt.subplot(4,2,2*i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.subplot(4,2,2*i+2)
    plt.imshow(res[i].reshape(28,28))


# In[10]:


#ploting the cost vs epochs
plt.plot(np.arange(epochs),losses)
plt.title("training samples: " + str(x_train.shape[0]))
plt.xlabel("epochs")
plt.ylabel("cost")
plt.show()

