
# coding: utf-8

# # Importing the packages

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread , imresize
import os
import time


# # Parameters

# In[30]:


no_latent = 10
lr = 1e-3
epochs = 100
batch_size = 5000


# # Declaring functions

# In[6]:


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


# In[7]:


def load_flattened_images(Loc):
    Images = []
    for root, dirs, files in os.walk(Loc):
        for file in files:
            Image = imread(os.path.join(root, file))
            Image = Image / 255.0
            Images.append(Image.flatten())
                
    Images = np.asmatrix(Images)
    print(Images.shape)

    return Images


# In[34]:


## fitting the model

def net_fit_vae(x_t , y_t , epochs = 100 , no_latent = 10 , lr = 1e-3 , batch_size = 5000):
    input_dim = x_t.shape[1]
    training_samples = x_t.shape[0]
    output_dim = y_t.shape[1]
    costs = []
    x_t = np.hstack((np.ones((training_samples , 1)), x_t))
    #initializig the parameters
    alpha = np.asmatrix(np.random.normal(0,1e-5,(input_dim + 1 , no_latent)))
    gaama = np.asmatrix(np.random.normal(0,1e-5,(input_dim + 1 , no_latent)))
    beta = np.asmatrix(np.random.normal(0,1e-5,(no_latent+1 , output_dim)))
    
    #looping for number of itretions
    for epoch in range(0,epochs):
        for i in range(0,int(training_samples/batch_size)):
            x_train = x_t[i*batch_size:(i+1)*batch_size , :]
            y_train = y_t[i*batch_size:(i+1)*batch_size , :]
            mu_raw = x_train * alpha
            mu = sigmoid(mu_raw)
            sigma_raw = x_train * gaama
            sigma = sigmoid(sigma_raw)

            #finding z matrix
            epsilon = (np.repeat(np.asmatrix(np.random.multivariate_normal(np.zeros((no_latent)) , np.eye(no_latent))), batch_size,axis = 0))
            z = np.multiply( epsilon, np.exp(0.5*sigma))+mu
            z_biased = np.asmatrix(np.hstack((np.ones((batch_size,1)),z)))

            #finding y matrix
            y_raw = z_biased * beta
            y_predicted = sigmoid(y_raw)

            ##finding the cost
            cost = squared_error(y_train , y_predicted) + 0.5*(np.sum(np.square(mu)) + np.sum(np.exp(sigma)) - np.sum(sigma) - no_latent*batch_size)
            costs.append(cost)
            #finding gradient w.r.t beta
            delta = np.multiply((y_predicted - y_train), deriv_sigmoid(y_raw))
            d_beta = z_biased.T * delta

            temp_beta = beta[1:,:]

            #finding gradient w.r.t alpha
            ss_alpha = np.multiply((delta * temp_beta.T),deriv_sigmoid(mu_raw))
            d_alpha = x_train.T * (ss_alpha + np.multiply(mu , deriv_sigmoid(mu_raw)))

            #finding gradient w.r.t gaama
            ss_beta = np.multiply((delta * temp_beta.T),deriv_sigmoid(sigma_raw))
            d_gaama = x_train.T * 0.5*( - 1* np.multiply(epsilon , np.multiply(ss_beta , np.exp(0.5 * sigma))) + np.multiply(np.exp(sigma) , deriv_sigmoid(sigma_raw)) - deriv_sigmoid(sigma_raw))

            #updating the weights
            beta = beta - lr * d_beta
            alpha = alpha - lr*d_alpha
            gaama = gaama - lr*d_gaama
#         print(np.max(alpha) , np.max(beta) , np.min(alpha) , np.min(beta))
            print("\nEpoch: " + str(epoch+1) + "   batch : " + str(i+1) + "   cost : " + str(cost))
    return beta , costs


# In[80]:


#prediction

def net_generate(number , weights , no_latent):
    generated = []
    for i in range(0,number):
        x = np.asmatrix(np.random.multivariate_normal(np.zeros((no_latent)) , np.eye(no_latent)))
        x = np.asmatrix(np.hstack(([[1]],x)))
        gen = sigmoid(x * weights)
        gen_image = np.round(gen.reshape(28,28))
        generated.append(gen_image)
    return generated


# # Generating training data

# In[89]:


# x_train = load_flattened_images("/home/snehith/Documents/machine learning/datasets/mnist/mnistasjpg/testSet/")
# np.save("train_set_vae.npy" , x_train)
x_train = np.asmatrix(np.load("train_set_vae.npy"))
print(x_train.shape)


# # Training the Model

# In[35]:


# alpha , beta , losses = net_fit(x_Train , y_train_and , hidden_nodes = hidden_nodes , epochs = epochs ,lr = lr)
tic = time.time()
weights , lossses = net_fit_vae(x_train , x_train , no_latent = no_latent , epochs = epochs ,lr = lr)
print("time taken: "+ str(time.time() - tic) + "sec")
# print("\nalpha:\n",alpha ,"\nbeta:\n", beta,"\n" ,"\nloss:\n", losses[epochs-1])
np.save("weights_vae.npy" , weights)
# np.save("beta_weights_vae.npy" , beta)


# ### Generating new images

# In[81]:


#Generating new images
result = net_generate(16 , weights , no_latent)
plt.figure(figsize = (15,15))
for i in range(0,16):
    plt.subplot(4,4,i+1)
    plt.imshow(result[i] , cmap = 'gray')


# In[37]:


#ploting the cost vs epochs
plt.plot(np.arange(len(lossses)),lossses)
plt.title("training samples: " + str(x_train.shape[0]))
plt.xlabel("epochs")
plt.ylabel("cost")
plt.show()

