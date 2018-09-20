
# coding: utf-8

# # Importing the packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # Parameters

# In[2]:


input_dim = 2
hidden_nodes = 10
lr = 1e-3
epochs = 10000
training_samples = 1000
out_dim = 1


# # Declaring functions

# In[3]:


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


# In[4]:


## fitting the model

def net_fit(x_train , y_train , epochs = 100 , hidden_nodes = 2 , lr = 1e-3):
    input_dim = x_train.shape[1]
    training_samples = x_train.shape[0]
    output_dim = y_train.shape[1]
    costs = []
    x_train = np.hstack((np.ones((training_samples , 1)), x_train))
    #initializig the parameters
    alpha = np.asmatrix(np.random.rand(input_dim + 1 , hidden_nodes))
    beta = np.asmatrix(np.random.rand(hidden_nodes+1 , out_dim))
    
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
        cost = squared_error(y_train , y_predicted)
        costs.append(cost)
        
        #finding gradient w.r.t beta
        delta = np.multiply((y_predicted - y_train), deriv_sigmoid(y_raw))
        d_beta = np.zeros(beta.shape)
        for i in range(0,d_beta.shape[0]):
            for j in range(0,d_beta.shape[1]):
                d_beta[i,j] = np.sum(np.multiply(delta[:,j],z_biased[:,i]))
        
        temp_beta = beta[1:,:]
        
        #finding gradient w.r.t alpha
        ss = np.multiply((delta * temp_beta.T),deriv_sigmoid(z_raw))
        d_alpha = np.zeros(alpha.shape)
        for i in range(0,d_alpha.shape[0]):
            for j in range(0,d_alpha.shape[1]):
                d_alpha[i,j] = np.sum(np.multiply(ss[:,j],x_train[:,i]))
        
        #updating the weights
        beta = beta - lr * d_beta
        alpha = alpha - lr*d_alpha
#         print("\n\nEpoch: " + str(epoch+1) + "   cost : " + str(cost))
    return alpha , beta , costs


# In[5]:


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

# In[6]:


variance = 0.1
n = int(training_samples/2**input_dim)
x_00 = np.hstack((np.random.normal(0,variance,(n,1)) , np.random.normal(0,variance,(n,1)) ))
x_01 = np.hstack((np.random.normal(0,variance,(n,1)) , np.random.normal(1,variance,(n,1)) ))
x_10 = np.hstack((np.random.normal(1,variance,(n,1)) , np.random.normal(0,variance,(n,1)) ))
x_11 = np.hstack((np.random.normal(1,variance,(n,1)) , np.random.normal(1,variance,(n,1)) ))
x_Train = np.asmatrix(np.concatenate((x_00,x_01,x_10,x_11)))
print(x_Train.shape)
# y_train = np.asmatrix(np.append(np.zeros((3*n,1)),np.ones((n,1)))).T
y_train_and = np.asmatrix(np.append(np.zeros((3*n,1)),np.ones((n,1)))).T
y_train_or = np.asmatrix(np.append(np.zeros((n,1)),np.ones((3*n,1)))).T
y_train_xor = np.asmatrix(np.concatenate((np.zeros((n,1)),np.ones((2*n,1)),(np.zeros((n,1))))))

### Trying to solve classification problem , so commenting the below noise
# y_train_and += np.random.normal(1,variance,(4*n,1))
# y_train_or += np.random.normal(1,variance,(4*n,1))
# y_train_xor += np.random.normal(1,variance,(4*n,1))
print(y_train_and.shape , y_train_or.shape , y_train_xor.shape)


# # Training the Model

# ## And Gate

# ### training

# In[7]:


alpha , beta , losses = net_fit(x_Train , y_train_and , hidden_nodes = hidden_nodes , epochs = epochs ,lr = lr)
print("\nalpha:\n",alpha ,"\nbeta:\n", beta,"\n" ,"\nloss:\n", losses[9999])


# ### Predicting

# In[8]:


#testing samples
x_test = np.matrix([[0,0],[0,1],[1,0],[1,1]])
print(x_test.shape)
#predicting the output
res = net_predict(x_test , alpha , beta)
print(res)
# print(losses.shape)
#ploting the cost vs epochs
plt.plot(np.arange(epochs),losses)
plt.title("training samples: " + str(training_samples))
plt.xlabel("epochs")
plt.ylabel("cost")
plt.show()


# ## Xor Gate

# ### training

# In[9]:


alpha , beta , loss = net_fit(x_Train , y_train_xor , hidden_nodes = hidden_nodes , epochs = epochs ,lr = lr)
print("\nalpha:\n",alpha ,"\nbeta:\n", beta,"\n","\nloss:\n", losses[9999])


# ### Predicting

# In[10]:


#testing samples
x_test = np.matrix([[0,0],[0,1],[1,0],[1,1]])
print(x_test.shape)
#predicting the output
res = net_predict(x_test , alpha , beta)
print(res)
# print(losses.shape)
#ploting the cost vs epochs
plt.plot(np.arange(epochs),losses)
plt.title("training samples: " + str(training_samples))
plt.xlabel("epochs")
plt.ylabel("cost")
plt.show()


# ## Or Gate

# ### training

# In[11]:


alpha , beta , loss = net_fit(x_Train , y_train_or , hidden_nodes = hidden_nodes , epochs = epochs ,lr = lr)
print("\nalpha:\n",alpha ,"\nbeta:\n", beta,"\n","\nloss:\n", losses[9999])


# ### Predicting

# In[12]:


#testing samples
x_test = np.matrix([[0,0],[0,1],[1,0],[1,1]])
print(x_test.shape)
#predicting the output
res = net_predict(x_test , alpha , beta)
print(res)
# print(losses.shape)
#ploting the cost vs epochs
plt.plot(np.arange(epochs),losses)
plt.title("training samples: " + str(training_samples))
plt.xlabel("epochs")
plt.ylabel("cost")
plt.show()

