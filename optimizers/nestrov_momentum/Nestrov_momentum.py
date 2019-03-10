#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = load_iris()
print(iris.DESCR)


# ## Loading Data 

# In[2]:


x = iris.data

label = iris.target

y = np.zeros(label.shape + (3,))
y[np.arange(label.shape[0]),label] = 1

print x.shape, y.shape


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print x_train.shape, x_test.shape, y_train.shape, y_test.shape


# ## Activation Functions

# In[3]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    x[x < 0] = 0
    return x

def tanh(x):
    return np.tanh(x)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def deriv(x, activation = 'relu'):
    if(activation == 'relu'):
        x[x > 0] = 1
        x[x < 0] = 0
        return x


# ## Function to Initialize Weights

# In[4]:


def xavier_initializer(fan_in,fan_out):
    return np.random.normal(0,np.sqrt(2*1.0/(fan_in+fan_out)),(fan_out,fan_in+1))


# ## Compute the Output shapes of each layer

# In[5]:


def get_model(feed_dict):
    feed_dict['input_shape'] = feed_dict['train_input'].shape[1:]
    inp_shape = feed_dict['input_shape']
    feed_dict['output'] = []
    layers = feed_dict['layers']
    
    for i in range(len(layers)):
        output_shape = (layers[i]['nodes'],1)
        out_dict = {'layer_number': i , 'type': 'fc', 'output_shape': output_shape}
        feed_dict['output'].append(out_dict)
        inp_shape = output_shape
    return feed_dict


# ## Fully Connected layer

# In[6]:


def fully_connected(inp, weights,  nodes, activation):
    inp = np.asarray(inp).reshape(len(inp),1)
    inp = np.vstack((np.array(inp),1))
    #initiazing weights
#     weights = np.asmatrix(np.random.rand(nodes, len(inp)))
    output_raw = np.matmul(weights, inp)
    #normalizing the output to ensure no overflow in exp
#     print np.max(output_raw)
    output_raw = output_raw 
    #applying activation function
    if(activation == 'sigmoid'):
        output = sigmoid(output_raw)
    elif(activation == 'relu'):
        output = relu(output_raw)
    elif(activation == 'tanh'):
        output = tanh(output_raw)
    elif(activation == 'softmax'):
        output = softmax(output_raw)
    else:
        output = output_raw
    #making the output vector as column matrix
    if(output.shape[0] == 1):
        output = np.moveaxis(output, 0,1)
        output_raw = np.moveaxis(output_raw, 0,1)
    return output, output_raw


# ## Defining the model

# In[7]:


feed_dict = {}
feed_dict['train_input'] = x_train
feed_dict['train_label'] = y_train
feed_dict['test_input'] = x_test
feed_dict['test_label'] = y_test
feed_dict['learning_rate_alpha'] = 0.2
feed_dict['learning_rate_epsilon'] = 1e-3
feed_dict['epochs'] = 1000
feed_dict['batch_size'] = 5
feed_dict['layers'] = [{'type': 'fc',  'nodes': 10, 'activation' : 'relu'},
                       {'type': 'fc',  'nodes': 10, 'activation' : 'relu'},
                       {'type': 'fc',  'nodes': 3, 'activation' : 'softmax'}]


# ## Computing the outputs and initializng the weights

# In[8]:


feed_dict = get_model(feed_dict)

print 'output shapes:'
for i in range(len(feed_dict['layers'])):
    print feed_dict['layers'][i]['type']+str(i) , ': ', feed_dict['output'][i]['output_shape']

print("\n")

feed_dict['output'][0]['weights'] = xavier_initializer(feed_dict['input_shape'][0], feed_dict['layers'][0]['nodes'])
feed_dict['output'][1]['weights'] = xavier_initializer(feed_dict['output'][0]['output_shape'][0], feed_dict['layers'][1]['nodes'])
feed_dict['output'][2]['weights'] = xavier_initializer(feed_dict['output'][1]['output_shape'][0], feed_dict['layers'][2]['nodes'])

print 'weight matrices shapes (with biases):'
print feed_dict['layers'][0]['type']+str(0),feed_dict['output'][0]['weights'].shape
print feed_dict['layers'][1]['type']+str(1),feed_dict['output'][1]['weights'].shape
print feed_dict['layers'][2]['type']+str(2),feed_dict['output'][2]['weights'].shape


# In[9]:


epochs = feed_dict['epochs']
no_samples = len(x_train)
batch_size = feed_dict['batch_size']
no_batches = no_samples/batch_size


# In[10]:


tpp = [np.random.rand(3,3), np.random.rand(4,5,6), np.random.rand(5,8,7,6,5)]
print tpp[1].shape


# In[11]:


layers = feed_dict['layers']
train_losses = []
feed_dict['vel'] = [np.zeros(feed_dict['output'][0]['weights'].shape), np.zeros(feed_dict['output'][1]['weights'].shape), np.zeros(feed_dict['output'][2]['weights'].shape)]
for epoch in range(epochs):
    cost_per_epoch = 0
    #shuffling the data
    s = np.arange(feed_dict['train_input'].shape[0])
    np.random.shuffle(s)
    feed_dict['train_input'] = feed_dict['train_input'][s]
    feed_dict['train_label'] = feed_dict['train_label'][s]
    for batch in range(no_batches):
        # weight matrices for sum of updates of batch
        weights_fc_0 = np.zeros(feed_dict['output'][0]['weights'].shape)
        weights_fc_1 = np.zeros(feed_dict['output'][1]['weights'].shape)
        weights_fc_2 = np.zeros(feed_dict['output'][2]['weights'].shape)
        for i in range(batch_size):
            #feeding forward
            feed_dict['output'][0]['output'], feed_dict['output'][0]['output_raw'] = fully_connected(feed_dict['train_input'][i + batch * batch_size], weights = feed_dict['output'][0]['weights'], nodes = layers[0]['nodes'], activation = layers[0]['activation'])
            feed_dict['output'][1]['output'], feed_dict['output'][1]['output_raw'] = fully_connected(feed_dict['output'][0]['output'], weights = feed_dict['output'][1]['weights'], nodes = layers[1]['nodes'], activation = layers[1]['activation'])
            feed_dict['output'][2]['output'], feed_dict['output'][2]['output_raw'] = fully_connected(feed_dict['output'][1]['output'], weights = feed_dict['output'][2]['weights'], nodes = layers[2]['nodes'], activation = layers[2]['activation'])
            
            #cost calculation
            cost_per_epoch = cost_per_epoch - np.log(feed_dict['output'][2]['output'][np.argmax(feed_dict['train_label'][i + batch * batch_size])])
            
            #calculating the gradients
            feed_dict['output'][2]['semi_update'] = feed_dict['output'][2]['output'] - feed_dict['train_label'][i + batch * batch_size].reshape(-1,1)
            feed_dict['output'][2]['update'] = np.matmul(feed_dict['output'][2]['semi_update'] , np.transpose(np.vstack((feed_dict['output'][1]['output'],1))))
            
            temp = feed_dict['output'][2]['weights'][:,0:feed_dict['output'][2]['weights'].shape[1]-1]
            feed_dict['output'][1]['semi_update'] = np.matmul(np.transpose(temp), feed_dict['output'][2]['semi_update']) * deriv(feed_dict['output'][1]['output_raw'])
            feed_dict['output'][1]['update'] = np.matmul(feed_dict['output'][1]['semi_update'] , np.transpose(np.vstack((feed_dict['output'][0]['output'],1))))
            
            temp = feed_dict['output'][1]['weights'][:,0:feed_dict['output'][1]['weights'].shape[1]-1]
            feed_dict['output'][0]['semi_update'] = np.matmul(np.transpose(temp), feed_dict['output'][1]['semi_update']) * deriv(feed_dict['output'][0]['output_raw'])
            feed_dict['output'][0]['update'] = np.matmul(feed_dict['output'][0]['semi_update'],np.transpose(np.vstack((np.expand_dims(feed_dict['train_input'][i + batch * batch_size],axis = 1),1))))
            
            weights_fc_0 += feed_dict['output'][0]['update']
            weights_fc_1 += feed_dict['output'][1]['update']
            weights_fc_2 += feed_dict['output'][2]['update']
            
        #updating the gradient after each batch
        feed_dict['vel'][0] = (feed_dict['learning_rate_alpha']**2) *feed_dict['vel'][0] - (feed_dict['learning_rate_alpha']+1)*feed_dict['learning_rate_epsilon'] * weights_fc_0
        feed_dict['vel'][1] = (feed_dict['learning_rate_alpha']**2) *feed_dict['vel'][1] - (feed_dict['learning_rate_alpha']+1)*feed_dict['learning_rate_epsilon'] * weights_fc_1
        feed_dict['vel'][2] = (feed_dict['learning_rate_alpha']**2) *feed_dict['vel'][2] - (feed_dict['learning_rate_alpha']+1)*feed_dict['learning_rate_epsilon'] * weights_fc_2 
        feed_dict['output'][0]['weights'] += feed_dict['vel'][0]
        feed_dict['output'][1]['weights'] += feed_dict['vel'][1]
        feed_dict['output'][2]['weights'] += feed_dict['vel'][2]
        
    #printing the Average Loss after each epoch
    if((epoch+1)%50 == 0):
        print("Epoch: " + str(epoch+1) + " Loss: " + str(cost_per_epoch/no_samples))
    train_losses.append(cost_per_epoch/no_samples)


# ## Saving the Model

# In[12]:


np.save("nestrov_momentum.npy", feed_dict)


# ## Plotting the training loss

# In[13]:


train_losses = np.array(train_losses)
plt.plot(train_losses)
plt.title('LOSS Vs Epochs')
plt.ylabel('Cross Entrphoy Loss')
plt.xlabel('Number of Epochs')
plt.show()


# ## Prediction on one sample

# In[14]:


feed_dict['output'][0]['output'], feed_dict['output'][0]['output_raw'] = fully_connected(feed_dict['train_input'][0], weights = feed_dict['output'][0]['weights'], nodes = layers[0]['nodes'], activation = layers[0]['activation'])
feed_dict['output'][1]['output'], feed_dict['output'][1]['output_raw'] = fully_connected(feed_dict['output'][0]['output'], weights = feed_dict['output'][1]['weights'], nodes = layers[1]['nodes'], activation = layers[1]['activation'])
feed_dict['output'][2]['output'], feed_dict['output'][2]['output_raw'] = fully_connected(feed_dict['output'][1]['output'], weights = feed_dict['output'][2]['weights'], nodes = layers[2]['nodes'], activation = layers[2]['activation'])

print 'output of softmax for one sample:'
print feed_dict['output'][2]['output']

print '\nGround Truth of the same sample above:'
print feed_dict['train_label'][0]


# ## Predicting on Test Data

# In[15]:


test_predicted = []
gt = []
# print(np.argmax(out))
for i in range(feed_dict['test_input'].shape[0]):
    feed_dict['output'][0]['output'], feed_dict['output'][0]['output_raw'] = fully_connected(feed_dict['test_input'][i], weights = feed_dict['output'][0]['weights'], nodes = layers[0]['nodes'], activation = layers[0]['activation'])
    feed_dict['output'][1]['output'], feed_dict['output'][1]['output_raw'] = fully_connected(feed_dict['output'][0]['output'], weights = feed_dict['output'][1]['weights'], nodes = layers[1]['nodes'], activation = layers[1]['activation'])
    feed_dict['output'][2]['output'], feed_dict['output'][2]['output_raw'] = fully_connected(feed_dict['output'][1]['output'], weights = feed_dict['output'][2]['weights'], nodes = layers[2]['nodes'], activation = layers[2]['activation'])
    
    test_predicted.append(np.argmax(feed_dict['output'][2]['output']))
    gt.append(np.argmax(feed_dict['test_label'][i]))


# ## Outputs and the respective Groucd Truths

# In[16]:


print 'predicted: ',test_predicted
print 'Actual   : ', gt


# ## Accuracy on the Test Dataset

# In[17]:


a = np.array(test_predicted) - np.array(gt)
test_accuracy = (len(a) - np.count_nonzero(a))/float(len(a))

print 'accuracy: ', str(test_accuracy)

