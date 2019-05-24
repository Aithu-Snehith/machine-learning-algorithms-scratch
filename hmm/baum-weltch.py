#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import librosa as ls
import pyprind
# Couldn't save it as a pdf file as pyprind is used for progressbar

# import pyprind

import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()
# In[2]:


zero_mfcc = []
for file in os.listdir('digits_speech/zero/'):
    if file.endswith(".wav"):
        path = os.path.join('digits_speech/zero/',file)
        signal,sr = ls.load(path ,sr=None,  duration=0.21)
        mfccs = ls.feature.mfcc(y=signal, sr=sr, n_mfcc=13, hop_length=int(0.015*sr), n_fft=int(0.025*sr))
        zero_mfcc.append(mfccs.T)
        
seven_mfcc = []
for file in os.listdir('digits_speech/seven/'):
    if file.endswith(".wav"):
        path = os.path.join('digits_speech/seven/',file)
        signal,sr = ls.load(path ,sr=None,  duration=0.21)
        mfccs = ls.feature.mfcc(y=signal, sr=sr, n_mfcc=13, hop_length=int(0.015*sr), n_fft=int(0.025*sr))
#         print(mfccs.T.shape)
        seven_mfcc.append(mfccs.T)
        
zero_mfcc = np.array(zero_mfcc)
seven_mfcc = np.array(seven_mfcc)
print(zero_mfcc.shape, seven_mfcc.shape)


# In[3]:


temp = zero_mfcc - np.mean(zero_mfcc, axis = 0)
input1 = temp/np.std(zero_mfcc, axis = 0)

temp = seven_mfcc - np.mean(seven_mfcc, axis = 0)
input2 = temp/np.std(seven_mfcc, axis = 0)

in1_train,in1_test = train_test_split(input1, test_size=0.2)
in2_train,in2_test = train_test_split(input2, test_size=0.2)


print(in1_test.shape, in2_test.shape)


# In[4]:


n_states = 5
m_gmm = 3
vect_len = 13
epochs = 20


# In[5]:


def pdf(x, state, weights):
    wt = weights['w'][state]
    mean = weights['mu'][state]
    var = weights['co_var'][state]
    
    pdf = 0
    
    for i in range(m_gmm):
        a = (np.sqrt((np.linalg.det(var[i]) * (2*np.pi)**len(x))))
        b = np.exp((-np.matmul(np.matmul(np.transpose(x-mean[i]) , np.matrix(var[i]).I ), (x-mean[i]))/2))
        pdf = pdf + float(b/a)
    return pdf


# In[6]:


def forward_pass(x, weights):
    alpha = np.zeros((x.shape[0], n_states))
    
    for j in range(alpha.shape[1]):
        alpha[0][j] = weights['phi'][j] * pdf(x[0],j, weights)
        
    for i in range(1,alpha.shape[0]):
        for j in range(alpha.shape[1]):
#             print(A[:,j].shape, alpha[i-1].shape)
            alpha[i][j] = np.dot(weights['A'][:,j].reshape(1,-1), alpha[i-1].reshape(-1,1)) * pdf(x[i], j, weights)
    
    return alpha

def backward_pass(x, weights):
    beta = np.zeros((x.shape[0], n_states))
    
    for j in range(beta.shape[1]):
        beta[x.shape[0]-1][j] = 1
        
    for t in reversed(range(0,beta.shape[0]-1)):
        for i in range(beta.shape[1]):
            temp = 0
            for j in range(n_states):
                temp += beta[t+1][j]*weights['A'][i][j]*pdf(x[t+1], j, weights)
            beta[t][i] = temp
    
    return beta


# In[7]:


def cal_gamma(alpha,beta, weights):
    gamma = []
    for i in range(0,len(alpha)):
        gamma.append((alpha[i]*beta[i])/np.sum(alpha[i]*beta[i]))
    gamma = np.asarray(gamma)
    
    return gamma

def cal_zeta(x, alpha, beta, weights):
    zeta = []
    for t in range(alpha.shape[0]-1):
        temp = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                temp[i][j] = alpha[t][i]*weights['A'][i][j]*pdf(x[t+1], j, weights)*beta[t+1][j]
        zeta.append(temp/np.sum(temp))
    zeta = np.array(zeta)
    return zeta


# In[8]:


def gmm_gamma(x, gamma, weights):
    gmm_g = np.zeros((x.shape[0], n_states, m_gmm))
    for t in range(x.shape[0]):
        for i in range(n_states):
            temp = []
            for m in range(m_gmm):
                a = (np.sqrt((np.linalg.det(weights['co_var'][i][m]) * (2*np.pi)**len(x[t]))))
                b = np.exp((-np.matmul(np.matmul(np.transpose(x[t]-weights['mu'][i][m]) , np.matrix(weights['co_var'][i][m]).I ), (x[t]-weights['mu'][i][m]))/2))
                temp.append(a/b)
            gmm_g[t][i] = gamma[t][i] * np.squeeze(np.array(temp/np.sum(temp)))
    return gmm_g


# In[13]:


def new_params(x, gamma, zeta, gmm_g, weights):
    new_phi = gamma[0]
    new_A = np.sum(zeta, axis = 0)/np.sum(gamma, axis = 0)
    
    new_w = np.sum(gmm_g, axis = 0)/np.sum(np.sum(gmm_g, axis = 0), axis = 0)
    
    new_mu = np.sum(np.tile(gmm_g.reshape(gmm_g.shape + (1,)), (1,1,1,x.shape[1])) * np.tile(x.reshape(x.shape[0], 1,1, x.shape[1]), (1,gmm_g.shape[1], gmm_g.shape[2],1)), axis = 0)/np.tile(np.sum(gmm_g, axis = 0).reshape(gmm_g.shape[1:] + (1,)), (1,1,x.shape[1]))
    
    new_co_var = np.zeros((n_states, m_gmm, x.shape[1], x.shape[1]))
    
    for j in range(n_states):
        for m in range(m_gmm):
            temp = []
            for t in range(x.shape[0]):
                te = gmm_g[t][j][m] * np.matmul((x[t] - weights['mu'][j][m]).reshape(-1,1), (x[t] - weights['mu'][j][m]).reshape(1,-1))
                temp.append(te)
            new_co_var[j][m] = np.sum(np.array(temp), axis = 0)/np.sum(gmm_g[:,j,m], axis = 0)
    return new_phi, new_A, new_w, new_mu, new_co_var


# In[14]:


hmm1_weights = {}
hmm1_weights['phi'] = np.ones(n_states)/n_states
print(hmm1_weights['phi'])

hmm1_weights['A'] = np.ones((n_states, n_states))/(n_states)
print(hmm1_weights['A'])

w = np.random.uniform(size = (n_states,m_gmm))
hmm1_weights['w'] = np.transpose(np.transpose(w)/np.sum(w, axis = 1))
print(hmm1_weights['w'])

hmm1_weights['mu'] = np.random.rand(n_states, m_gmm, vect_len)
print(hmm1_weights['mu'].shape)

co_var = [np.eye(vect_len, vect_len) for _ in range(n_states*m_gmm)]
hmm1_weights['co_var'] = np.array(co_var).reshape(n_states, m_gmm, vect_len, vect_len)
print(hmm1_weights['co_var'].shape)
# print(co_var[4,2,:,:])


# In[15]:


print('training HMM for number Zero')

for epoch in range(epochs):
    n_phi = np.zeros_like(hmm1_weights['phi'])
    n_A = np.zeros_like(hmm1_weights['A'])
    n_w = np.zeros_like(hmm1_weights['w'])
    n_mu = np.zeros_like(hmm1_weights['mu'])
    n_co_var = np.zeros_like(hmm1_weights['co_var'])
    
    print('epoch:', epoch)
    bar = pyprind.ProgBar(in1_train.shape[0])
    
    for i in range(in1_train.shape[0]):
        bar.update()
        inp = in1_train[i]
        alpha = forward_pass(inp, hmm1_weights)
        beta = backward_pass(inp, hmm1_weights)
        gamma = cal_gamma(alpha, beta, hmm1_weights)
        zeta = cal_zeta(inp, alpha, beta, hmm1_weights)
        gmm_g = gmm_gamma(inp, gamma, hmm1_weights)
        new_phi, new_A, new_w, new_mu, new_co_var = new_params(inp, gamma, zeta, gmm_g, hmm1_weights)
        
        n_phi = n_phi + new_phi
        n_A = n_A + new_A
        n_w = n_w + new_w
        n_mu = n_mu + new_mu
        n_co_var = n_co_var + new_co_var
#         print('sample:', i)
    
    hmm1_weights['phi'] = n_phi/inp.shape[0]
    hmm1_weights['A'] = n_A/inp.shape[0]
    hmm1_weights['mu'] = n_mu/inp.shape[0]
    hmm1_weights['co_var'] = n_co_var/inp.shape[0]


# In[16]:


hmm2_weights = {}
hmm2_weights['phi'] = np.ones(n_states)/n_states
print(hmm2_weights['phi'])

hmm2_weights['A'] = np.ones((n_states, n_states))/(n_states)
print(hmm2_weights['A'])

w = np.random.uniform(size = (n_states,m_gmm))
hmm2_weights['w'] = np.transpose(np.transpose(w)/np.sum(w, axis = 1))
print(hmm2_weights['w'])

hmm2_weights['mu'] = np.random.rand(n_states, m_gmm, vect_len)
print(hmm2_weights['mu'].shape)

co_var = [np.eye(vect_len, vect_len) for _ in range(n_states*m_gmm)]
hmm2_weights['co_var'] = np.array(co_var).reshape(n_states, m_gmm, vect_len, vect_len)
print(hmm2_weights['co_var'].shape)
# print(co_var[4,2,:,:])


# In[17]:


print('training HMM for number Seven')


for epoch in range(epochs):
    n_phi = np.zeros_like(hmm2_weights['phi'])
    n_A = np.zeros_like(hmm2_weights['A'])
    n_w = np.zeros_like(hmm2_weights['w'])
    n_mu = np.zeros_like(hmm2_weights['mu'])
    n_co_var = np.zeros_like(hmm2_weights['co_var'])
    
    print('epoch:', epoch)
    bar = pyprind.ProgBar(in2_train.shape[0])
    
    
    for i in range(in2_train.shape[0]):
        bar.update()
        inp = in2_train[i]
        alpha = forward_pass(inp, hmm2_weights)
        beta = backward_pass(inp, hmm2_weights)
        gamma = cal_gamma(alpha, beta, hmm2_weights)
        zeta = cal_zeta(inp, alpha, beta, hmm2_weights)
        gmm_g = gmm_gamma(inp, gamma, hmm2_weights)
        new_phi, new_A, new_w, new_mu, new_co_var = new_params(inp, gamma, zeta, gmm_g, hmm2_weights)
        
        n_phi = n_phi + new_phi
        n_A = n_A + new_A
        n_w = n_w + new_w
        n_mu = n_mu + new_mu
        n_co_var = n_co_var + new_co_var
#         print('epoch:', epoch, 'sample:', i)
    
    hmm2_weights['phi'] = n_phi/inp.shape[0]
    hmm2_weights['A'] = n_A/inp.shape[0]
    hmm2_weights['mu'] = n_mu/inp.shape[0]
    hmm2_weights['co_var'] = n_co_var/inp.shape[0]


# In[25]:


px_in1 = []
px_in2 = []

for l in range(input1.shape[0]):
    px_in1.append(np.sum(forward_pass(input1[l], hmm1_weights)))
    px_in2.append(np.sum(forward_pass(input1[l], hmm2_weights)))

count = 0
for i in range(input1.shape[0]):
    if(px_in1[i]>=px_in2[i]):
        count += 1
print(count/float(input1.shape[0]))
# print(px_in1)
# print()
# print(px_in2)

px_in1 = []
px_in2 = []

for l in range(input2.shape[0]):
    px_in1.append(np.sum(forward_pass(input2[l], hmm1_weights)))
    px_in2.append(np.sum(forward_pass(input2[l], hmm2_weights)))

count = 0
for i in range(input2.shape[0]):
    if(px_in2[i]>=px_in1[i]):
        count += 1
print(count/float(input2.shape[0]))

# print(px_in1)
# print()
# print(px_in2)

