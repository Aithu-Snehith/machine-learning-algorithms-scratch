#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io.wavfile import read
import numpy as np 


# In[2]:


def perform_dtw(x_path,y_path):
    a = read(x_path)
    b = read(y_path)

    x = np.array(a[1],dtype="float").reshape(-1,1)
    y = np.array(b[1],dtype="float").reshape(-1,1)
#     print(x.shape, y.shape)

    distance_matrix = np.square(np.abs(np.tile(x, (1,len(y))) - np.transpose(np.tile(y, (1,len(x))))))
    
    #DTW ALGORITHM

    #INITIATING THE FIRST ROW AND COLUMN
    cost = np.zeros((len(x), len(y)))
    cost[0][0] = distance_matrix[0][0]
    for i in range(1,len(x)):
        cost[i][0]=distance_matrix[i][0] + cost[i-1][0]
    for j in range(1,len(y)):
        cost[0][j]=distance_matrix[0][j] + cost[0][j-1]

    #actual DP step
    for i in range(1,len(x)):
        for j in range(1,len(y)):
            cost[i][j]=distance_matrix[i][j] + min(cost[i-1][j],cost[i][j-1],cost[i-1][j-1])
            
    return cost[len(x)-1][len(y)-1]


# In[5]:


dist_0_3 = perform_dtw("digits_speech/zero/0_jackson_22.wav","digits_speech/three/3_theo_23.wav" )
dist_0_0 = perform_dtw("digits_speech/zero/0_jackson_22.wav","digits_speech/zero/0_jackson_0.wav" )

print(dist_0_3, dist_0_0, dist_0_0<dist_0_3)

