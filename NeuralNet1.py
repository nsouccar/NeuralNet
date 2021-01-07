#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sympy import *


# In[23]:


training_data = [[[1.0,1.0,1.0],[1.0,1.0]], [[0.0,1.0,1.0],[0.0,1.0]]]

def sigmoid(x):
    z = 1/(1 + np.exp(-x)) 
    return z

# 3, 4, 2
    
class Neural_Net_One():
    layer_location = 1
    current_inputs = []
    
    def __init__(self, sizes): 
        self.layers = sizes
        self.weights = []
        self.biases  = []
        for i in range(0, (len(sizes) - 1)):
            layer_weight_size = sizes[i] 
            self.weights.append(np.random.randn(sizes[i],sizes[i+1]))
            self.biases.append(np.random.randn((sizes[i+1])))
                     
        
    def feedforward(self, data):
        current_inputs = data
        activations    = []
        activations.append(data)
        for layer in range(1, len(self.layers)):
            outputs = sigmoid((np.dot(current_inputs, self.weights[(layer - 1)]) - self.biases[(layer-1)]))
            current_inputs = outputs
            activations.append(current_inputs.astype(float))

        return activations
     
    def train(self, training_data, epochs):
        
            
        for epoch in range(0, epochs):
          
            for data in training_data:
                activations = self.feedforward(data[0])
                gradients = self.gradient_descent(activations, data[1])
                self.backprop(gradients[0], gradients[1])

            
    def backprop(self, gradients_w, gradients_b):
        for i in range(0, len(self.biases)):
            self.biases[i] += np.flip(gradients_b)[i]
        for j in range(0, len(self.biases)):
            self.weights[j] += np.flip(gradients_w)[j]

            
        
    def gradient_descent(self, activations, y):
        gradients_w = []
        gradients_b = []
        cost = np.square(activations[(len(self.layers) - 1)] - [y])
        d_cost_outputs = 2 * (activations[(len(self.layers) - 1)] - [y])
        for layer in range((len(self.layers) - 1), 0, -1):
            if(layer < (len(self.layers) - 1) ):
                for node in range(0, self.layers[layer]):
                        pathways = self.weights[layer][node]
                        for weight in range(0, len(self.weights[(layer-1)][node])):
                            gradients_w.append(activations[layer - 1][node] * (self.weights[layer][node][1] + self.weights[layer][node][0]))
                         
                        gradients_b.append(-1 * (self.weights[layer][node][1] + self.weights[layer][node][0]))
                        
            else:
            
                for node in range(0, self.layers[layer]):
                    for weight in range(0, len(self.weights[layer-1][:, node])):
                        gradient_w = (activations[layer-1][weight]) * (d_cost_outputs[0][node])
                        gradients_w.append(gradient_w)
                        
                    gradient_b = -1 * d_cost_outputs[0][node]
                    gradients_b.append(gradient_b)
                    
        return [gradients_w, gradients_b]
                    
    
                    
                    




        
        
    
    


    


# In[30]:


net = Neural_Net_One([3, 3, 2])
net.train(training_data, 200)






# In[32]:


print(net.feedforward([0.0, 0.0, 0.0]))


# In[ ]:




