# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:41:31 2018

@author: roel
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

########################################################################
## Data Preparation ####################################################
########################################################################

# Read the data set
sc_data = pd.read_csv('crx.data', header=None, decimal=".", na_values='?')

# Drop all rows that have NA's.
sc_data = sc_data.dropna(axis=0,how='any')

# The last column of the data set is the one we want to predict
sc_target = sc_data.iloc[:,-1]

# Replace the '+' and '-' labels with 1 and 0 integers
sc_target = sc_target.replace('+',1)
sc_target = sc_target.replace('-',0)

# You have to OneHotEncode it for Tensorflow to function properly
sc_target = OneHotEncoder(sparse=False).fit_transform(np.reshape(sc_target, [-1, 1]))

# Remove the last column (the target) of the data set and convert it to a numpy matrix
sc_data = sc_data.iloc[:,0:15]
sc_data = pd.get_dummies(sc_data)
sc_data = sc_data.as_matrix()

# Split the data set into training and testing sets
data_train, data_test, label_train, label_test = train_test_split(sc_data, sc_target)

########################################################################
## Graph Construction ##################################################
########################################################################

# Model hyperparameters 
g_learning_rate = 0.0005
g_hidden_layer_sizes = [30,120,30]

# We set placeholders for our training data and target.
g_data = tf.placeholder(tf.float32, [None, np.shape(sc_data)[1]])
g_label= tf.placeholder(tf.float32, [None, np.shape(sc_target)[1]])

# The amount of features determines the amount of nodes in the input layer
# The amount of possible values in your target determine the amount of nodes in your output layer.
g_num_features = int(g_data.get_shape()[1])
g_num_classes = int(g_label.get_shape()[1])

# Once we know the number of features and target values, we can construct the neural network.
# It consists of the input layer, the three hidden layers and the output layer.
g_layer_sizes = []
g_layer_sizes.append(g_num_features)
g_layer_sizes.extend(g_hidden_layer_sizes)
g_layer_sizes.append(g_num_classes)

# The central features of neural networks are weights and biases.
# We give each link a weight and each node a bias.
g_weights = []
g_biases = []

for i, layer_size in enumerate(g_layer_sizes[:-1]):
    g_weights.append(tf.Variable(tf.random_normal([layer_size, g_layer_sizes[(i+1)]])))
    g_biases.append(tf.Variable(tf.random_normal([g_layer_sizes[(i+1)]])))

# We tell our network to calculate the output of every node.
# And run the output through a sigmoid function.
latest_layer = tf.add(tf.matmul(g_data,g_weights[0]),g_biases[0])
for bias, weight in zip(g_biases[1:], g_weights[1:]):
    layer_output = tf.add(tf.matmul(latest_layer,weight),bias)
    latest_layer = tf.nn.sigmoid(layer_output)
g_prediction = latest_layer


g_loss_function = tf.losses.mean_squared_error(g_prediction, g_label)
g_training = tf.train.GradientDescentOptimizer(g_learning_rate).minimize(g_loss_function)
g_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(g_label, 1), tf.argmax(g_prediction, 1)), tf.float32))

########################################################################
## Running the graph ###################################################
########################################################################

# How large should batches be? 1 = online learning
batch_size = 5

# How many times do we want the model to loop over the full data set (epochs)
num_epochs = 1000

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Determine the amount of batches.
num_batches = int(len(data_train) / batch_size)

# Run the complete data set multiple times through the neural network.
for i in range(num_epochs):
    
    # randomly shuffle the data for stochastic gradient descent to work properly
    assert np.shape(data_train)[0] == np.shape(label_train)[0]
    p = np.random.permutation(len(data_train))
    data_train, label_train = data_train[p], label_train[p]
    
    # For every epoch, all batches should be processed.
    for j in range(num_batches):
        batch_label = label_train[j * batch_size:(j+1) * batch_size]
        batch_data = data_train[j * batch_size:(j+1) * batch_size]
        sess.run(g_training, {g_data: np.array(batch_data), g_label: np.array(batch_label)})
        
    # Determine the accuracy after every epoch
    model_accuracy = sess.run(g_accuracy, {g_data: data_test, g_label: label_test})
    print('Epoch ' + str(i) + ': the accuracy of the model after this epoch is ' + str(model_accuracy))