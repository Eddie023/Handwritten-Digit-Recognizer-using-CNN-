#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:12:58 2017

@author: rattlehead
"""
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

filter_size1 = 5
num_filters1 = 16

filter_size2 = 5
num_filters2 = 36

fc_size = 128 # number of neurons in fully connected layer

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST',one_hot = True)

#data dimensions

img_size = 28
img_size_flat = img_size * img_size 
img_shape = (img_size,img_size)
num_channels = 1 #cause our imput is gray scale that is 1 dimns
num_classes = 10

#helper function for plotting images

def plot_images(images,cls_true,cls_pred=None):
    assert len(images)== len(cls_true)==9
    
    fig,axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape),cmap = 'binary')
        
        if cls_pred is None:
            xlabel = "True: {0}".format(np.argmax(cls_true[i]))
        else:
            xlabel = "True: {0},Pred: {1}".format(np.argmax(cls_true[i]),np.argmax(cls_pred[i]))
        
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
    

images = data.test.images[0:9]
cls_true = data.test.labels[0:9]

plot_images(images=images, cls_true=cls_true)

#helper function for defining weights and biases 
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))

#helper function for creating a new convolution layer
def new_conv_layer(input,
                  num_input_channels,
                  filter_size,
                  num_filters,
                  use_pooling=True):
    #shape of filter-weights for the convolution
    shape =[filter_size,filter_size,num_input_channels,num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    
    layer= tf.nn.conv2d(input=input,filter=weights,
                        strides=[1,1,1,1],
                        padding='SAME')
    layer += biases
    
    #use pooling to down-sample the image resolution
    if use_pooling:
        layer= tf.nn.max_pool(value=layer,
                              ksize=[1,2,2,1],
                              strides=[1,2,2,1],
                              padding='SAME')
    
    layer = tf.nn.relu(layer)
    
    return layer,weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer,[-1,num_features])
    
    return layer_flat , num_features

def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu = True):
    
    weights = new_weights (shape=[num_inputs,num_outputs])
    biases = new_biases(length= num_outputs)
    
    layer = tf.matmul(input,weights)+biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

#placeholder variables
x = tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
x_image = tf.reshape(x,[-1,img_size,img_size,num_channels])
y_true = tf.placeholder(tf.float32,shape=[None,num_classes],name='y_true')
y_true_cls = tf.argmax(y_true,axis =1 )
    
layer_conv1, weights_conv1 = new_conv_layer(
        input= x_image,
        num_input_channels = num_channels,
        filter_size = filter_size1,
        num_filters = num_filters1,
        use_pooling = True)

layer_conv2, weights_conv2 = new_conv_layer(
          input= layer_conv1,
          num_input_channels = num_filters1,
          filter_size = filter_size2,
          num_filters = num_filters2,
          use_pooling = True)

layer_flat,num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs= num_features,
                         num_outputs=fc_size,
                         use_relu= True)
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred,axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels= y_true )
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)

correct_prediction = tf.equal(y_pred_cls,y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

###tensorflow session

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64
total_iterations = 0

def fit(num_iterations):
    global total_iterations
    
    start_time = time.time()
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        x_batch, y_true_batch= data.train.next_batch(train_batch_size)
        feed_dict_train = {x : x_batch,
                          y_true: y_true_batch}
        session.run(optimizer,feed_dict= feed_dict_train)
        
        if i%100 == 0:
            acc = session.run(accuracy,feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6},Training Accuracy: {1:>6.1%}"
            print(msg.format(i+1,acc))
    
    total_iterations += num_iterations
    end_time = time.time()
    time_diff = end_time - start_time
    
    print("time usage:" +str(timedelta(seconds= int(round(time_diff)))))
    

test_batch_size = 256    




def print_test_accuracy(show_example_errors=False
                        ):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test,dtype=np.int)
    
    
    
    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.


    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.labels
    cls_true = np.argmax(cls_true,axis=1)
    

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
   

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()
    

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)
        
   
    


def predict(image):
    num_test = 1
    cls_pred = np.zeros(shape=1,dtype=np.int)
    own_image = image
    own_image = own_image.reshape(-1)
    own_image = own_image.reshape(1,784)
    own_image = np.absolute(own_image -255)
    #labels = data.test.labels[1]
    #labels = labels.reshape(1,10)
    feed_dict = {x:own_image}
    cls_pred = session.run(y_pred_cls,feed_dict = feed_dict)
    print("Network predicts:",cls_pred[0])



    
    
    
    #matplotlib.pyplot.imshow(data.test.images[6].reshape(28,28),cmap="Greys")
    # img_ = scipy.misc.imread('seven.png',flatten=True)
    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.pre


    # The starting index for the next batch is denoted i.
   