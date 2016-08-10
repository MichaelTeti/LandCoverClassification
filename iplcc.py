import scipy
from scipy.io import loadmat
import scipy.misc
import numpy as np
import os 
import sys
sys.path.append('/home/mpcr/newproject/venv/lib/python2.7/site-packages')
import tensorflow as tf


# load data and ground truth
a=loadmat('Indian_pines_corrected.mat')
b=loadmat('Indian_pines_gt.mat')
gt=b['indian_pines_gt']
gt=np.lib.pad(gt, ((2, 2), (2, 2)), 'edge')
data=a['indian_pines_corrected']
channels=np.ma.shape(data)
depth=channels[2]
dta=np.zeros([149, 149, 200])
for i in range(depth):
	d=data[:, :, i]
	d=np.lib.pad(d, ((2, 2), (2, 2)), 'edge')
	dta[:, :, i]=d
data=dta
output=np.ndarray.max(gt)+1
batch=65
h1=900
h2=900
h3=900
h4=900

sess=tf.InteractiveSession() # begin tensorflow session

x = tf.placeholder(tf.float32, shape=[None, depth*9])
y = tf.placeholder(tf.float32, shape=[None, output])
keep_prob = tf.placeholder(tf.float32)

def training_set(X, y):
	td=np.empty([1, depth*9])
	labels=np.zeros([batch, 17])
	rr=np.arange(2, 145)
	rrow=np.random.permutation(rr)
	rrow=rrow[0:batch]
	rc=np.arange(2,145)
	rcol=np.random.permutation(rc)
	rcol=rcol[0:batch]
	for i in range(np.ma.size(rrow)):
		train_data=X[rrow[i]-1:rrow[i]+2, rcol[i]-1:rcol[i]+2, :]
		train_data=train_data.flatten()
		train_data=train_data[np.newaxis, :]
		td=np.concatenate((td, train_data), axis=0)
		label=y[rrow[i], rcol[i]]
		labels[i, label]=1
	td=td[1:, :]
	return td, labels

# create small, random weights
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# create biases
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


theta1=weight_variable([depth*9, h1])
bias1=bias_variable([h1])
activation2=tf.nn.relu(tf.matmul(x, theta1) + bias1)
activation2 = tf.nn.dropout(activation2, keep_prob)

theta2=weight_variable([h1, h2])
bias2=bias_variable([h2])
activation3=tf.nn.relu(tf.matmul(activation2, theta2) + bias2)
activation3=tf.nn.dropout(activation3, keep_prob)

theta3=weight_variable([h2, h3])
bias3=bias_variable([h3])
activation4=tf.nn.relu(tf.matmul(activation3, theta3) + bias3)
activation4=tf.nn.dropout(activation4, keep_prob)

theta4=weight_variable([h3, h4])
bias4=bias_variable([h4])
activation5=tf.nn.relu(tf.matmul(activation4, theta4) + bias4)
activation5=tf.nn.dropout(activation5, keep_prob)

theta5=weight_variable([h4, output])
bias5=bias_variable([output])
out=tf.nn.softmax(tf.matmul(activation5, theta5) + bias5)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

training_iters=10000

# training
for i in range(training_iters):
  train_data, train_labels=training_set(data, gt)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:train_data, y:train_labels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    #tf.Print(y_, train_labels)
  train_step.run(feed_dict={x: train_data, y: train_labels, keep_prob: 0.6})

