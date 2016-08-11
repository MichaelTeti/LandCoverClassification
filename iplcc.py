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
gt=np.pad(gt, (4, 4), 'edge')
data=a['indian_pines_corrected']
data=np.pad(data, (4, 4), 'edge')
channels=np.ma.shape(data)
depth=200
output=np.ndarray.max(gt)+1
batch=100
c1=75
c2=150
c3=300
c4=400
h1=600
h2=600
h3=600

sess=tf.InteractiveSession() # begin tensorflow session

x = tf.placeholder(tf.float32, shape=[None, depth*49])
y = tf.placeholder(tf.float32, shape=[None, output])
keep_prob = tf.placeholder(tf.float32)

def training_set(X, y):
	td=np.empty([1, depth*49])
	labels=np.zeros([batch, 17])
	for i in range(batch):
		rr=np.arange(4, 149)
		rrow=np.random.permutation(rr)
		rcol=np.random.permutation(rr)
		train_data=X[rrow[0]-3:rrow[0]+4, rcol[0]-3:rcol[0]+4, 4:204]
		train_data=train_data.flatten()
		train_data=train_data[np.newaxis, :]
		td=np.concatenate((td, train_data), axis=0)
		label=y[rrow[0], rcol[0]]
		labels[i, label]=1
	td=td[1:, :]
	if np.ma.size(td, 0)!=batch:
		print 'Data not loaded correctly. Goodbye'
		sys.exit()
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

X=tf.reshape(x, [-1, 7, 7, 200])
w1=weight_variable([2, 2, depth, c1])
bias1=bias_variable([c1])
act1=max_pool_2x2(tf.nn.relu(conv2d(X, w1)+bias1))

w2=weight_variable([2, 2, c1, c2])
bias2=bias_variable([c2])
act2=max_pool_2x2(tf.nn.relu(conv2d(act1, w2)+bias2))
act2flat=tf.reshape(act2, [-1, 4*c2])

theta1=weight_variable([4*c2, h1])
bias1=bias_variable([h1])
activation2=tf.nn.relu(tf.matmul(act2flat, theta1) + bias1)
activation2 = tf.nn.dropout(activation2, keep_prob)

theta2=weight_variable([h1, h2])
bias2=bias_variable([h2])
activation3=tf.nn.relu(tf.matmul(activation2, theta2) + bias2)
activation3=tf.nn.dropout(activation3, keep_prob)

theta3=weight_variable([h2, output])
bias3=bias_variable([output])
out=tf.nn.softmax(tf.matmul(activation3, theta3) + bias3)

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
  train_step.run(feed_dict={x: train_data, y: train_labels, keep_prob: 0.75})
