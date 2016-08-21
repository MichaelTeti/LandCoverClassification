import scipy
from scipy.io import loadmat
import scipy.misc
import numpy as np
import os 
import sys
sys.path.append('/home/mpcr/newproject/venv/lib/python2.7/site-packages')
import tensorflow as tf


# load data and ground truth
a=loadmat('Pavia.mat') # .mat file that has the data
b=loadmat('Pavia_gt.mat') # load .mat file with the ground truth
gt=b['pavia_gt'] 
gt=np.pad(gt, (4, 4), 'edge') # pad ground truth to coincide with data
data=a['pavia']
depth=np.ma.size(data, 2) # number of bands used in sensing
data=np.pad(data, (4, 4), 'edge') # pad the data matrix to select patches
output=np.ndarray.max(gt) # 9 output nodes in CNN for 9 classes in scene
batch=100 # training batch size
c1=75 # convolutions in layer 1
c2=200 # convolutions in layer 2
h1=400 # nodes in 1st fully-connected layer
h2=400 # nodes in second fully-connected layer

sess=tf.InteractiveSession() # begin tensorflow interactive session


x = tf.placeholder(tf.float32, shape=[None, depth*49]) # input layer of nodes to send data through
y = tf.placeholder(tf.float32, shape=[None, output]) # output layer of the network
keep_prob = tf.placeholder(tf.float32) # dropout probability placeholder

# create the training batch
def training_set(X, y):
	td=np.empty([1, depth*49])
	labels=np.zeros([batch, output])
	for i in range(batch):
		train_pixelr, train_pixelc=np.nonzero(y) # find nonzero elements in ground truth (zeros correspond to no class)
		randp=np.random.permutation(np.ma.size(train_pixelr)) # choose a random selection of these elements
		train_pixelr=train_pixelr[randp[0]]
		train_pixelc=train_pixelc[randp[0]]
		# Don't select padded elements - leave space for patch size
		if train_pixelr<4: 
			train_pixelr=4
		elif train_pixelr>(np.ma.size(y, 0)-4):
			train_pixelr=np.ma.size(y, 0)-4
		if train_pixelc<4:
			train_pixelc=4
		elif train_pixelc>(np.ma.size(y, 1)-4):
			train_pixelc=np.ma.size(y, 1)-4
		# select 7x7 patches with the pixel to be classified in the center
		train_data=X[train_pixelr-3:train_pixelr+4, train_pixelc-3:train_pixelc+4, 4:depth+4]
		train_data=train_data.flatten()
		train_data=train_data[np.newaxis, :]
		td=np.concatenate((td, train_data), axis=0)
		label=y[train_pixelr, train_pixelc]
		labels[i, label-1]=1 # create label matrix from ground truth
	td=td[1:, :]
	# feature scaling of the data
	mu=np.mean(td, axis=0)
	sigma=np.std(td, axis=0)
	td=np.divide((td-mu), sigma)
	td=np.nan_to_num(td)
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

# first convolutional layer 
X=tf.reshape(x, [-1, 7, 7, depth]) # reshape training example into 7x7x103 patches
w1=weight_variable([1, 3, depth, c1]) # first convolutional layer is 1x3
bias1=bias_variable([c1])
act1=max_pool_2x2(tf.nn.relu(conv2d(X, w1)+bias1)) # apply convolutions, rectified linear activation function, and max pooling

w2=weight_variable([3, 1, c1, c2]) # second convolutional layer is 3x1
bias2=bias_variable([c2])
act2=max_pool_2x2(tf.nn.relu(conv2d(act1, w2)+bias2))
act2flat=tf.reshape(act2, [-1, 4*c2]) # reshape layer 2 activations into a vector for fully-connected layer


theta1=weight_variable([4*c2, h1]) # weights for fully-connected layer
bias1=bias_variable([h1]) # bias for first fully-connected layer
activation2=tf.nn.relu(tf.matmul(act2flat, theta1) + bias1) # activations for fully-connected layer
activation2 = tf.nn.dropout(activation2, keep_prob) # apply dropout to activations to avoid overfitting

theta2=weight_variable([h1, h2]) # weights for second fully-connected layer
bias2=bias_variable([h2]) # bias for 2nd layer
activation3=tf.nn.relu(tf.matmul(activation2, theta2) + bias2) # multiply activations from layer one by weights and add bias
activation3=tf.nn.dropout(activation3, keep_prob) # apply dropout to this layer of activations too

theta3=weight_variable([h2, output]) # weights for final layer of the network (output layer)
bias3=bias_variable([output]) # biases for the output nodes
out=tf.nn.softmax(tf.matmul(activation3, theta3) + bias3) # send through a softmax regression function 


# calculate cross-entropy cost function to use with backpropagation
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))

# backpropagation optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

# return one if the output node for a training example is the same as the label
correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))

# calculate the mean for the training batch
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables()) # initialize the tensorflow variables

training_iters=1000 

# training
for i in range(training_iters):
  train_data, train_labels=training_set(data, gt)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:train_data, y:train_labels, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    #tf.Print(y_, train_labels)
  train_step.run(feed_dict={x: train_data, y: train_labels, keep_prob: 0.7})
