import tensorflow as tf 
from tensorflow.contrib import rnn  #, #rnn_cell
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data

random.seed(3)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  #call mnist function

learningRate = .02 #.01 .001
trainingIters = 2000000
batchSize = 280
displayStep = 100

nInput = 28  #we want the input to take the 28 pixels
nSteps = 28  #every 28
nHidden = 100 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(value=x, num_or_size_splits=nSteps, axis=0) #configuring so you can get it as needed for the 28 pixels

	#lstmCell = rnn.BasicLSTMCell(nHidden, forget_bias=1.0)  #find which lstm to use in the documentation
	#lstmCell = rnn.BasicRNNCell(nHidden)	#Basic RNN
	lstmCell = rnn.GRUCell(nHidden)			#GRU Cell

	outputs, states = tf.contrib.rnn.static_rnn(lstmCell, x, dtype=tf.float32) #for the rnn where to get the output and hidden state

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(cost)

correctPred = tf.equal(tf.arg_max(pred,1), tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	step = 1

	testData = mnist.test.images.reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels

	while step* batchSize < trainingIters:
		batchX, batchY = mnist.train.next_batch(batchSize)  #mnist has a way to get the next batch
		batchX = batchX.reshape((batchSize, nSteps, nInput))

		sess.run(optimizer, feed_dict={x: batchX, y: batchY})


		if step % displayStep == 0:
			acc = sess.run(accuracy, feed_dict={x: batchX, y: batchY})
			loss = sess.run(cost, feed_dict={x: batchX, y: batchY})
			testacc = sess.run(accuracy, feed_dict={x: testData, y: testLabel})
			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + "{:0.5f}".format(loss)
				  + ", Training Accuracy= " + "{:0.5f}".format(acc) + ", Test Accuracy= " + "{:0.5f}".format(testacc))
		step +=1
	print('Optimization finished')


	print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
