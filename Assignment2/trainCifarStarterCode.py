from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import os

# --------------------------------------------------
# setup

result_dir = './results/'  # directory where the results from the training are saved
random.seed(3)

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_max


# Loading CIFAR10 images from director

ntrain = 1000
ntest = 100
nclass = 10
imsize = 28
nchannels = 1
batchsize = 64

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'Data/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'Data/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

# Show image as a check
#img = Train[2,:,:,0]
#print(LTrain[2,:])
#plt.imshow(img, cmap='gray')
#plt.show()

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])  #tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass])  #tf variable for labels
fc1_drop = tf.placeholder(tf.float32)
layer1_drop = tf.placeholder(tf.float32)
layer2_drop = tf.placeholder(tf.float32)

# --------------------------------------------------
# model
#create your model

#reshape input image to a 4D tensor
x_image = tf.reshape(tf_data, [-1, imsize, imsize, nchannels])


# First Layer

# Conv layer with kernel 5x5 and 32 filter maps followed by ReLu activation
W_conv1 = weight_variable([5, 5, nchannels, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Max Pooling layer subsampling by 2
h_pool1 = max_pool_2x2(h_conv1)
h_pool1 = tf.nn.dropout(h_pool1, layer1_drop)


# Second Layer

# Conv layer with kernal 5x5 and 64 filter maps followed by ReLu activation
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2 = tf.nn.dropout(h_pool2, layer2_drop)

# FC layer that has input 7*7*64 vector and output 1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# flatten h_pool2 back to a 1xn vector
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
h_fc1_drop = tf.nn.dropout(h_fc1, fc1_drop)

# Second FC layer that has 1024 input and output 10 classes
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# --------------------------------------------------
# loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))
#set up the loss, optimization, evaluation, and accuracy

#optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to result_dir
summary_op = tf.summary.merge_all()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

# --------------------------------------------------
# optimization

sess.run(tf.global_variables_initializer())
batch_xs = np.zeros([batchsize, imsize, imsize, nchannels])  #setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize, nclass])  #setup as [batchsize, the how many classes]
nsamples = ntrain*nclass
for i in range(20000):  # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]
    if i%100 == 0:
        #calculate train accuracy and print it
        print("After %d steps, Train accuracy: %g" % (i, accuracy.eval(feed_dict={tf_data: batch_xs,
                                                                                  tf_labels: batch_ys, layer1_drop: 1.0,
                                                                                  layer2_drop: 1.0, fc1_drop: 1.0})))
        #print("test accuracy: %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, layer1_drop: 1.0,
        #                                                     layer2_drop: 1.0, fc1_drop: 1.0}))
        #summary_str, acc = sess.run([summary_op, accuracy], feed_dict={tf_data: batch_xs, tf_labels: batch_ys,
        #                                                               layer1_drop: 1.0, layer2_drop: .5, fc1_drop: 0.4})
        #summary_writer.add_summary(summary_str, i)
        #summary_str, loss = sess.run([summary_op, cross_entropy], feed_dict={tf_data: batch_xs, tf_labels: batch_ys,
         #                                                                    layer1_drop: 1.0, layer2_drop: .5, fc1_drop: 0.4})
        #summary_writer.add_summary(summary_str, i)
        #summary_writer.flush()
        #checkpoint_file = os.path.join(result_dir, 'checkpoint')
        #saver.save(sess, checkpoint_file, global_step=i)
        #print("Test accuracy %g:" % (accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})))

    if i%1000 == 0:
        print("Test accuracy: %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, layer1_drop: 1.0,
                                                         layer2_drop: 1.0, fc1_drop: 1.0}))
    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, layer1_drop: 1.0, layer2_drop: 1.0, fc1_drop: 0.5}) # dropout only during training

# --------------------------------------------------
# test

print("test accuracy %g" % accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest,
                                                    layer1_drop: 1.0, layer2_drop: 1.0, fc1_drop: 1.0}))

weights1 = W_conv1.eval()
sess.close()

# Visualizing Weights in first layer
#img = Train[2,:,:,0]
#print(LTrain[2,:])

print(weights1.shape)

for i in range(32):
    plt.subplot(4, 8, i+1)
    plt.imshow(weights1[:, :, 0, i], cmap='gray')
    plt.title('Filter ' + str(i+1))
    plt.axis('off')
    #plt.subplots_adjust(hspace=.5, wspace=1.0)

plt.show()

#img = mp.pyplot.imshow(wcon1[:,:,0,1])
#mp.pyplot.show()