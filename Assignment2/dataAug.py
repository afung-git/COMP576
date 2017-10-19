import tensorflow as tf
from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from PIL import Image

ntrain = 1000
ntest = 100
nclass = 10
imsize = 28


'''
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

'''


# Show image as a check

for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        img = Image.open(path)
        img.transpose(Image.FLIP_LEFT_RIGHT).save('Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample+1000))

print('Complete')
