import tensorflow as tf
from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from PIL import Image

ntrain = 4000
ntest = 100
nclass = 10
imsize = 28


# Show image as a check

'''
Flips left to right
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        img = Image.open(path)
        img.transpose(Image.FLIP_LEFT_RIGHT).save('Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample+1000))


for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        img = Image.open(path)
        img.transpose(Image.FLIP_TOP_BOTTOM).save('Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample+2000))
'''

for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        img = Image.open(path)
        img.transpose(Image.ROTATE_90).save('Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample+4000))

print('Complete')

