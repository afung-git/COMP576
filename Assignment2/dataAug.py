from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

ntrain = 6000
ntest = 100
nclass = 10
imsize = 28


# Show image as a check


#Flips left to right
'''
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
        img.transpose(Image.ROTATE_90).save('Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample+6000))



'''
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        img = Image.open(path)
        img.filter(ImageFilter.GaussianBlur(1)).save('Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample+2000))


       
img = Image.open('Data/CIFAR10/Train/0/Image00000.png')
imgblur = img.filter(ImageFilter.GaussianBlur(1.5)).save('Data/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample+1000))
imgblur.save('blurr.png')


plt.subplot(1, 2, 1)
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.imshow(imgblur)
plt.show()
'''

print("complete")
