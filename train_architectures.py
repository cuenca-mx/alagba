# University of Nottingham
# School of COmputer Science
#
# Ricardo Sánchez Castillo
# Student ID 4225015
#
# Program used for training and validating architectures
# It has to be executed from the caffe directory in order to use the library

#=============================PARAMETERS======================================
niter = 1510
useFineTunning = True
routeToFine = 'vgg_turtle/VGG_ILSVRC_19_layers.caffemodel'
folderToTrain = 'arch'
blobLoss = 'loss'
blobAccuracy = 'accuracy'
directoriesTraining = ['Video1positive', 'Video1negative']
directoriesValidate = ['Video2positive', 'Video2negative']
fileToTest = 'turtle_iter_1500.caffemodel'
path = '/Images/Samples/'
test_iters = 342
test_train = 342
files_train = 624
test_valid = 342
files_valid = 342
input_image_size = 224
#=============================PARAMETERS======================================

import os
os.chdir('..')
from os import listdir
from os.path import isfile, join
import sys
sys.path.insert(0, './python')

import caffe
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import random

train_loss = np.zeros(niter)

solver = caffe.SGDSolver('models/' + folderToTrain + '/solver.prototxt')
if useFineTunning:
    solver.net.copy_from('models/' + routeToFine)

print "Start training"

for r in range(niter):
    solver.step(1)
    train_loss[r] = solver.net.blobs[blobLoss].data
    if r % 10 == 0:
        print "iter %d, loss = %f" % (r, train_loss[r])
        np.save('arch', train_loss)
print "Done"

#plt.plot(np.vstack([train_loss, scratch_train_loss]).T)
#plt.title("Fine-tuning versus Traing from scratch")
#plt.show()
print "Calculating accuracy"
accuracy = 0
for it in arange(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs[blobAccuracy].data
accuracy /= test_iters
print 'Accuracy:', accuracy
np.save('arch', train_loss)
print "Saved"

#===================================================TEST TRAIN DATA===============================================
print "Testing training data"

net = caffe.Net('models/' + folderToTrain +'/deploy.prototxt', 'models/' + folderToTrain + '/' + fileToTest, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].reshape(50,3,input_image_size,input_image_size)

a = range(files_train)
indexes = []
for i in xrange(test_train):
    b = a[random.randint(0,len(a) - 1)]
    a.remove(b)
    indexes = indexes + [b]

positivePositives = 0
positiveNegatives = 0
negativePositive = 0
negativeNegatives = 0

r = 0
for directory in directoriesTraining:
    print "Working {}".format(directory)
    newPath = join(path, directory)
    files = [ s for s in listdir(newPath) if isfile(join(newPath, s))]
    for i in indexes:
        f = files[i]
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(join(newPath, f)))
        out = net.forward()
        result = out['prob'].argmax()
        if directory.find('positive') != -1:
            if result == 1:
                positivePositives += 1
            else:
                positiveNegatives += 1
        else:
            if result == 1:
                negativePositive += 1
            else:
                negativeNegatives += 1
    r += 1

#Show results
print "positivePositives: {}".format(positivePositives)
print "positiveNegatives: {}".format(positiveNegatives)
print "negativePositive: {}".format(negativePositive)
print "negativeNegatives: {}".format(negativeNegatives)
accTrain = ((positivePositives + negativeNegatives) * 100) / (positivePositives + positiveNegatives + negativePositive + negativeNegatives)
print "Accuracy: {}".format(accTrain)

#===================================================TEST VALIDATING DATA===============================================
print "Testing validating data"

a = range(files_valid)
indexes = []
for i in xrange(test_valid):
    b = a[random.randint(0,len(a) - 1)]
    a.remove(b)
    indexes = indexes + [b]

positivePositives = 0
positiveNegatives = 0
negativePositive = 0
negativeNegatives = 0

r = 0
for directory in directoriesValidate:
    print "Working {}".format(directory)
    newPath = join(path, directory)
    files = [ s for s in listdir(newPath) if isfile(join(newPath, s))]
    for i in indexes:
        f = files[i]
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(join(newPath, f)))
        out = net.forward()
        result = out['prob'].argmax()
        if directory.find('positive') != -1:
            if result == 1:
                positivePositives += 1
            else:
                positiveNegatives += 1
        else:
            if result == 1:
                negativePositive += 1
            else:
                negativeNegatives += 1
    r += 1

#Show results
print "positivePositives: {}".format(positivePositives)
print "positiveNegatives: {}".format(positiveNegatives)
print "negativePositive: {}".format(negativePositive)
print "negativeNegatives: {}".format(negativeNegatives)
accTrain = ((positivePositives + negativeNegatives) * 100) / (positivePositives + positiveNegatives + negativePositive + negativeNegatives)
print "Accuracy: {}".format(accTrain)
