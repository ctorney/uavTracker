
import SimpleCV
import time
from SimpleCV import Image, VirtualCamera, Display, Features, TreeClassifier, ImageSet, Color
from SimpleCV import SVMClassifier, HueHistogramFeatureExtractor, EdgeHistogramFeatureExtractor, HaarLikeFeatureExtractor
from circularHOGExtractor import circularHOGExtractor
import sys

# def thresholdOpGreen(in_image):
#      return in_image.hueDistance(60).binarize(thresh=70).invert().dilate(2)
#
# def thresholdOp(in_image):
#     return in_image.binarize
#
hhfe = HueHistogramFeatureExtractor(10) #look for hue
#
ehfe = circularHOGExtractor(10) # look at edge orientation
# # look at the basic shape
extractors = [hhfe,ehfe] # put these all together
svm = SVMClassifier(extractors) # try an svm, default is an RBF kernel function
trainPaths = ['./images/train/yes/','./images/train/no/']
testPaths = ['./images/test/yes/','./images/test/no/']
# # define the names of our classes
classes = ['ball','notaball']
# # # train the data
print svm.train(trainPaths,classes,verbose=False)
print "----------------------------------------"
 # now run it against our test data.
print svm.test(testPaths,classes,verbose=False)

import random
test = ImageSet()
for p in trainPaths: # load the data
	test += ImageSet(p)
random.shuffle(test) # shuffle it
test = test[0:10] # pick ten off the top
i = 0
for t in test:
	className = svm.classify(t) # classify them
     	t.drawText(className,10,10,fontsize=80,color=Color.RED)
     	fname = "./timgs/classification"+str(i)+".png"
     	t.applyLayers().resize(w=128).save(fname)
     	i = i + 1
 	test.show()

