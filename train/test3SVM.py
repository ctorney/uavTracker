
import SimpleCV
import time
from SimpleCV import Image, VirtualCamera, Display, Features, TreeClassifier, ImageSet, Color
from SimpleCV import DrawingLayer
from SimpleCV import SVMClassifier, HueHistogramFeatureExtractor, EdgeHistogramFeatureExtractor, HaarLikeFeatureExtractor
import sys
sys.path.append("/home/ctorney/workspace/ungTracker/featureExtractor/")
from circularHOGExtractor import circularHOGExtractor

# def thresholdOpGreen(in_image):
#      return in_image.hueDistance(60).binarize(thresh=70).invert().dilate(2)
#
# def thresholdOp(in_image):
#     return in_image.binarize
#
#
ehfe = circularHOGExtractor(4,4,4) 
hhfe = HueHistogramFeatureExtractor(10) #look for hue
# # look at the basic shape
extractors = [ehfe,hhfe] # put these all together
svm = SVMClassifier(extractors) # try an svm, default is an RBF kernel function
svm.load('trainedSVM.xml')
import random
testPaths = ['./trainyes/','./trainno/']
test = ImageSet()
for p in testPaths: # load the data
	test += ImageSet(p)
random.shuffle(test) # shuffle it
test = test[0:145] # pick ten off the top
i = 0
for t in test:
    finalClass = 'no'
    #if className == 'yes':
    finalClass = svm.classify(t) # classify them
    t.drawText(finalClass,10,10,fontsize=10,color=Color.RED)
    fname = "./timgs/classification"+str(i)+".png"
    t.applyLayers().resize(w=128).save(fname)
    i = i + 1

