
import SimpleCV
import time
from SimpleCV import Image, VirtualCamera, Display, Features, TreeClassifier, ImageSet, Color
from SimpleCV import SVMClassifier, TreeClassifier
import sys
import random

sys.path.append("/home/ctorney/workspace/ungTracker/featureExtractor/")
#from circularHOGExtractor import circularHOGExtractor

#ch = circularHOGExtractor(4,4,4) 
#extractor = [ch] # put these all together
#tree = TreeClassifier(extractor,flavor='Forest') # also try a decision tree
#tree.mBoostedFlavorDict['NTrees'] = 10
#tree.mforestFlavorDict['NTrees'] = 200
#trainPaths = ['./yes/','./no/']
# # define the names of our classes
#classes = ['yes','no']
# # # train the data
#print tree.train(trainPaths,classes,verbose=True)
#tree.save('trainedTREE.xml')

outTest = True

if outTest:
    tree2 = TreeClassifier.load('trainedTREE.xml')
 #   tree2.mClassifier = extractor
    testPaths = ['./trainyes/','./trainno/']
    test = ImageSet()
    for p in testPaths: # load the data
        test += ImageSet(p)
    random.shuffle(test) # shuffle it
    test = test[0:10] # pick ten off the top
    i = 0
    for t in test:
        finalClass = 'no'
        #if className == 'yes':
        finalClass = tree2.classify(t) # classify them
        t.drawText(finalClass,10,10,fontsize=10,color=Color.RED)
        fname = "./timgs/classification"+str(i)+".png"
        t.applyLayers().resize(w=128).save(fname)
        i = i + 1

