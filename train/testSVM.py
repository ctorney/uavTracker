
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
ehfe = circularHOGExtractor(4,6,4) 
# # look at the basic shape
extractors = [ehfe] # put these all together
svm = SVMClassifier(extractors) # try an svm, default is an RBF kernel function
#trainPaths = ['/home/ctorney/Dropbox/projects/wildebeest/images/train/yes/','/home/ctorney/Dropbox/projects/wildebeest/images/train/no/']
trainPaths = ['./tmpyes/','./tmpno/']
#testPaths = ['/home/ctorney/Dropbox/projects/wildebeest/images/test/yes/','/home/ctorney/projects/wildebeest/images/test/no/']
# # define the names of our classes
classes = ['yes','no']
# # # train the data
print svm.train(trainPaths,classes,verbose=False)
print "----------------------------------------"
 # now run it against our test data.
#print svm.test(testPaths,classes,verbose=False)

import random
test = ImageSet()
for p in trainPaths: # load the data
	test += ImageSet(p)
random.shuffle(test) # shuffle it
test = []#test[0:10] # pick ten off the top
i = 0
for t in test:
	className = svm.classify(t) # classify them
     	t.drawText(className,10,10,fontsize=10,color=Color.RED)
     	fname = "./timgs/classification"+str(i)+".png"
     	t.applyLayers().resize(w=128).save(fname)
     	i = i + 1
 	test.show()

inputfile = '../Wildies_Movie_03.mov'

vir = VirtualCamera(inputfile, "video")
display = Display()


counter = 0
while display.isNotDone():
    vir.skipFrames(10)
    thisIm = vir.getImage()
    w = thisIm.width
    h = thisIm.height
    dl = DrawingLayer((w,h))

    blobs = []

    maxSize = 5000
    minSize = 200

    blobs = thisIm.dilate(3).findBlobs()

    if(len(blobs) > 0 ):
            # sort the blobs by size
        blobs = blobs.sortArea()
        for i in range(len(blobs)):
            blob = blobs[i]
            if (blob.mArea < maxSize) and (blob.mArea > minSize):
                box_dim = 48#min(2*max(blob.minRectWidth(),blob.minRectHeight()),min(w,h))


                tmpImg = thisIm.crop(blob.minRectX(), blob.minRectY(), box_dim,box_dim, centered=True)
                if ((tmpImg.width + tmpImg.height) == 2*box_dim):
                    className = svm.classify(tmpImg) # classify them
                    if className=='yes':
                        tmpImg.drawText(className,10,10,fontsize=10,color=Color.RED)
                        fname = "./timgs/classification"+str(counter)+".png"
    #                    tmpImg.applyLayers().resize(w=128).save(fname)
                        counter = counter + 1
                        center_point = (blob.minRectX(), blob.minRectY())
                        cx = blob.minRectX()
                        cy = blob.minRectY()
                        points = [(cx+10,cy+10),(cx-10,cy+10),(cx+10,cy-10),(cx-10,cy-10)]
                        dl.polygon(points, filled=True, color=Color.RED)
                        dl.circle(center_point, 10)
                        #drawbox = dl.centeredRectangle(center_point, (box_dim,box_dim))
                        thisIm.addDrawingLayer(dl)
    thisIm.applyLayers()
    thisIm.save(display)

    if vir.getImage().getBitmap() == '': display.done = True
    if display.mouseRight: display.done = True

display.quit()
