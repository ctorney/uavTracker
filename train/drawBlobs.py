
import SimpleCV
import time
from SimpleCV import Image, VirtualCamera, Display, Features, TreeClassifier, ImageSet, Color
from SimpleCV import DrawingLayer
from SimpleCV import SVMClassifier, HueHistogramFeatureExtractor, EdgeHistogramFeatureExtractor, HaarLikeFeatureExtractor
import sys
sys.path.append("/home/ctorney/workspace/ungTracker/featureExtractor/")
from circularHOGExtractor import circularHOGExtractor




counter = 0
f = 1000;
#while not display.isDone():
for f in range(1,100):
    filename = '../frames/image-{0:07d}.png'.format(f)
    display = Display()

    thisIm = Image(filename)
    blobs = []

    maxSize = 5000
    minSize = 2000

    blobs = thisIm.dilate(3).findBlobs()
    b2 = thisIm.binarize(blocksize=51,p=10)
    b2.save(display)
    time.sleep(100)
    w = thisIm.width
    h = thisIm.height

    dl = DrawingLayer((w,h))
    if(len(blobs) > 0 ):
            # sort the blobs by size
        blobs = blobs.sortArea()
        for i in range(len(blobs)):
            blob = blobs[i]
            if (blob.mArea < maxSize) and (blob.mArea > minSize):

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
    time.sleep(1)

sys.exit("Error message")
result = tkMessageBox.askquestion("Wildbeest!", "Is this one? (cancel if you don't know)", icon='warning', type='yesnocancel')
if result == 'yes':
    save_path = "yes/img-" + str(counter) + ".png"
    tmpImg.save(save_path)
    counter += 1
if result == 'no':
    save_path = "no/img-" + str(counter) + ".png"
    tmpImg.save(save_path)
    counter += 1




for f in range(1,100):
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
