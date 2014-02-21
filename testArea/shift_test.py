
from SimpleCV import *
import cv2
import time
import numpy as np
import sys, getopt


def findHomography(img,template,quality=500.00,minDist=0.2,minMatch=0.4):
    # cribbed from SimpleCV, the homography sucks for this
    # just use the median of the x offset of the keypoint correspondences
    # to determine how to align the image
    skp,sd = img._getRawKeypoints(quality)
    tkp,td = template._getRawKeypoints(quality)
    if( skp == None or tkp == None ):
        warnings.warn("I didn't get any keypoints. Image might be too uniform or blurry." )
        return None

    template_points = float(td.shape[0])
    sample_points = float(sd.shape[0])
    magic_ratio = 1.00
    if( sample_points > template_points ):
        magic_ratio = float(sd.shape[0])/float(td.shape[0])

    idx,dist = img._getFLANNMatches(sd,td) # match our keypoint descriptors
    p = dist[:,0]
    result = p*magic_ratio < minDist
    pr = result.shape[0]/float(dist.shape[0])

    if( pr > minMatch and len(result)>4 ): # if more than minMatch % matches we go ahead and get the data
        #FIXME this code computes the "correct" homography
        lhs = []
        rhs = []
        for i in range(0,len(idx)):
            if( result[i] ):
                lhs.append((tkp[i].pt[0], tkp[i].pt[1]))             #FIXME note index order
                rhs.append((skp[idx[i]].pt[0], skp[idx[i]].pt[1]))   #FIXME note index order

        rhs_pt = np.array(rhs)
        lhs_pt = np.array(lhs)
        xm = np.median(rhs_pt[:,1]-lhs_pt[:,1])
        ym = np.median(rhs_pt[:,0]-lhs_pt[:,0])
        homography,mask = cv2.findHomography(lhs_pt,rhs_pt,cv2.RANSAC, ransacReprojThreshold=1.1 )
        return (homography,mask, (xm,ym))
    else:
        return None


def main(argv):
    vs = VideoStream("myvideo.avi", 25, True)
    display = Display()



    lastIm = Image("first.png")
    vir = VirtualCamera('Wildies_Movie_03.mov', "video")
    display = Display()# (640,480), title="Simon")



    lastIm = vir.getImage()
    lastIm = lastIm.crop(170,100,1750,1200)
    xt = lastIm.height
    yt = lastIm.width
    xo = int(0.5*xt)
    yo = int(0.5*yt)
    fullSize = np.zeros((xt*2,yt*2,3))

    fullSize[xo:xo+xt, yo:yo+yt,:] = lastIm.getNumpyCv2()
   # Image(fullSize,cv2image=True).save(display)



    #fullSize = np.zeros((thisIm.height*2,thisIm.width*2,3))

    final = Image((lastIm.width*2,lastIm.height*2))
    xs = int(0.5*lastIm.height)
    ys = int(0.5*lastIm.width)
    final = final.blit(lastIm,pos=(ys,xs))
    final.save(display)
    i = 0
    while display.isNotDone():
        thisIm = vir.getImage()# Image("second.png")
        thisIm = thisIm.crop(170,100,1750,1200)


        time.sleep(1)
        H, M, offset = findHomography(lastIm,thisIm)
        xs += int(offset[1])
        ys += int(offset[0])

        final = final*0# Image((lastIm.width*2,lastIm.height*2))
        final = final.blit(thisIm,pos=(ys,xs))
        final.save("imgs/"+ str(i).zfill(3)  + ".png")
        i+=1
        #final.save(vs)
        lastIm = thisIm
        if vir.getImage().getBitmap() == '': display.done = True
        if display.mouseRight: display.done = True

    display.quit()
    #motion = thisIm.findMotion(lastIm,method="BM",window=100)
    #motion.show(width=3)
    #time.sleep(5)
    #vecs = np.array([m.vector() for m in motion])

    #[yav, xav] = np.median(vecs,axis=0)
    #xo -= int(xav)
    #yo -= int(yav)
    #fullSize[xo:xo+xt, yo:yo+yt,:] = thisIm.getNumpyCv2()
    #Image(fullSize,cv2image=True).save(display)
    #lastIm = thisIm
    #temp_image = np.zeros((lastIm.height + int(xav),lastIm.width+int(yav),3))
    #if xav > 0:
    #    xstart = 0
    #    xend = thisIm.height
    #else:
    #    xstart = -int(xav)
    #    xend = thisIm.height
    #if yav > 0:
    #    ystart = 0
    #    yend = thisIm.width
    #else:
    #    ystart = -int(yav)
    #    yend = thisIm.width
    #temp_image[xstart:xend, ystart:yend,:] = thisIm.getNumpyCv2()







if __name__ == "__main__":
   main(sys.argv[1:])