
from SimpleCV import Image,  VirtualCamera, Display
import sys
import cv
import cv2
import numpy as np
import time
import Tkinter
import tkMessageBox
import pygame

def findShift(img,template,quality=500.00,minDist=0.2,minMatch=0.4):

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
        lhs = []
        rhs = []
        for i in range(0,len(idx)):
            if( result[i] ):
                lhs.append((tkp[i].pt[0], tkp[i].pt[1]))
                rhs.append((skp[idx[i]].pt[0], skp[idx[i]].pt[1]))

        rhs_pt = np.array(rhs)
        lhs_pt = np.array(lhs)
        ym = np.median(rhs_pt[:,1]-lhs_pt[:,1])
        xm = np.median(rhs_pt[:,0]-lhs_pt[:,0])

        return xm,ym
    else:
        return 0,0

def createMap(scaling):

    xs = 1920
    ys = 1440
    fx = 960
    fy = 720
    cx = 960*scaling
    cy = 720*scaling
    k1 = -0.105418957
    k2 = 0.0123732
    p1 = 0
    p2 =  0
    k3 = -0.0008337

    intrinsics = np.zeros((3, 3))
    intrinsics[0, 0] = float(fx)
    intrinsics[1, 1] = float(fy)
    intrinsics[2, 2] = 1.0
    intrinsics[0, 2] = float(cx)
    intrinsics[1, 2] = float(cy)

    dist_coeffs = np.zeros(5)
    dist_coeffs[0] = float(k1)
    dist_coeffs[1] = float(k2)
    dist_coeffs[2] = float(p1)
    dist_coeffs[3] = float(p2)
    dist_coeffs[4] = float(k3)

    return cv2.initUndistortRectifyMap(intrinsics, dist_coeffs,None,intrinsics,(int(scaling*xs), int(scaling*ys)),cv2.CV_32FC1)

def reMap(src, mapx, mapy, scaling):


    in_image = Image((src.width*scaling,src.height*scaling))
    xs = int(0.5*(scaling-1)*src.height)
    ys = int(0.5*(scaling-1)*src.width)
    in_image = in_image.blit(src,pos=(ys,xs))


    dst = cv2.remap(in_image.getNumpyCv2(), mapx, mapy,cv2.INTER_LINEAR)
    cx = 330*scaling
    cy = 180*scaling
    lx = 1350*scaling
    ly = 1050*scaling
    return Image(dst,cv2image=True)#.crop(cx,cy,lx,ly)




def main():
    inputfile = '../Wildies_Movie_03.mov'

    scaling = 1.5

    vir = VirtualCamera(inputfile, "video")
    display = Display(title="positive samples")

    mapx,mapy = createMap(scaling)

    lastIm = vir.getImage()
    lastIm = reMap(lastIm, mapx, mapy, scaling)
    final = Image((lastIm.width*2,lastIm.height*2))
    xs = int(0.5*lastIm.height)
    ys = int(0.5*lastIm.width)
    final = final.blit(lastIm,pos=(ys,xs))
    i=0
    counter = 0
    thisIm = vir.getImage()
    thisIm.save(display)
    counter = 56
    while display.isNotDone():
        vir.skipFrames(100)

 #       thisIm = reMap(thisIm, mapx, mapy, scaling)



        if display.mouseLeft: 
            box_dim = 3*32#min(2*max(blob.minRectWidth(),blob.minRectHeight()),min(w,h))
            tmpImg = thisIm.crop(display.mouseX, display.mouseY, box_dim,box_dim, centered=True)
            display2 = Display(title="click centre")
            tmpImg.save(display2)
            while display2.isNotDone():
                if display2.mouseLeft: 
                    display2.done = True
                    box_dim = 32#min(2*max(blob.minRectWidth(),blob.minRectHeight()),min(w,h))
                    tmpImg = tmpImg.crop(display2.mouseX, display2.mouseY, box_dim,box_dim, centered=True)
                    save_path = "yes/img-" + str(counter) + ".png"
                    tmpImg.save(save_path)
                    counter += 1
                    wait = False
            thisIm.save(display)
            display.mouseLeft = False




        #final.save("imgs/"+ str(i).zfill(4)  + ".png")

        if display.mouseRight: 
            thisIm = vir.getImage()
            thisIm.save(display)
        if thisIm.getBitmap() == '': display.done = True

if __name__ == "__main__":
   main()
