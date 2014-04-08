
from SimpleCV import Image,  VirtualCamera, Display
import sys
import cv
import cv2
import numpy as np
import time

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
    inputfile = 'Wildies_Movie_03.mov'

    scaling = 1.5

    vir = VirtualCamera(inputfile, "video")
    display = Display()

    mapx,mapy = createMap(scaling)

    lastIm = vir.getImage()
    lastIm = reMap(lastIm, mapx, mapy, scaling)
    lastIm.save(display)
    final = Image((lastIm.width*2,lastIm.height*2))
    xs = int(0.5*lastIm.height)
    ys = int(0.5*lastIm.width)
    final = final.blit(lastIm,pos=(ys,xs))
    final.save(display)
    i=0
    while display.isNotDone():

        thisIm = vir.getImage()
        thisIm = reMap(thisIm, mapx, mapy, scaling)
        thisxs,thisys = findShift(lastIm,thisIm)
        xs += thisxs
        ys += thisys
        final = final*0# Image((lastIm.width*2,lastIm.height*2))
        final = final.blit(thisIm,pos=(int(ys),int(xs)))


        lastIm = thisIm
        #final.save(display)
        final.save("imgs/"+ str(i).zfill(4)  + ".png")
        i+=1

        if vir.getImage().getBitmap() == '': display.done = True
        if display.mouseRight: display.done = True

    display.quit()

if __name__ == "__main__":
   main()