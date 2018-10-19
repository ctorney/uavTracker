import os, sys, glob
import csv
import cv2
import numpy as np

from deep_sort import nn_matching
from deep_sort.detection import Detection
from yolo_detector import yoloDetector
from deep_sort.tracker import Tracker


from sort import *

data_dir =  '/home/staff1/ctorney/data/horses/departures/'
#train_images =  glob.glob( image_dir + "*.png" )
output_file1 = 'out2.avi'

width = 1920
height = 1080

display = True

filelist = glob.glob(data_dir + "*.mp4")

for input_file in filelist:
    direct, ext = os.path.split(input_file)
    noext, _ = os.path.splitext(ext)
    data_file = data_dir + '/tracks/' +  noext + '_POS.txt'
    video_file = data_dir + '/tracks/' +  noext + '_TR.avi'
    if os.path.isfile(data_file):
        continue
    #print(input_file, video_file)
    ##########################################################################
    ##          set-up yolo detector and tracker
    ##########################################################################
    max_cosine_distance = 0.2

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance)
    tracker = Tracker(metric,max_age=60)
    tracker2 = Sort(max_age=25) 

    yolo = yoloDetector(width,height, '../weights/horses-yolo.h5')
    results = []


    ##########################################################################
    ##          open the video file for inputs and outputs
    ##########################################################################
    cap = cv2.VideoCapture(input_file)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    S = (1920,1080)
    if display:
        fourCC = cv2.VideoWriter_fourcc('X','V','I','D')
        out = cv2.VideoWriter(video_file, fourCC, fps, S, True)


    ##########################################################################
    ##          corrections for camera motion
    ##########################################################################
    warp_mode = cv2.MOTION_HOMOGRAPHY
    number_of_iterations = 20
    # Specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-6;
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    im1_gray = np.array([])
    warp_matrix = np.eye(3, 3, dtype=np.float32) 
    full_warp = np.eye(3, 3, dtype=np.float32)



    frame_idx=0
    nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(nframes):

        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%% %d/%d" % ('='*int(20*i/float(nframes)), int(100.0*i/float(nframes)), i,nframes)) 
        sys.stdout.flush()
        ret, frame = cap.read() 
        if not(im1_gray.size):
            # enhance contrast in the image
            im1_gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        
        im2_gray =  cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        

        try:
            # find difference in movement between this frame and the last frame
            (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
            # this frame becames the last frame for the next iteration
            im1_gray = im2_gray.copy()
        except cv2.error as e:
            warp_matrix = np.eye(3, 3, dtype=np.float32)


        # all moves are accumalated into a matrix
        full_warp = np.dot(warp_matrix,full_warp)
        #    full_warp = np.linalg.inv(full_warp)

        detections = yolo.create_detections(frame, np.linalg.inv(full_warp))


        # Update tracker.
        tracks = tracker2.update(np.asarray(detections))

   #     print('===========')
        for track in tracks:
            bbox = track[0:4]
            if display:
                iwarp = (full_warp)
                bwidth = bbox[2]-bbox[0]
                bheight = bbox[3]-bbox[1]
                centre = np.expand_dims([0.5*(bbox[0]+bbox[2]),0.5*(bbox[1]+bbox[3])], axis=0)
                centre = np.expand_dims(centre,axis=0)
                centre = cv2.perspectiveTransform(centre,iwarp)[0,0,:]
  #              if not track.is_confirmed():
  #                  print('uc:', bbox[0],bbox[1],bbox[2],bbox[3], corner1[0],corner1[1])
  #              else:
  #                  print('c:', bbox[0],bbox[1],bbox[2],bbox[3], corner1[0],corner1[1])
             #   corner2 = np.expand_dims([[bbox[2],bbox[3]]], axis=0)
             #   corner2 = cv2.perspectiveTransform(corner2,iwarp)[0,0,:]
                minx = centre[0]-bwidth*0.5
                maxx = centre[0]+bwidth*0.5
                miny = centre[1]-bheight*0.5
                maxy = centre[1]+bheight*0.5

 #               print(track[4])
                cv2.putText(frame, str(int(track[4])),(int(minx), int(miny)),0, 5e-3 * 200, (0,255,0),2)
   #             if bbox[1]>0:
                cv2.rectangle(frame, (int(minx), int(miny)), (int(maxx), int(maxy)),(255,0,0), 4)
    #            else:
    #                cv2.rectangle(frame, (int(corner1[0]), int(corner1[1])), (int(corner2[0]), int(corner2[1])),(0,0,255), 4)

            results.append([frame_idx, track[4], bbox[0], bbox[1], bbox[2], bbox[3]])
  #      print('===========')

#
#        # Update tracker.
#        tracker.predict()
#        tracker.update(detections)
#        for det in detections:
#            dt = det.to_tlbr()
#
#        for track in tracker.tracks:
#            if not track.is_confirmed():
#                continue 
#            if track.time_since_update > 5 :
#                continue
#            bbox = track.to_tlbr()
#            if display:
#                iwarp = (full_warp)
#                bwidth = bbox[2]-bbox[0]
#                bheight = bbox[3]-bbox[1]
#                centre = np.expand_dims([0.5*(bbox[0]+bbox[2]),0.5*(bbox[1]+bbox[3])], axis=0)
#                centre = np.expand_dims(centre,axis=0)
#                centre = cv2.perspectiveTransform(centre,iwarp)[0,0,:]
#  #              if not track.is_confirmed():
#  #                  print('uc:', bbox[0],bbox[1],bbox[2],bbox[3], corner1[0],corner1[1])
#  #              else:
#  #                  print('c:', bbox[0],bbox[1],bbox[2],bbox[3], corner1[0],corner1[1])
#             #   corner2 = np.expand_dims([[bbox[2],bbox[3]]], axis=0)
#             #   corner2 = cv2.perspectiveTransform(corner2,iwarp)[0,0,:]
#                minx = centre[0]-bwidth*0.5
#                maxx = centre[0]+bwidth*0.5
#                miny = centre[1]-bheight*0.5
#                maxy = centre[1]+bheight*0.5
#
#                cv2.putText(frame, str(track.track_id),(int(minx), int(miny)),0, 5e-3 * 200, (0,255,0),2)
#   #             if bbox[1]>0:
#                cv2.rectangle(frame, (int(minx), int(miny)), (int(maxx), int(maxy)),(255,0,0), 4)
#    #            else:
#    #                cv2.rectangle(frame, (int(corner1[0]), int(corner1[1])), (int(corner2[0]), int(corner2[1])),(0,0,255), 4)
#
#            results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
#
        frame_idx+=1

        if display:
     #       cv2.imshow('', frame)
     #       cv2.waitKey(10)
     #       frame = cv2.resize(frame,S)
         #   im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
            out.write(frame)

     #   cv2.imwrite('out.jpg',frame)
     #   break

    with open(data_file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(results)
 #   break
     #   for val in results:
      #      writer.writerow([val])    

