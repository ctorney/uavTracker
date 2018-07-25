import os, sys, glob

import cv2
import numpy as np

from deep_sort import nn_matching
from deep_sort.detection import Detection
from yolo_detector import yoloDetector
from deep_sort.tracker import Tracker



data_dir =  '/home/staff1/ctorney/data/horses/departures/'
input_file = '180602-4.mp4'
input_file = '170610-1-1.mp4'
#train_images =  glob.glob( image_dir + "*.png" )
output_file = 'out2.avi'

width = 1920
height = 1080
max_cosine_distance = 0.2
display = True


metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance)
tracker = Tracker(metric)
yolo = yoloDetector(width,height, '../weights/horses-yolo.h5')
results = []

cap = cv2.VideoCapture(data_dir + input_file)

fps = round(cap.get(cv2.CAP_PROP_FPS))
    
S = (1920,1080)
                        
fourCC = cv2.VideoWriter_fourcc('X','V','I','D')

out = cv2.VideoWriter(output_file, fourCC, fps, S, True)
frame_idx=0
nframes=3600
nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(nframes):

    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%% %d/%d" % ('='*int(20*i/float(nframes)), int(100.0*i/float(nframes)), i,nframes)) 
    sys.stdout.flush()
    ret, frame = cap.read() 
    detections = yolo.create_detections(frame)

    # Update tracker.
    tracker.predict()
    tracker.update(detections)
    for det in detections:
        dt = det.to_tlbr()

    for track in tracker.tracks:
        if not track.is_confirmed():
            continue 
        if track.time_since_update > 5 :
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    frame_idx+=1

    if display:
 #       cv2.imshow('', frame)
 #       cv2.waitKey(10)
 #       frame = cv2.resize(frame,S)
        out.write(frame)

 #   cv2.imwrite('out.jpg',frame)
 #   break

