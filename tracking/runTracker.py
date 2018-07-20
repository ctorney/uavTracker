import os, sys

import cv2
import numpy as np

from deep_sort import nn_matching
from deep_sort.detection import Detection
from yolo_detector import yoloDetector
from deep_sort.tracker import Tracker

sys.path.append("..")
from models.yolo_models import get_yolo_sort



input_file = 'test.avi'
output_file = 'out.avi'

width = 608#1920
height = 608#1056
max_cosine_distance = 0.2
display = True


metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance)
tracker = Tracker(metric)
yolo = yoloDetector(width,height)
results = []

cap = cv2.VideoCapture('test2.avi')

fps = round(cap.get(cv2.CAP_PROP_FPS))
    
S = (1920,1080)
                        
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M','J','P','G'), fps, S, True)
frame_idx=0
for i in range(1000):
    ret, frame = cap.read() 
 #   if i%5!=0:
 #       continue
 #   inframe = cv2.imread('birds.jpg') 
 #  frame=inframe.copy()
 #   if i>0:
 #       frame[:,:-i,:]=inframe[:,i:,:] 
 #       frame[:,-i:,:]=inframe[:,:i,:] 
    
    if ret != True:
        break;

    frame = cv2.resize(frame,(width,height))
    detections = yolo.create_detections(frame)
    # Update tracker.
    tracker.predict()
    tracker.update(detections)
    for det in detections:
        dt = det.to_tlbr()
        cv2.rectangle(frame, (int(dt[0]), int(dt[1])), (int(dt[2]), int(dt[3])),(255,0,0), 4)

    for track in tracker.tracks:
        if not track.is_confirmed() and track.time_since_update >1 :
            continue 
        bbox = track.to_tlbr()
 #       cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
        cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    frame_idx+=1

    if display:
 #       cv2.imshow('', frame)
  #      cv2.waitKey(10)
        frame = cv2.resize(frame,S)
        out.write(frame)

 #   cv2.imwrite('out.jpg',frame)
 #   break

