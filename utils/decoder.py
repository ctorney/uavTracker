
import numpy as np
import pandas as pd
import os,sys
import cv2

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def bbox_iou(box1, box2):
    
    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
    w2, h2 = box2[2]-box2[0], box2[3]-box2[1]
    
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union



def decode_netout(netout, obj_thresh):

    xpos = netout[...,0]
    ypos = netout[...,1]
    wpos = netout[...,2]
    hpos = netout[...,3]
                    
    objectness = netout[...,4]

    # select only objects above threshold
    indexes = objectness > obj_thresh

    new_boxes = np.column_stack((xpos[indexes]-wpos[indexes]/2, \
        ypos[indexes]-hpos[indexes]/2, \
        xpos[indexes]+wpos[indexes]/2, \
        ypos[indexes]+hpos[indexes]/2, \
        objectness[indexes])).tolist()

    return new_boxes
        
def do_nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return
        
    sorted_indices = np.argsort([-box[4] for box in boxes])

    for i in range(len(sorted_indices)):
        index_i = sorted_indices[i]

        if boxes[index_i][4] == 0: continue

        for j in range(i+1, len(sorted_indices)):
            index_j = sorted_indices[j]

            if bbox_iou(boxes[index_i][0:4], boxes[index_j][0:4]) >= nms_thresh:
                boxes[index_j][4] = 0
    return

def decode(yolos, obj_thresh=0.9, nms_thresh=0.5):
    boxes = []

    for i in range(len(yolos)):
        # decode the output of the network
        boxes += decode_netout(yolos[i][0], obj_thresh)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)     

    return_boxes = []

    for b in boxes:
        if b[4]>0:
            return_boxes.append([b[0],b[1],b[2],b[3]])

    return return_boxes
