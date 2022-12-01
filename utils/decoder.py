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
        if b[4]>0 and (not np.isnan(b).any()):
            return_boxes.append([b[0],b[1],b[2],b[3],b[4]])

    return return_boxes



"""
for each prediction decide if it is a TP or FP,
if there are multiple pred for the same gt
take one with the highest IoU.
"""
def get_prediction_results(boxes_pred,boxes_gt, iou_thresh):
    prediction_list = []
    if len(boxes_pred) == 0:
        nall = len(boxes_gt) #we'll need to know sum of TP and TNs
        return prediction_list, nall

    predictions_array = np.zeros((len(boxes_gt),len(boxes_pred)))
    for iii in range(len(boxes_gt)):
        bgt = boxes_gt[iii]
        for jjj in range(len(boxes_pred)):
            bpred = boxes_pred[jjj]
            if bbox_iou(bgt,bpred) > iou_thresh:
                predictions_array[iii,jjj] = bbox_iou(bgt,bpred)

    best_preds = np.argmax(predictions_array,axis=1)
    #here we will perform the simplest matching, as we are not going to use this code for reporting final results, only for internal validation

    for jjj in range(len(boxes_pred)):
        v = 'tp' if jjj in best_preds else 'fp'
        prediction_list.append((v,boxes_pred[jjj][4]))


    nall = len(boxes_gt) #we'll need to know sum of TP and TNs
    return prediction_list, nall

#prediction list can be unsorted
def get_AP(prediction_list, nall):
    if len(prediction_list) == 0:
        return 0

    prediction_list.sort(key= lambda x: x[1],reverse=True)

    roc_curve = []
    acc_tp = 0
    acc_fp = 0
    for npred in prediction_list:
        if npred[0] =='tp':
            acc_tp += 1
        else:
            acc_fp += 1

        nrecall = acc_tp / nall
        nprec = acc_tp / (acc_tp + acc_fp)
        roc_curve.append((nrecall,nprec))

    #it will be a list fold
    recalls = [x[0] for x in roc_curve]
    precisions = [x[1] for x in roc_curve]

    AP = 0
    for rrr in range(1,len(recalls)):
        AP += (recalls[rrr]-recalls[rrr-1]) * max(precisions[rrr:])
    return AP
