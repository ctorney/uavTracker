import numpy as np
import os, cv2, sys
import time, math
sys.path.append("..")
from models.yolo_models import get_yolo_model


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

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


class yoloDetector(object):
    """
    This class creates a yolo object detector

    """

    #
    base = 32.0


    def __init__(self, width, height, wt_file, obj_threshold=0.001, nms_threshold=0.5, max_length=256):
        # YOLO dimensions have to be a multiple of 32 so we'll choose the closest multiple to the image
        self.width = int(round(width / self.base) * self.base)
        self.height = int(round(height / self.base) * self.base)
        self.weight_file = wt_file
        
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.max_length = max_length
        
        self.model = get_yolo_model(self.width, self.height, num_class=1)
        self.model.load_weights(self.weight_file,by_name=True)
        

    def create_detections(self, image, warp=None):

        # resize and normalize
        image = cv2.resize(image, (self.width, self.height))
        new_image = image[:,:,::-1]/255.
        new_image = np.expand_dims(new_image, 0)

        # get detections
        preds = self.model.predict(new_image)

        #print('yolo time: ', (stop-start)/batches)
        new_boxes = np.zeros((0,5))
        for i in range(3):
            netout=preds[i][0]
            grid_h, grid_w = netout.shape[:2]
            xpos = netout[...,0]
            ypos = netout[...,1]
            wpos = netout[...,2]
            hpos = netout[...,3]
                    
            objectness = netout[...,4]

            # select only objects above threshold
            indexes = (objectness > self.obj_threshold) & (wpos<self.max_length) & (hpos<self.max_length)

            if np.sum(indexes)==0:
                continue

            corner1 = np.column_stack((xpos[indexes]-wpos[indexes]/2.0, ypos[indexes]-hpos[indexes]/2.0))
            corner2 = np.column_stack((xpos[indexes]+wpos[indexes]/2.0, ypos[indexes]+hpos[indexes]/2.0))

            if warp is not None:
                # corner1=min,min, corner2=max,max, corner3=min,max, corner4=max,min
                corner3 = np.column_stack((xpos[indexes]-wpos[indexes]/2.0, ypos[indexes]+hpos[indexes]/2.0))
                corner4 = np.column_stack((xpos[indexes]+wpos[indexes]/2.0, ypos[indexes]-hpos[indexes]/2.0))
                # now rotate all 4 corners
                corner1 = np.expand_dims(corner1, axis=0)
                corner1 = cv2.perspectiveTransform(corner1,warp)[0]
                corner2 = np.expand_dims(corner2, axis=0)
                corner2 = cv2.perspectiveTransform(corner2,warp)[0]
                corner3 = np.expand_dims(corner3, axis=0)
                corner3 = cv2.perspectiveTransform(corner3,warp)[0]
                corner4 = np.expand_dims(corner4, axis=0)
                corner4 = cv2.perspectiveTransform(corner4,warp)[0]
                min_x = np.min(np.column_stack((corner1[:,0],corner2[:,0],corner3[:,0],corner4[:,0])),axis=1)
                max_x = np.max(np.column_stack((corner1[:,0],corner2[:,0],corner3[:,0],corner4[:,0])),axis=1)
                min_y = np.min(np.column_stack((corner1[:,1],corner2[:,1],corner3[:,1],corner4[:,1])),axis=1)
                max_y = np.max(np.column_stack((corner1[:,1],corner2[:,1],corner3[:,1],corner4[:,1])),axis=1)
                corner1 = np.column_stack((min_x,min_y))
                corner2 = np.column_stack((max_x,max_y))

            new_boxes = np.append(new_boxes, np.column_stack((corner1, corner2, objectness[indexes])),axis=0)

        # do nms 
        sorted_indices = np.argsort(-new_boxes[:,4])
        boxes=new_boxes.tolist()

        for i in range(len(sorted_indices)):

            index_i = sorted_indices[i]

            if new_boxes[index_i,4] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i][0:4], boxes[index_j][0:4]) >= self.nms_threshold:
                    new_boxes[index_j,4] = 0

        new_boxes = new_boxes[new_boxes[:,4]>0]
        detection_list = []
        for row in new_boxes:
            stacker = (row[0],row[1],row[2],row[3], row[4])
            detection_list.append(stacker)

 
        return detection_list


