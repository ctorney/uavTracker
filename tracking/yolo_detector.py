import numpy as np
import os, cv2, sys
import time, math
from deep_sort.detection import Detection
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

 #   anchors = np.array([[53.57159857, 42.28639429], [29.47927551, 51.27168234], [37.15496912, 26.17125211]])
    
    anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
    obj_thresh=0.5
    nms_thresh=0.4 #0.25
    nb_box=3
    base = 32.0


    def __init__(self, width, height, wt_file):
        self.width = 608#int(round(width / self.base) * self.base)
        self.height = 608#int(round(height / self.base) * self.base)
        self.weight_file = wt_file
        
        self.model = get_yolo_model(self.width, self.height, num_class=1,features = True)
        self.model.load_weights(self.weight_file,by_name=True)
        

    def create_detections(self, image):
        start_all = time.time()
        image = cv2.resize(image, (self.width, self.height))
        new_image = image[:,:,::-1]/255.
        new_image = np.expand_dims(new_image, 0)
        batches=1
        process  = np.tile(new_image, (batches,1,1,1))

        start = time.time()
        preds = self.model.predict(process)
        stop = time.time()
        print('yolo time: ', (stop-start)/batches)
        new_boxes = np.zeros((0,261))
        features = preds[3][0]
        for i in range(3):
            netout=preds[i][0]
            grid_h, grid_w = netout.shape[:2]
            xpos = netout[...,0]
            ypos = netout[...,1]
            wpos = netout[...,2]
            hpos = netout[...,3]
                    
            objectness = netout[...,4]

            # select only objects above threshold
            indexes = objectness > self.obj_thresh
            skip = features.shape[0]//grid_h
            thisfeat = features[::skip,::skip,:]
            thisfeat = np.expand_dims(thisfeat,2) 
            thisfeat = np.tile(thisfeat,(1,1,3,1))
            new_boxes = np.append(new_boxes, np.column_stack((xpos[indexes]-wpos[indexes]/2.0, \
                                         ypos[indexes]-hpos[indexes]/2.0, \
                                         xpos[indexes]+wpos[indexes]/2.0, \
                                         ypos[indexes]+hpos[indexes]/2.0, \
                                         objectness[indexes], thisfeat[indexes])),axis=0)

        # do nms 
        sorted_indices = np.argsort(-new_boxes[:,4])
        boxes=new_boxes.tolist()

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if new_boxes[index_i,4] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i][0:4], boxes[index_j][0:4]) >= self.nms_thresh:
                    new_boxes[index_j,4] = 0

        new_boxes = new_boxes[new_boxes[:,4]>0]
        detection_list = []
        for row in new_boxes:
            bbox, confidence, feature = (row[0],row[1],row[2]-row[0],row[3]-row[1]), row[4], row[5:]
            detection_list.append(Detection(bbox, confidence, feature))

        stop_all = time.time()
        print('total time: ', stop_all-start_all)
        return detection_list


