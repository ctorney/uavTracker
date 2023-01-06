"""
Comparing output of my calculation to the example from
https://github.com/rafaelpadilla/Object-Detection-Metrics

The above repository seem to round IoU value before thresholding, meaning that to get the results matching my
ioU thresh needs to be 0.29 instead of 0.3

"""
import numpy as np
import cv2 as cv
import sys
sys.path.append('..')
from utils.decoder import get_prediction_results, get_AP
import glob

#reminder:
#Bounding boxes are x_top_left, y_tl, x_bpr, y_br


fdets = glob.glob('detections/*txt')
fgts = glob.glob('groundtruths/*txt')
paired_boxes = []

for iii in range(len(fdets)):
    detlist = []
    fdet = fdets[iii]
    cdet= open(fdet, "r")
    for line in cdet:
        sl = line.split(" ")
        detlist.append([int(sl[2]),
                        int(sl[3]),
                        int(sl[2]) + int(sl[4]),
                        int(sl[3]) + int(sl[5]),
                        int(100*float(sl[1]))
                        ])

    gtlist = []
    fgt = fgts[iii]
    cgt= open(fgt, "r")
    for line in cgt:
        sl = line.split(" ")
        gtlist.append([int(sl[1]),
                        int(sl[2]),
                        int(sl[1]) + int(sl[3]),
                        int(sl[2]) + int(sl[4]),
                        ])
    paired_boxes.append((detlist,gtlist))

prediction_list_full = []
nall_full = 0

for (boxes_pred,boxes_gt) in paired_boxes:
    print('=============== = = = =============')
    # Create a black image
    img = np.zeros((200,200,3), np.uint8)
    # xmin, ymin, xmax, ymax, prob_det


    for b in boxes_gt:
        cv.rectangle(img,(b[0],b[1]),(b[2],b[3]),(12,234,40),3)

    for b in boxes_pred:
        cv.rectangle(img,(b[0],b[1]),(b[2],b[3]),(0,34,233),1)
        cv.putText(img,str(b[4]),(b[2]+5,b[3]+5), cv.FONT_HERSHEY_SIMPLEX, 0.3,(255,255,255),1,cv.LINE_AA)

    iou_thresh=0.29

    prediction_list, nall = get_prediction_results(boxes_pred,boxes_gt, iou_thresh)
    nall_full = nall_full + nall
    prediction_list_full = prediction_list_full + prediction_list

    AP =  float(get_AP(prediction_list, nall))

    cv.putText(img,str(f'{AP:.2}'),(30,400), cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv.LINE_AA)

    cv.namedWindow("Display window")
    cv.imshow("Display window", img)
    k = cv.waitKey(0)
    cv.destroyWindow("Display window")
    cv.destroyAllWindows()

AP =  float(get_AP(prediction_list_full, nall_full))
print(AP)
