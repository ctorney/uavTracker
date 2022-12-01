import numpy as np
import cv2 as cv
import sys
sys.path.append('..')
from utils.decoder import get_prediction_results


# Create a black image
img = np.zeros((512,512,3), np.uint8)
# xmin, ymin, xmax, ymax, prob_det
boxes_pred = [[0,0,20,120,89],
              [25,22,30,98,100],
              [21,22,30,78,45],
              [55,50,160,130,80],
              [209,221,280,265,40],
              [220,221,265,244,70],
              [500,20,570,50,99],
              ]

boxes_gt = [[20,20,35,80],
              [50,50,150,150],
              [200,220,270,250],
              [500,20,570,50],
              ]


for b in boxes_gt:
    cv.rectangle(img,(b[0],b[1]),(b[2],b[3]),(12,234,40),3)

for b in boxes_pred:
    cv.rectangle(img,(b[0],b[1]),(b[2],b[3]),(0,34,233),1)
    cv.putText(img,str(b[4]),(b[2]+5,b[3]+5), cv.FONT_HERSHEY_SIMPLEX, 0.3,(255,255,255),1,cv.LINE_AA)

iou_thresh=0

prediction_list, nall = get_prediction_results(boxes_pred,boxes_gt, iou_thresh)
AP =  get_AP(prediction_list, nall)

cv.namedWindow("Display window")
cv.imshow("Display window", img)
k = cv.waitKey(0)
cv.destroyWindow("Display window")
cv.destroyAllWindows()
