import numpy as np
import cv2 as cv
import sys
sys.path.append('..')
from utils.decoder import get_prediction_results, get_AP


#reminder:
#Bounding boxes are x_top_left, y_tl, x_bpr, y_br

real_pred = [[971,330,1100,412,69],
             [784,378,921,501,99],
             [168,428,315,489,41],
             [20,457,202,525,98],
             [1,529,44,639,95]]

real_gt = [[783,380,926,511],
           [12,436,190,518]]



boxes_pred = [[0,0,20,120,89],
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

paired_boxes = [(real_pred, real_gt),
                (boxes_pred,boxes_gt), #a larger example, too complex to interpret
                ([
                    [20,10,100,102,100],
                    [70,102,144,152,88]
                  ],[
                      [20,10,100,100],
                      [70,100,140,150]
                  ]), #almost perfect
                ([
                    [0,10,10,20,87],
                    [10,10,20,20,88]
                  ],
                 [
                      [200,200,400,400],
                      [70,100,140,150]
                  ]), #complete miss
                ([
                    [0,0,100,100,77],
                    [100,120,200,300,84]
                  ],
                 [
                      [0,0,80,120]
                  ]), #one fp one tp
                ([
                    [0,0,100,100,88],
                    [100,120,200,300,70]
                  ],
                 [
                      [0,0,80,120]
                  ]), #one fp one tp, reverted prob
                ]

for (boxes_pred,boxes_gt) in paired_boxes:
    print('=============== = = = =============')
    # Create a black image
    img = np.zeros((800,1200,3), np.uint8)
    # xmin, ymin, xmax, ymax, prob_det


    for b in boxes_gt:
        cv.rectangle(img,(b[0],b[1]),(b[2],b[3]),(12,234,40),3)

    for b in boxes_pred:
        cv.rectangle(img,(b[0],b[1]),(b[2],b[3]),(0,34,233),1)
        cv.putText(img,str(b[4]),(b[2]+5,b[3]+5), cv.FONT_HERSHEY_SIMPLEX, 0.3,(255,255,255),1,cv.LINE_AA)

    iou_thresh=0.5

    prediction_list, nall = get_prediction_results(boxes_pred,boxes_gt, iou_thresh)
    AP =  float(get_AP(prediction_list, nall))

    cv.putText(img,str(f'{AP:.2}'),(30,400), cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv.LINE_AA)

    cv.namedWindow("Display window")
    cv.imshow("Display window", img)
    k = cv.waitKey(0)
    cv.destroyWindow("Display window")
    cv.destroyAllWindows()
