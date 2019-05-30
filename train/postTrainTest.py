'''
Test the output of a newly trained classifier.
'''
import numpy as np
import pandas as pd
import os,sys,glob
import cv2
import yaml
import pickle
sys.path.append('..')
from models.yolo_models import get_yolo_model, get_yolo_model_feats
from utils.decoder import decode
from utils.utils import md5check

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


def main(argv):
    if(len(sys.argv) != 3):
        print('Usage ./postTrainTest.py [data_dir] [config.yml]')
        sys.exit(1)
    #Load data
    data_dir = argv[1]  + '/' #in case we forgot '/'
    print('Opening file' + argv[2])
    with open(argv[2], 'r') as configfile:
        config = yaml.safe_load(configfile)

    image_dir = data_dir
    train_dir = data_dir
    weights_dir = data_dir + config['weights_dir']

    #Training type dependent
    trained_weights = weights_dir + config['trained_weights']

    #based on get_yolo_model defaults and previous makTrain.py files
    num_class=config['specific']['num_class']
    obj_thresh=config['specific']['obj_thresh']
    nms_thresh=config['specific']['nms_thresh']

    list_of_train_files = config['checked_annotations_fname']

    annotations_file = train_dir + config['untrained_annotations_fname']
    with open (annotations_file, 'r') as fp:
        all_imgs = yaml.load(fp)

    max_l=config['MAX_L'] #maximal object size in pixels
    min_l=config['MIN_L']
    im_size=config['IMAGE_H'] #size of training imageas for yolo

    ##################################################
    print("Loading YOLO models")
    yolov3 = get_yolo_model(im_size,im_size,num_class,trainable=False)
    yolov3.load_weights(trained_weights,by_name=True) #TODO is by_name necessary here?
    print("YOLO models loaded, my dear.")
    ########################################

    #read in all images from checked annotations (GROUND TRUTH)
    for i in range(len(all_imgs)):
        basename = os.path.basename(all_imgs[i]['filename'])

        #remove extension from basename:
        name_seed_split = basename.split('.')[:-1]
        name_seed = '.'.join(name_seed_split)
        fname_gt = image_dir + "/groundtruths/" + name_seed + ".txt"
        fname_pred = image_dir + "/predictions/" + name_seed + ".txt"

        img_data = {'object':[]}
        img_data['filename'] = basename
        img_data['width'] = all_imgs[i]['width']
        img_data['height'] = all_imgs[i]['height']

        #Reading ground truth
        boxes_gt=[]
        for obj in all_imgs[i]['object']:
            boxes_gt.append([obj['xmin'],obj['ymin'],obj['xmax'],obj['ymax']])
        sys.stdout.write('GT objects:')
        sys.stdout.write(str(len(boxes_gt)))
        sys.stdout.flush()
        #do box processing
        img = cv2.imread(image_dir + basename)

        with open(fname_gt, 'w') as file_gt: #left top righ bottom
            for b in boxes_gt:
                obj = {}
                if ((b[2]-b[0])*(b[3]-b[1]))<10:
                    continue
                obj['name'] = 'aoi'
                obj['xmin'] = int(b[0])
                obj['ymin'] = int(b[1])
                obj['xmax'] = int(b[2])
                obj['ymax'] = int(b[3])
                img_data['object'] += [obj]
                file_gt.write(obj['name'] + " " )
                file_gt.write(str(obj['xmin']) + " " )
                file_gt.write(str(obj['ymin']) + " " )
                file_gt.write(str(obj['xmax']) + " " )
                file_gt.write(str(obj['ymax']))
                file_gt.write('\n')

        # preprocess the image
        image_h, image_w, _ = img.shape
        new_image = img[:,:,::-1]/255.
        new_image = np.expand_dims(new_image, 0)

        # run the prediction
        sys.stdout.write('Yolo predicting...')
        sys.stdout.flush()
        yolos = yolov3.predict(new_image)
        sys.stdout.write('decoding...')
        sys.stdout.flush()
        boxes_predict = decode(yolos, obj_thresh, nms_thresh)
        sys.stdout.write('done!#of boxes_predict:')
        sys.stdout.write(str(len(boxes_predict)))
        sys.stdout.write('\n')
        sys.stdout.flush()

        with open(fname_pred, 'w') as file_pred: #left top righ bottom
            for b in boxes_predict:
                xmin=int(b[0])
                xmax=int(b[2])
                ymin=int(b[1])
                ymax=int(b[3])
                confidence=float(b[4])
                objpred = {}

                objpred['name'] = 'aoi'

                if xmin<0: continue
                if ymin<0: continue
                if xmax>im_size: continue
                if ymax>im_size: continue
                if (xmax-xmin)<min_l: continue
                if (xmax-xmin)>max_l: continue
                if (ymax-ymin)<min_l: continue
                if (ymax-ymin)>max_l: continue

                objpred['xmin'] = xmin
                objpred['ymin'] = ymin
                objpred['xmax'] = xmax
                objpred['ymax'] = ymax
                objpred['confidence'] = confidence
                file_pred.write(objpred['name'] + " " )
                file_pred.write(str(objpred['confidence']) + " " )
                file_pred.write(str(objpred['xmin']) + " " )
                file_pred.write(str(objpred['ymin']) + " " )
                file_pred.write(str(objpred['xmax']) + " " )
                file_pred.write(str(objpred['ymax']))
                file_pred.write('\n')


        #precision = tp / (tp + fp)
        # for box_gt in boxes_gt:
        #     for box_predict in boxes_predict:
        #         iou_val = bbox_iou(box_predict,box_gt)
        #         print(iou_val)

    #count prediction which reache a threshold of let's say 0.5
    # if we cahnge the dection threshold I think we'll get ROC curve - that'd be cute.

    print('Finished! :o)')
if __name__ == '__main__':
    main(sys.argv)
