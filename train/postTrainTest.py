'''
Test the output of a newly trained classifier.
'''
import numpy as np
import pandas as pd
import os, sys, glob, argparse
import cv2
import yaml
import pickle
sys.path.append('..')
from models.yolo_models import get_yolo_model
from utils.decoder import decode
from utils.utils import md5check


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):

    intersect_w = _interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = _interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])

    intersect = intersect_w * intersect_h

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def main(args):
    #Load data
    data_dir = args.ddir[0] + '/'  #in case we forgot '/'
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    image_dir = data_dir + config['preped_images_dir']
    train_dir = data_dir
    weights_dir = data_dir + config['weights_dir']
    groundtruths_dir = data_dir + config['groundtruths_dir']
    predictions_dir = data_dir + config['predictions_dir']

    #Training type dependent
    tracking_setup = config["tracking_setup"]
    trained_weights = weights_dir + config[tracking_setup]['weights']

    #based on get_yolo_model defaults and previous makTrain.py files
    num_class = config[tracking_setup]['num_class']
    obj_thresh = config[tracking_setup]['obj_thresh']
    nms_thresh = config[tracking_setup]['nms_thresh']

    annotations_dir = data_dir + config['annotations_dir']
    list_of_train_files = annotations_dir + config['checked_annotations_fname']
    annotations_file = annotations_dir + config['checked_annotations_fname']
    print("With annotations file")
    print(annotations_file)

    with open(annotations_file, 'r') as fp:
        all_imgs = yaml.load(fp)

    if args.annotated:
        print('Opening the already predicted files in file ' + args.annotated[0])
        with open(args.annotated[0], 'r') as fp:
            pred_imgs = yaml.load(fp)



    if args.visual:
        cv2.namedWindow('tracker', cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow('tracker', 20,20)

    max_l = config['MAX_L']  #maximal object size in pixels
    min_l = config['MIN_L']
    im_size_h = config['IMAGE_H']  #size of training imageas for yolo
    im_size_w = config['IMAGE_W']  #size of training imageas for yolo

    ##################################################
    print("Loading YOLO models")
    print("We will use the following model for testing: ")
    print(trained_weights)
    yolov3 = get_yolo_model(im_size_w, im_size_h, num_class, trainable=False)
    yolov3.load_weights(
        trained_weights, by_name=True)  #TODO is by_name necessary here?
    print("YOLO models loaded, my dear.")
    ########################################

    #read in all images from checked annotations (GROUND TRUTH)
    for i in range(len(all_imgs)):
        basename = os.path.basename(all_imgs[i]['filename'])
        #remove extension from basename:
        name_seed_split = basename.split('.')[:-1]
        name_seed = '.'.join(name_seed_split)
        fname_gt = groundtruths_dir + name_seed + ".txt"
        fname_pred = predictions_dir + name_seed + ".txt"

        img_data = {'object': []}
        img_data['filename'] = basename
        img_data['width'] = all_imgs[i]['width']
        img_data['height'] = all_imgs[i]['height']

        #Reading ground truth
        boxes_gt = []
        for obj in all_imgs[i]['object']:
            boxes_gt.append(
                [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
       # sys.stdout.write('GT objects:')
       # sys.stdout.write(str(len(boxes_gt)))
       # sys.stdout.flush()
        #do box processing
        img = cv2.imread(image_dir + basename)

        # print("File, {}".format(image_dir + basename))
        mmFname = basename.split('_f')[-1].split('_')[0]
        mmFrame = basename.split('_f')[-1].split('_')[1].split('f')[0]
        mmGT = str(len(boxes_gt))

        frame = img.copy()

        with open(fname_gt, 'w') as file_gt:  #left top righ bottom
            for b in boxes_gt:
                obj = {}
                if ((b[2] - b[0]) * (b[3] - b[1])) < 10:
                    continue
                obj['name'] = 'aoi'
                obj['xmin'] = int(b[0])
                obj['ymin'] = int(b[1])
                obj['xmax'] = int(b[2])
                obj['ymax'] = int(b[3])
                img_data['object'] += [obj]
                file_gt.write(obj['name'] + " ")
                file_gt.write(str(obj['xmin']) + " ")
                file_gt.write(str(obj['ymin']) + " ")
                file_gt.write(str(obj['xmax']) + " ")
                file_gt.write(str(obj['ymax']))
                file_gt.write('\n')

                if args.visual:
                    cv2.rectangle(
                        frame, (int(obj['xmin']) - 2, int(obj['ymin']) - 2),
                        (int(obj['xmax']) + 2, int(obj['ymax']) + 2), (200, 0, 0), 1)

        if args.annotated:
            boxes_pred = []
            for obj in pred_imgs[i]['object']:
                boxes_pred.append(
                    [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            with open(fname_pred, 'w') as file_pred:  #left top righ bottom
                for b in boxes_pred:
                    obj = {}
                    if ((b[2] - b[0]) * (b[3] - b[1])) < 10:
                        continue
                    obj['name'] = 'aoi'
                    obj['xmin'] = int(b[0])
                    obj['ymin'] = int(b[1])
                    obj['xmax'] = int(b[2])
                    obj['ymax'] = int(b[3])
                    img_data['object'] += [obj]
                    file_pred.write(obj['name'] + " ")
                    file_pred.write('100' + " ") # we don't store probability of detection in annotations
                    file_pred.write(str(obj['xmin']) + " ")
                    file_pred.write(str(obj['ymin']) + " ")
                    file_pred.write(str(obj['xmax']) + " ")
                    file_pred.write(str(obj['ymax']))
                    file_pred.write('\n')

                    if args.visual:
                        cv2.rectangle(
                            frame, (int(obj['xmin']) - 2, int(obj['ymin']) - 2),
                            (int(obj['xmax']) + 2, int(obj['ymax']) + 2), (200, 0, 0), 1)

            #caluclate scores for this image
            mmTP = 0
            for bgt in boxes_gt:
                for bpred in boxes_pred:
                    if bbox_iou(bgt,bpred) > 0.5:
                        mmTP = mmTP + 1 #find one matching prediction
                        break

            mmFP = 0
            has_match = False
            for bpred in boxes_pred:
                for bgt in boxes_gt:
                    if bbox_iou(bgt,bpred) > 0.5:
                        has_match = True
                        break # found a match for predicion
                if has_match == True:
                    has_match = False
                else:
                    mmFP = mmFP + 1

            #display scores for this image
            print(mmFname + ', ' + mmFrame + ', ' + str(mmGT) + ', ' + str(mmTP) + ', ' + str(mmFP))
 
        else:
            # preprocess the image
            image_h, image_w, _ = img.shape
            new_image = img[:, :, ::-1] / 255.
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

            with open(fname_pred, 'w') as file_pred:  #left top righ bottom
                for b in boxes_predict:
                    xmin = int(b[0])
                    xmax = int(b[2])
                    ymin = int(b[1])
                    ymax = int(b[3])
                    confidence = float(b[4])
                    objpred = {}

                    objpred['name'] = 'aoi'

                    if xmin < 0: continue
                    if ymin < 0: continue
                    if xmax > im_size_w: continue
                    if ymax > im_size_h: continue
                    if (xmax - xmin) < min_l: continue
                    if (xmax - xmin) > max_l: continue
                    if (ymax - ymin) < min_l: continue
                    if (ymax - ymin) > max_l: continue

                    objpred['xmin'] = xmin
                    objpred['ymin'] = ymin
                    objpred['xmax'] = xmax
                    objpred['ymax'] = ymax
                    objpred['confidence'] = confidence
                    file_pred.write(objpred['name'] + " ")
                    file_pred.write(str(objpred['confidence']) + " ")
                    file_pred.write(str(objpred['xmin']) + " ")
                    file_pred.write(str(objpred['ymin']) + " ")
                    file_pred.write(str(objpred['xmax']) + " ")
                    file_pred.write(str(objpred['ymax']))
                    file_pred.write('\n')

                    if args.visual:
                        cv2.rectangle(
                            frame, (int(objpred['xmin']) - 2, int(objpred['ymin']) - 2),
                            (int(objpred['xmax']) + 2, int(objpred['ymax']) + 2), (0, 0, 198), 1)
                        str_conf = "{:.1f}".format(objpred['confidence'])
                        cv2.putText(frame, str_conf,  (int(objpred['xmax']),int(objpred['ymax'])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1);

        if args.visual:
            cv2.imshow('tracker', frame)
            key = cv2.waitKey(1)  #& 0xFF
        #precision = tp / (tp + fp)
        # for box_gt in boxes_gt:
        #     for box_predict in boxes_predict:
        #         iou_val = bbox_iou(box_predict,box_gt)
        #         print(iou_val)

    #count prediction which reache a threshold of let's say 0.5
    # if we cahnge the dection threshold I think we'll get ROC curve - that'd be cute.

    print('Finished! :o)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Prepare list of detection of original input files using final classifier. Those files are saved in groundtruths and predictions directories which can be interpreted by program https://github.com/rafaelpadilla/Object-Detection-Metrics',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument(
        '--ddir',
        '-d',
        required=True,
        nargs=1,
        help='Root of your data directory')
    parser.add_argument('--visual', '-v', default=False, action='store_true',
                        help='Display tracking progress')
    parser.add_argument('--annotated', '-a', required=False, nargs=1,
                        help='Provide file with annotated results if you have already run prediction')

    args = parser.parse_args()
    main(args)
