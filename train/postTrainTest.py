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
from utils.decoder import decode, _interval_overlap, bbox_iou
from utils.utils import md5check, read_tsets

def main(args):
    args_visual = args.visual
    #Load data
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    data_dir = config['project_directory']
    project_name = config['project_name']

    #loading args ends here, so it is easier to run section by section in interpreter shell
    groundtruths_dir = data_dir + config['groundtruths_dir']
    predictions_dir_general = data_dir + config['predictions_dir']
    os.makedirs(predictions_dir_general, exist_ok=True)
    os.makedirs(groundtruths_dir, exist_ok=True)

    annotations_dir = data_dir + config['annotations_dir']
    weights_dir = data_dir + config['weights_dir']

    results_config_file = data_dir + config['results_dir'] + config['results_config_name']
    with open(results_config_file, 'r') as handle:
        results_config = yaml.safe_load(handle)
    c_date = results_config['c_date']

    list_of_test_files = read_tsets(config,'testing',c_date,config['testing_sets'])

    with open(list_of_test_files, 'r') as fp:
        all_imgs = yaml.safe_load(fp)

    if args_visual:
        cv2.namedWindow('tracker', cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow('tracker', 20,20)

        # # # # # # # #
        # WIP WIP WIP WIP
        # HACK TODO HACK
        ###############

    for model_name in config['models'].keys():

        c_model = config['models'][model_name]
        num_class = c_model['num_class']
        obj_thresh = c_model['obj_thresh']
        nms_thresh = c_model['nms_thresh']
        trained_weights = dict()
        trained_weights['phase_one'] = f'{weights_dir}{project_name}_{model_name}_phase_one_{c_date}.h5'
        trained_weights['phase_two'] = f'{weights_dir}{project_name}_{model_name}_phase_two_{c_date}.h5'

        for iii in range(c_model['phases']):
            training_phase = 'phase_one' if iii == 0 else 'phase_two'

            if results_config['predictions_performed']:
                model_predictions_file = results_config['annotated_predictions'][model_name]
                print('Opening the already predicted files in file ' + model_predictions_file)
                with open(annot_model, 'r') as fp:
                    pred_imgs = yaml.safe_load(fp)

            max_l = config['common']['MAX_L']  #maximal object size in pixels
            min_l = config['common']['MIN_L']
            im_size_h = config['common']['IMAGE_H']  #size of training imageas for yolo
            im_size_w = config['common']['IMAGE_W']  #size of training imageas for yolo

            ##################################################
            print("Loading YOLO models")
            print("We will use the following model for testing: ")
            print(trained_weights[training_phase])
            yolov3 = get_yolo_model(im_size_w, im_size_h, num_class, trainable=False)
            yolov3.load_weights(
                trained_weights[training_phase], by_name=True)  #TODO is by_name necessary here?
            print("YOLO models loaded, my dear.")
            ########################################
            predictions_dir = predictions_dir_general + '/' + model_name + '_' + training_phase + '_' + c_date + '/'
            os.makedirs(predictions_dir, exist_ok=True)

            #read in all images from checked annotations (GROUND TRUTH)
            for i in range(len(all_imgs)):
                filename = all_imgs[i]['filename']
                full_name = filename.replace('/','-').lstrip('.').lstrip('-')
                fname_gt = groundtruths_dir + full_name + ".txt"
                fname_pred = predictions_dir + full_name + ".txt"

                img_data = {'object': []}
                img_data['filename'] = filename
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
                img = cv2.imread(filename)

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

                        if args_visual:
                            cv2.rectangle(
                                frame, (int(obj['xmin']) - 2, int(obj['ymin']) - 2),
                                (int(obj['xmax']) + 2, int(obj['ymax']) + 2), (200, 0, 0), 1)

                if results_config['predictions_performed']:
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

                            if args_visual:
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

                            if args_visual:
                                cv2.rectangle(
                                    frame, (int(objpred['xmin']) - 2, int(objpred['ymin']) - 2),
                                    (int(objpred['xmax']) + 2, int(objpred['ymax']) + 2), (0, 0, 198), 1)
                                str_conf = "{:.1f}".format(objpred['confidence'])
                                cv2.putText(frame, str_conf,  (int(objpred['xmax']),int(objpred['ymax'])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1);

                if args_visual:
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
    parser.add_argument('--visual', '-v', default=False, action='store_true',
                        help='Display tracking progress')

    args = parser.parse_args()
    main(args)
