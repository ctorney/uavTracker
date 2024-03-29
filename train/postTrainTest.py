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
from utils.decoder import decode, interval_overlap, bbox_iou, get_prediction_results, get_AP
from utils.utils import md5check, read_tsets, init_config

def main(args):

    config = init_config(args)

    step_by_step = config['args_step']
    args_visual = config['args_visual']
    if step_by_step:
        args_visual = True

    #loading args ends here, so it is easier to run section by section in interpreter shell
    data_dir = config['project_directory']
    project_name = config['project_name']

    obj_label = config['common']['LABELS'][0]

    groundtruths_dir = os.path.join(data_dir, config['groundtruths_dir'])
    predictions_dir_general = os.path.join(data_dir, config['predictions_dir'])
    os.makedirs(predictions_dir_general, exist_ok=True)
    os.makedirs(groundtruths_dir, exist_ok=True)

    annotations_dir = os.path.join(data_dir, config['annotations_dir'])
    weights_dir = os.path.join(data_dir, config['weights_dir'])

    results_config_file = os.path.join(data_dir, config['results_dir'], config['results_config_name'])
    try:
        with open(results_config_file, 'r') as handle:
            results_config = yaml.safe_load(handle)
    except:
        raise Exception('The results file doesn\'t exist and should have been created during training stage.')

    c_date = results_config['c_date']

    results_config['AP'] = dict()
    for tset_it in range(len(config['testing_sets'])+1):
        #all sets
        setname = ''
        if tset_it == len(config['testing_sets']):
            list_of_test_files = read_tsets(config,'testing',c_date,config['testing_sets'])
            setname = 'all_sets'
        else:
            list_of_test_files = read_tsets(config,'testing',c_date,[config['testing_sets'][tset_it]])
            setname = config['testing_sets'][tset_it]

        with open(list_of_test_files, 'r') as fp:
            all_imgs = yaml.safe_load(fp)

        if args_visual:
            cv2.namedWindow('tracker', cv2.WINDOW_GUI_EXPANDED)
            cv2.moveWindow('tracker', 20,20)

        results_config['AP'][setname] = dict()
        for model_name in config['models'].keys():

            results_config['AP'][setname][model_name] = dict()

            c_model = config['models'][model_name]
            num_class = c_model['num_class']
            obj_thresh = c_model['obj_thresh']#we are not using that as we need small threshold to let the mAP calculation get a good range.
            nms_thresh = c_model['nms_thresh']
            trained_weights = dict()
            trained_weights['phase_one'] = f'{weights_dir}{project_name}_{model_name}_phase_one_{c_date}.h5'
            trained_weights['phase_two'] = f'{weights_dir}{project_name}_{model_name}_phase_two_{c_date}.h5'

            for iii in range(c_model['phases']):
                training_phase = 'phase_one' if iii == 0 else 'phase_two'
                results_config['AP'][setname][model_name][training_phase] = dict()
                pr_list = {
                    0.25 : ([], 0),
                    0.5 : ([], 0),
                    0.75 : ([], 0),
                    0.9 : ([], 0),
                    0.95 : ([], 0),
                    }

                if results_config['predictions_performed']:
                    model_predictions_file = results_config['annotated_predictions'][model_name]
                    print('Opening the already predicted files in file ' + model_predictions_file)
                    with open(annot_model, 'r') as fp:
                        pred_imgs = yaml.safe_load(fp)

                max_l = config['common']['MAX_L']  #maximal object size in pixels
                min_l = config['common']['MIN_L']

                ##################################################
                print("Loading YOLO models")
                print("We will use the following model for testing: ")
                print(trained_weights[training_phase])
                yolov3 = get_yolo_model(num_class, trainable=False)
                try:
                    yolov3.load_weights(
                        trained_weights[training_phase], by_name=True)  #TODO is by_name necessary here?
                except:
                    print("Missing this YOLO model, skipping")
                    continue

                print("YOLO models loaded, my dear.")
                ########################################
                predictions_dir = os.path.join(predictions_dir_general,model_name + '_' + training_phase + '_' + c_date)
                os.makedirs(predictions_dir, exist_ok=True)

                #read in all images from checked annotations (GROUND TRUTH)
                for i in range(len(all_imgs)):
                    filename = all_imgs[i]['filename']
                    full_name = filename.replace('/','-').lstrip('.').lstrip('-')
                    fname_gt = os.path.join(groundtruths_dir, full_name + ".txt")
                    fname_pred = os.path.join(predictions_dir, full_name + ".txt")

                    img_data = {'object': []}
                    img_data['filename'] = filename
                    img_data['width'] = all_imgs[i]['width']
                    img_data['height'] = all_imgs[i]['height']

                    #Reading ground truth
                    boxes_gt = []
                    for obj in all_imgs[i]['object']:
                        boxes_gt.append(
                            [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])

                    #do box processing
                    img = cv2.imread(filename)

                    frame = img.copy()

                    with open(fname_gt, 'w') as file_gt:  #left top righ bottom
                        for b in boxes_gt:
                            obj = {}
                            if ((b[2] - b[0]) * (b[3] - b[1])) < 10:
                                continue
                            obj['name'] = obj_label
                            obj['xmin'] = int(b[0])
                            obj['ymin'] = int(b[1])
                            obj['xmax'] = int(b[2])
                            obj['ymax'] = int(b[3])
                            img_data['object'] += obj_label
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

                    boxes_predict = []
                    if os.path.exists(fname_pred):
                        sys.stdout.write('x')
                        sys.stdout.flush()
                        with open(fname_pred, 'r') as cpred:
                            for line in cpred:
                                sl = line.split(" ")
                                boxes_predict.append([int(sl[2]),
                                                      int(sl[3]),
                                                      int(sl[4]),
                                                      int(sl[5]),
                                                      int(100*float(sl[1]))
                                                          ])
                    # #this is the actual prediction!
                    else:
                        sys.stdout.write('o')
                        sys.stdout.flush()
                        # preprocess the image
                        im_size_h, im_size_w, _ = img.shape
                        new_image = img[:, :, ::-1] / 255.
                        new_image = np.expand_dims(new_image, 0)

                        # run the prediction
                        yolos = yolov3.predict(new_image)
                        boxes_predict = decode(yolos, 0.4, nms_thresh)#we are using a low object threshold to get all candidates
                        only_good_boxes = []
                        for b in boxes_predict:
                            xmin = int(b[0])
                            xmax = int(b[2])
                            ymin = int(b[1])
                            ymax = int(b[3])
                            if xmin < 0: continue
                            if ymin < 0: continue
                            if xmax > im_size_w: continue
                            if ymax > im_size_h: continue
                            if (xmax - xmin) < min_l: continue
                            if (xmax - xmin) > max_l: continue
                            if (ymax - ymin) < min_l: continue
                            if (ymax - ymin) > max_l: continue
                            only_good_boxes.append(b)

                        boxes_predict = only_good_boxes

                        if step_by_step:
                            sys.stdout.write('Yolo predicting...')
                            sys.stdout.flush()
                            sys.stdout.write('decoding...')
                            sys.stdout.flush()
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

                    ### ###
                    #caluclate scores for this image
                    #and add to the calculations for the whole dataset
                    for iou_thresh in pr_list.keys():
                        prediction_list, nall = get_prediction_results(boxes_predict,boxes_gt, iou_thresh)
                        acc_prediction_list = pr_list[iou_thresh][0]
                        acc_nall = pr_list[iou_thresh][1]
                        acc_prediction_list = acc_prediction_list + prediction_list
                        acc_nall += nall
                        pr_list[iou_thresh]=(acc_prediction_list,acc_nall)
                    # print(len(pr_list[0.5][0]))
                    if args_visual:
                        cv2.imshow('tracker', frame)
                        if step_by_step:
                            key = cv2.waitKey(0)  #& 0xFF
                        else:
                            key = cv2.waitKey(20)  #& 0xFF
                    #precision = tp / (tp + fp)
                    # for box_gt in boxes_gt:
                    #     for box_predict in boxes_predict:
                    #         iou_val = bbox_iou(box_predict,box_gt)
                    #         print(iou_val)

                #count prediction which reache a threshold of let's say 0.5
                # if we cahnge the dection threshold I think we'll get ROC curve - that'd be cute.
                results_config['AP'][setname][model_name][training_phase] = dict()
                print('\nNow calculating AP. If detector is really, really bad, this sage can take ages. Just see how many ')
                for iou_thresh in pr_list.keys():
                    prediction_list = pr_list[iou_thresh][0]
                    pred_per_img = len(prediction_list)/len(all_imgs)
                    print(f'IoU thresh is {iou_thresh} and we have {pred_per_img:.2f} predictions per image.')
                    nall = pr_list[iou_thresh][1]
                    results_config['AP'][setname][model_name][training_phase][iou_thresh] = get_AP(prediction_list,nall)
                AP5 = results_config['AP'][setname][model_name][training_phase][0.5]
                print(f'Finished {model_name} phase {training_phase} on setname {setname} with 0.5 AP of {AP5}! :o)')

    with open(results_config_file, 'w') as handle:
        yaml.dump(results_config, handle)


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
    parser.add_argument('--step', '-s', default=False, action='store_true',
                        help='step by step!')

    args = parser.parse_args()
    main(args)
