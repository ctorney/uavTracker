'''
'''
import numpy as np
import pandas as pd
import os, sys, glob, yaml, argparse
import cv2
sys.path.append('../..')
sys.path.append('..')
from models.yolo_models import get_yolo_model, get_yolo_model_feats
from utils.decoder import decode
from utils.utils import md5check, makeYoloCompatible, pleaseCheckMyDirectories


def main(args):
    #Load data
    data_dir = args.ddir[0] + '/'  #in case we forgot '/'
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    pleaseCheckMyDirectories(config, data_dir)

    train_dir = data_dir
    weights_dir = data_dir + config['weights_dir']
    annotations_dir = data_dir + config['annotations_dir']
    preped_images_dir = data_dir + config['preped_images_dir']
    preped_images_dir_short = config['preped_images_dir']
    bbox_images_dir = data_dir + config['bbox_images_dir']

    ###### TRAINING DETAILS:
    #Training type dependent
    training_setup = config['training_setup']
    print("Training setup type is " + training_setup)
    print(yaml.dump(config[training_setup]))

    pretrained_weights = weights_dir + config[training_setup]['weights']
    #check md5 of a weights file if available
    md5check(config[training_setup]['weights_md5'], pretrained_weights)

    num_class = config[training_setup]['num_class']
    obj_thresh = config[training_setup]['obj_thresh']
    nms_thresh = config[training_setup]['nms_thresh']

    train_files_regex = config[training_setup]['train_files_regex']
    train_images = glob.glob(data_dir + train_files_regex)
    annotations_file = annotations_dir + config['pretrained_annotations_fname']

    max_l = config['MAX_L']  #max/min object size in pixels
    min_l = config['MIN_L']
    im_width = config['IMAGE_W']  #size of training images for yolo
    im_height = config['IMAGE_H']

    ##################################################
    print("Loading YOLO model: " + pretrained_weights )
    yolov3 = get_yolo_model(im_width, im_height, num_class, trainable=False)
    yolov3.load_weights(
        pretrained_weights, by_name=True)  #TODO is by_name necessary here?
    print("YOLO model loaded, my dear.")
    ########################################

    im_num = 1
    all_imgs = []
    for imagename in train_images:
        im = cv2.imread(imagename)
        print('processing image ' + imagename + ', ' + str(im_num) + ' of ' +
              str(len(train_images)) + '...')
        im_yolo = makeYoloCompatible(im)
        height, width = im_yolo.shape[:2]
        im_num += 1
        n_count = 0

        for x in np.arange(
                0, 1 + width - im_width, im_width
        ):  #'1+' added to allow case when image has exactly size of one window
            for y in np.arange(0, 1 + height - im_height, im_height):
                img_data = {
                    'object': []
                }  #dictionary? key-value pair to store image data
                head, tail = os.path.split(imagename)
                noext, ext = os.path.splitext(tail)
                save_name = preped_images_dir + '/TR_' + noext + '-' + str(n_count) + '.png'
                save_name_short = preped_images_dir_short + '/TR_' + noext + '-' + str(n_count) + '.png'
                box_name = bbox_images_dir + '/ ' + noext + '-' + str(
                    n_count) + '.png'
                img = im[y:y + im_height, x:x + im_width, :]
                cv2.imwrite(save_name, img)
                img_data['filename'] = save_name_short
                img_data['width'] = im_width
                img_data['height'] = im_height

                n_count += 1
                # use the yolov3 model to predict 80 classes on COCO

                # preprocess the image
                image_h, image_w, _ = img.shape
                new_image = img[:, :, ::-1] / 255.
                new_image = np.expand_dims(new_image, 0)

                # run the prediction
                sys.stdout.write('Yolo predicting...')
                sys.stdout.flush()
                yolos = yolov3.predict(new_image)

                sys.stdout.write('Decoding...')
                sys.stdout.flush()
                boxes = decode(yolos, obj_thresh, nms_thresh)
                sys.stdout.write('Done!#of boxes:')
                sys.stdout.write(str(len(boxes)))
                sys.stdout.flush()
                for b in boxes:
                    xmin = int(b[0])
                    xmax = int(b[2])
                    ymin = int(b[1])
                    ymax = int(b[3])
                    obj = {}

                    obj['name'] = 'aoi'

                    if xmin < 0: continue
                    if ymin < 0: continue
                    if xmax > im_width: continue
                    if ymax > im_height: continue
                    if (xmax - xmin) < min_l: continue
                    if (xmax - xmin) > max_l: continue
                    if (ymax - ymin) < min_l: continue
                    if (ymax - ymin) > max_l: continue

                    obj['xmin'] = xmin
                    obj['ymin'] = ymin
                    obj['xmax'] = xmax
                    obj['ymax'] = ymax
                    img_data['object'] += [obj]
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0),
                                  2)

                cv2.imwrite(box_name, img)
                all_imgs += [img_data]

    #print(all_imgs)
    print('Saving data to ' + annotations_file)
    with open(annotations_file, 'w') as handle:
        yaml.dump(all_imgs, handle)

    print('Finished! :o)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects in images using a pre-trained model, and prepare images for the further processing. \n \n  This program is used to pre-annotate images with a pre-trained network (for instance yolo weights). It creates necessary output directories and cuts your images to a given size and writes them to disk.',epilog='Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument('--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--ddir', '-d', required=True, nargs=1, help='Root of your data directory' )

    args = parser.parse_args()
    main(args)
