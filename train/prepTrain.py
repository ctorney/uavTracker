'''
'''
import numpy as np
import pandas as pd
import os, sys, glob, yaml, argparse
import cv2
sys.path.append('../..')
sys.path.append('..')
from models.yolo_models import get_yolo_model
from utils.decoder import decode
from utils.utils import md5check, makeYoloCompatible, pleaseCheckMyDirectories

"""
Run through the existing annotations and create a list of files without any annotations.
checked(manual) and pre(auto)

Returns a dictionary where keys are directory paths and values are lists of filenames from those directories that need to be annotated
"""
def filter_out_annotations(ss_imgs_all,some_exists,annotations_file):
    all_annot_imgs = []
    to_annot_dict = dict()

    if some_exists:
        print(f"Loading autogen images annotations from {annotations_file}")
        with open(annotations_file, 'r') as fp:
            all_annot_imgs = all_annot_imgs + yaml.safe_load(fp)

        annot_filenames = []
        for annotation_data in all_annot_imgs:
            annot_filenames.append(annotation_data['filename'])

        for ssdir, ssi in ss_imgs_all.items():
            to_annot_dict[ssdir]=[]
            if not ssi in annot_filenames:
                to_annot_dict[ssdir].append(ssi)
    else:
        to_annot_dict = ss_imgs_all

    return to_annot_dict

"""
Get a list of files that are not in checked (manual), or pre-annotated (auto) annotations file.
Returns a list of images.
"""
def read_for_annotation(config):
    subsets = config['subsets']

    checked_annotations = config['project_directory'] + config['annotations_dir'] + '/' + config['checked_annotations_fname']
    autogen_annotations = config['project_directory'] + config['annotations_dir'] + '/' + config['auto_annotations_fname']
    some_checked = md5check(config['checked_annotations_md5'], checked_annotations)
    some_autogen = md5check(config['auto_annotations_md5'], auto_annotations)

    #Get a list of all the files that we need annotations for to do all those trainings and testings in the config.
    list_of_subsets = [x for x in subsets]
    ss_imgs_all = read_subsets(list_of_subsets,config)

    #run through the existing annotations and create a list of files without any annotations
    to_annot_imgs0 = filter_out_annotations(ss_imgs_all,some_checked,checked_annotations)
    to_annot_imgs1 = filter_out_annotations(ss_imgs_all,some_autogen,autogen_annotations)

    #TODO merge those two dictionaries.

    return to_annot_imgs

def main(args):
    #Load data
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    data_dir = config['project_directory'] + '/'

    #logging and debugging setup
    DEBUG = args.debug
    TEST_RUN = args.test_run

    pleaseCheckMyDirectories(config, data_dir)

    n_models_to_train = len(config['models'].keys())

    print(f'For every of {n_models_to_train} different models for this experiment we will review if provided training and testing files have annotations. ')

    for model in config['models'].keys():
        pre_annotate(model, config, data_dir, DEBUG, TEST_RUN)
    print(f'Finished pre-annotating for all models. Run annotate.py to correct annotations.')

def pre_annotate(model_name, config, data_dir, DEBUG, TEST_RUN):

    c_model = config['models'][model_name]
    project_name = config['project_name']

    train_dir = data_dir
    weights_dir = data_dir + config['weights_dir']
    annotations_dir = data_dir + config['annotations_dir']

    #The following are *I KID YOU NOT* sort of global variables. tf function yolo_loss has only two arguemts that are explicit x and y. the rest must be global...
    LABELS = config['common']['LABELS']
    IMAGE_H = config['common']['IMAGE_H']
    IMAGE_W = config['common']['IMAGE_W']
    max_l = config['common']['MAX_L']  #max/min object size in pixels
    min_l = config['common']['MIN_L']
    im_width = IMAGE_W
    im_height = IMAGE_H
    NO_OBJECT_SCALE = config['common']['NO_OBJECT_SCALE']
    OBJECT_SCALE = config['common']['OBJECT_SCALE']
    COORD_SCALE = config['common']['COORD_SCALE']
    CLASS_SCALE = config['common']['CLASS_SCALE']


    pretrained_weights = weights_dir + c_model['pretrained_weights']
    md5check(c_model['pretrained_weights_md5'], pretrained_weights)
    model = get_yolo_model(
        IMAGE_W, IMAGE_H, num_class=len(LABELS), headtrainable=True, raw_features=False)
    print("Loading weights %s", pretrained_weights)
    model.load_weights(pretrained_weights, by_name=True)

    # get a list of files that are not in checked, or pre-annotated
    todo_imgs = read_for_annotation(config)
    #for each subset
    #

    list_of_train_files = read_tsets(config,model_name,c_date,c_model['training_sets'])
    ##### OOOOOLD #####
    train_files_regex = config[training_setup]['train_files_regex']
    train_images = glob.glob(data_dir + train_files_regex)
    annotations_file = annotations_dir + config['pretrained_annotations_fname']


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
                save_name_short = 'TR_' + noext + '-' + str(n_count) + '.png'
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

    args = parser.parse_args()
    main(args)
