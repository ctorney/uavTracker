'''
prepTrain.py: Pre-annotate (autogen) training and testing files that do not have annotations provided.

WARNING!!!!
Currently a massive flaw of my new end-to-end setup is that this script will edit the images provided a little bit (adjust them to the size of the yolo detector. I know it is bad and has to be fixed in the future when we want to work with different sized images.
'''
import numpy as np
import pandas as pd
import os, sys, shutil, glob, yaml, argparse, re
import cv2
sys.path.append('../..')
sys.path.append('..')
from models.yolo_models import get_yolo_model
from utils.decoder import decode
from utils.utils import md5check, makeYoloCompatible, pleaseCheckMyDirectories, read_subsets, filter_out_annotations, getmd5

"""
Get a list of files that are not in checked (manual), or pre-annotated (auto) annotations file.
Returns a list of images.
"""
def read_for_annotation(config):
    subsets = config['subsets']

    checked_annotations = config['project_directory'] + config['annotations_dir'] + '/' + config['checked_annotations_fname']
    autogen_annotations = config['project_directory'] + config['annotations_dir'] + '/' + config['autogen_annotations_fname']
    some_checked = md5check(config['checked_annotations_md5'], checked_annotations)
    some_autogen = md5check(config['autogen_annotations_md5'], autogen_annotations)

    #Get a list of all the files that we need annotations for to do all those trainings and testings in the config.
    list_of_subsets = [x for x in subsets]
    ss_imgs_all = read_subsets(list_of_subsets,config)

    #run through the existing annotations and create a list of files without any annotations
    to_annot_imgs0 = filter_out_annotations(ss_imgs_all,some_checked,checked_annotations)
    to_annot_imgs1 = filter_out_annotations(ss_imgs_all,some_autogen,autogen_annotations)

    #for this part we want the ones that are neither auto nor checked
    to_annot_imgs = dict()
    for k_dir in ss_imgs_all.keys():
        to_annot_imgs[k_dir] = list(set(to_annot_imgs0[k_dir]) & set(to_annot_imgs1[k_dir]))

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

    pre_annotate(config, data_dir, DEBUG, TEST_RUN)

    print(f'Finished pre-annotating for all models. Run annotate.py to correct annotations.')

def pre_annotate(config, data_dir, DEBUG, TEST_RUN):

    c_model = config['autogen_model']
    project_name = config['project_name']

    train_dir = data_dir
    weights_dir = os.path.join(data_dir, config['weights_dir'])
    annotations_dir = os.path.join(data_dir, config['annotations_dir'])
    raw_imgs_dir_name = config['raw_imgs_dir']

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
    obj_thresh = c_model['obj_thresh']
    nms_thresh = c_model['nms_thresh']


    pretrained_weights = weights_dir + c_model['pretrained_weights']
    md5check(c_model['pretrained_weights_md5'], pretrained_weights)
    model = get_yolo_model(
        IMAGE_W, IMAGE_H, num_class=len(LABELS), headtrainable=True, raw_features=False)
    print("Loading weights %s", pretrained_weights)
    model.load_weights(pretrained_weights, by_name=True)

    # get a list of files that are not in checked, or pre-annotated
    todo_imgs = read_for_annotation(config)
    #Those files will most likely be altered by the process of making them yolo compatible


    #We are also providing a dict with lists of files in the subset in order to get them to be copied from raw directory to the processed one if needed
    flist_dict = dict()
    for _, s in config['subsets'].items():
        cdir = os.path.join(data_dir, s['directory'])
        cflist = s['filelist']
        flist_dict[cdir]=cflist

    im_num = 1
    all_imgs = []
    for ssdir, sslist in todo_imgs.items():
        if raw_imgs_dir_name in ssdir:
                newdir = re.sub(raw_imgs_dir_name,'',ssdir)
                os.makedirs(newdir)
                if flist_dict[newdir]!='':
                    shutil.copyfile(os.path.join(ssdir,flist_dict[newdir]),os.path.join(newdir,flist_dict[newdir]) )
        for imagename in sslist:
            fullname = ssdir + imagename
            im = cv2.imread(fullname)
            print('processing image ' + imagename + ', ' + str(im_num) + ' of ' +
                str(len(sslist)) + '...')
            #that's gonna be a hack........
            if raw_imgs_dir_name in ssdir:
                fullname = re.sub(raw_imgs_dir_name,'',fullname)

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
                    save_name = fullname
                    save_name_short = imagename
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
                    yolos = model.predict(new_image)

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

                    all_imgs += [img_data]

    autogen_annotations = os.path.join(config['project_directory'],config['annotations_dir'],config['autogen_annotations_fname'])
    print('Saving data to ' + autogen_annotations)
    #print(all_imgs)
    some_autogen = md5check(config['autogen_annotations_md5'], autogen_annotations)
    if some_autogen:
        with open(autogen_annotations, 'r') as fp:
            all_imgs = all_imgs + yaml.safe_load(fp)

    with open(autogen_annotations, 'w') as handle:
        yaml.dump(all_imgs, handle)

    getmd5(autogen_annotations)

    print('Finished! :o)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects in images using a pre-trained model, and prepare images for the further processing. \n \n  This program is used to pre-annotate images with a pre-trained network (for instance yolo weights). It creates necessary output directories and cuts your images to a given size and writes them to disk.',epilog='Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument('--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Set this flag to see a bit more wordy output')
    parser.add_argument('--test-run', default=False, action='store_true',
                        help='Set this flag to see meaningless output quicker. For instance in training it runs only for 1 epoch')

    args = parser.parse_args()
    main(args)
