import numpy as np
import pandas as pd
import os,sys,glob
import cv2
import yaml
sys.path.append('../..')
sys.path.append('..')
from models.yolo_models import get_yolo_model
from utils.decoder import decode
from random import shuffle

def main(argv):
    if(len(sys.argv) != 3):
        print('Usage ./makeTrain.py [root_dir] [config.yml]')
        sys.exit(1)
    #Load data
    root_dir = argv[1]  + '/' #in case we forgot
    print('Opening file' + root_dir + argv[2])
    with open(root_dir + argv[2], 'r') as configfile:
        config = yaml.safe_load(configfile)

    image_dir = root_dir + config['data_dir']
    train_dir = root_dir + config['data_dir']
    your_weights = root_dir + config['weights_dir'] + config['specific_weights']
    trained_annotations_fname = train_dir + config['trained_annotations_fname']
    train_files_regex = config['specific_train_files_regex']

    train_images =  glob.glob( train_dir + train_files_regex )
    shuffle(train_images)

    max_l=100
    min_l=10


    im_size=864 #size of training imageas for yolo


    ##################################################
    #im_size=416 #size of training imageas for yolo
    yolov3 = get_yolo_model(im_size,im_size,num_class=1,trainable=False)
    yolov3.load_weights(your_weights)


    ########################################
    im_num=1
    all_imgs = []
    for imagename in train_images:
        img = cv2.imread(imagename)
        print('processing image ' + imagename + ', ' + str(im_num) + ' of ' + str(len(train_images))  + '...')
        im_num+=1

        img_data = {'object':[]}     #dictionary? key-value pair to store image data
        head, tail = os.path.split(imagename)
        noext, ext = os.path.splitext(tail)
        box_name = train_dir + '/bbox/' + tail

        img_data['width'] = im_size
        img_data['height'] = im_size

        # use the trained yolov3 model to predict

        # preprocess the image
        image_h, image_w, _ = img.shape
        new_image = img[:,:,::-1]/255.
        new_image = np.expand_dims(new_image, 0)

        # run the prediction
        yolos = yolov3.predict(new_image)

        boxes = decode(yolos, obj_thresh=0.2, nms_thresh=0.3)
        for b in boxes:
            xmin=int(b[0])
            xmax=int(b[2])
            ymin=int(b[1])
            ymax=int(b[3])
            obj = {}

            obj['name'] = 'aoi'

            if xmin<0: continue
            if ymin<0: continue
            if xmax>im_size: continue
            if ymax>im_size: continue
            if (xmax-xmin)<min_l: continue
            if (xmax-xmin)>max_l: continue
            if (ymax-ymin)<min_l: continue
            if (ymax-ymin)>max_l: continue


            obj['xmin'] = xmin
            obj['ymin'] = ymin
            obj['xmax'] = xmax
            obj['ymax'] = ymax
            img_data['object'] += [obj]
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

        cv2.imwrite(box_name, img)
        all_imgs += [img_data]


    #print(all_imgs)
    with open(trained_annotations_fname, 'w') as handle:
        yaml.dump(all_imgs, handle)

if __name__ == '__main__':
    main(sys.argv)
