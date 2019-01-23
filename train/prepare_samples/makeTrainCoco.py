"""
This files is used to train ???

It uses still_images which I have NOT
"""
import numpy as np
import pandas as pd
import os,sys,glob
import cv2
import yaml
sys.path.append("../..")
sys.path.append("..")
from models.yolo_models import get_yolo_model
from utils.decoder import decode

image_dir =  '/home/ctorney/data/horses/still_images/'
train_dir = '../horse_images/'
your_weights = '../../weights/yolo-v3-coco.h5'
train_files_regex = "*.png"

train_images =  glob.glob( image_dir + train_files_regex )

max_l=100
min_l=10

width=1920
height=1080

im_size=864 #size of training imageas for yolo

nx = width//im_size
ny = height//im_size

##################################################
#im_size=416 #size of training imageas for yolo
yolov3 = get_yolo_model(im_size,im_size,trainable=False)
yolov3.load_weights(your_weights,by_name=True)


########################################
im_num=1
all_imgs = []
for imagename in train_images: 
    im = cv2.imread(imagename)
    print('processing image ' + imagename + ', ' + str(im_num) + ' of ' + str(len(train_images))  + '...')
    im_num+=1

    n_count=0
    for x in np.arange(0,width-im_size,im_size):
        for y in np.arange(0,height-im_size,im_size):
            img_data = {'object':[]}     #dictionary? key-value pair to store image data
            head, tail = os.path.split(imagename)
            noext, ext = os.path.splitext(tail)
            save_name = train_dir + '/TR_' + noext + '-' + str(n_count) + '.png'
            box_name = train_dir + '/bbox/' + noext + '-' + str(n_count) + '.png'
            img = im[y:y+im_size,x:x+im_size,:]
            cv2.imwrite(save_name, img)
            img_data['filename'] = save_name
            img_data['width'] = im_size
            img_data['height'] = im_size
            n_count+=1
            # use the yolov3 model to predict 80 classes on COCO

            # preprocess the image
            image_h, image_w, _ = img.shape
            new_image = img[:,:,::-1]/255.
            new_image = np.expand_dims(new_image, 0)

            # run the prediction
            yolos = yolov3.predict(new_image)

            boxes = decode(yolos, obj_thresh=0.005, nms_thresh=0.5)
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
with open(train_dir + '/annotations.pickle', 'w') as handle:
   yaml.dump(all_imgs, handle)

