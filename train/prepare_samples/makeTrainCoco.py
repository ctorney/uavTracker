'''
This files is used to train ???

It uses still_images which I have NOT
'''
import numpy as np
import pandas as pd
import os,sys,glob
import cv2
import yaml
sys.path.append('../..')
sys.path.append('..')
from models.yolo_models import get_yolo_model
from utils.decoder import decode

def main(argv):
    if(len(sys.argv) != 3):
        print('Usage ./makeTrainCoco.py [root_dir] [config.yml]')
        sys.exit(1)
    #Load data
    root_dir = argv[1]  + '/' #in case we forgot
    print('Opening file' + root_dir + argv[2])
    with open(root_dir + argv[2], 'r') as configfile:
        config = yaml.safe_load(configfile)

    image_dir = root_dir + config['data_dir']
    train_dir = root_dir + config['data_dir']
    weights_dir = root_dir + config['weights_dir']
    your_weights = weights_dir + config['generic_weights']
    annotations_file = train_dir + config['untrained_annotations_fname']
    train_files_regex = config['generic_train_files_regex']

    train_images =  glob.glob( image_dir + train_files_regex )

    max_l=100
    min_l=10

    im_size=config['IMAGE_H'] #size of training imageas for yolo

    ##################################################
    yolov3 = get_yolo_model(im_size,im_size,trainable=False)
    yolov3.load_weights(your_weights,by_name=True)


    ########################################
    im_num=1
    all_imgs = []
    for imagename in train_images: 
        im = cv2.imread(imagename)
        print('processing image ' + imagename + ', ' + str(im_num) + ' of ' + str(len(train_images))  + '...')
        height, width = im.shape[:2]
        im_num+=1
        n_count=0

        if (width-im_size < 0 or height-im_size < 0):
            print("Image too small for a defined yolo input size, adding a black stripe")
            new_height = height if height >= im_size else im_size
            new_width = width if width >= im_size else im_size
            enlarged_im = np.zeros((new_height, new_width,3), np.uint8)
            enlarged_im[:height,:width] = im.copy()
            im = enlarged_im.copy()
            height, width = im.shape[:2]


        for x in np.arange(0,1+width-im_size,im_size):#'1+' added to allow case when image has exactly size of one window
            for y in np.arange(0,1+height-im_size,im_size):
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
    print('Saving data to ' + annotations_file)
    with open(annotations_file, 'w') as handle:
        yaml.dump(all_imgs, handle)

    print('Finished! :o)')
if __name__ == '__main__':
    main(sys.argv)
