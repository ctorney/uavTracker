from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
import numpy as np
import yaml
import os, sys, cv2
import time
from generator import BatchGenerator
from operator import itemgetter
import random
sys.path.append('..')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.yolo_models import get_yolo_model

def main(argv):
    if(len(sys.argv) != 3):
        print('Usage ./train.py [root_dir] [config.yml]')
        sys.exit(1)
    #Load data
    root_dir = argv[1]  + '/' #in case we forgot
    print('Opening file' + root_dir + argv[2])
    with open(root_dir + argv[2], 'r') as configfile:
        config = yaml.safe_load(configfile)

    image_dir = root_dir + config['data_dir']
    train_dir = root_dir + config['data_dir']
    train_image_folder = root_dir + config['data_dir']
    weights_dir = root_dir + config['weights_dir']
    your_weights = weights_dir + config['specific_weights']
    generic_weights = weights_dir + config['generic_weights']
    trained_weights = weights_dir + config['trained_weights']
    list_of_train_file = train_dir + config['checked_annotations_fname']
    list_of_train_files = '/annotations-checked.yml'
    train_files_regex = config['generic_train_files_regex']

    FINE_TUNE = config['FINE_TUNE']
    TEST_RUN = config['TEST_RUN']
    LABELS = config['LABELS']
    IMAGE_H = config['IMAGE_H']
    IMAGE_W = config['IMAGE_W']
    NO_OBJECT_SCALE  = config['NO_OBJECT_SCALE']
    OBJECT_SCALE  = config['OBJECT_SCALE']
    COORD_SCALE  = config['COORD_SCALE']
    CLASS_SCALE  = config['CLASS_SCALE']

    valid_image_folder = train_image_folder
    valid_annot_folder = train_image_folder

    if FINE_TUNE:
        BATCH_SIZE= 4
        if TEST_RUN:
            EPOCHS=1
        else:
            EPOCHS=100
        model = get_yolo_model(IMAGE_W,IMAGE_H, num_class=1,headtrainable=True, trainable=True)
        model.load_weights(your_weights)
    else:
        BATCH_SIZE=32
        EPOCHS=500
        model = get_yolo_model(IMAGE_W,IMAGE_H, num_class=1,headtrainable=True)
        model.load_weights(generic_weights, by_name=True)

    ### read saved pickle of parsed annotations
    with open (train_image_folder + list_of_train_files, 'r') as fp:
        all_imgs = yaml.load(fp)

    print('Reading YaML file finished. Time to luck and load!\n')

    num_ims = len(all_imgs)
    indexes = np.arange(num_ims)
    random.shuffle(indexes)

    num_val = 0#num_ims//10

    #valid_imgs = list(itemgetter(*indexes[:num_val].tolist())(all_imgs))
    train_imgs = list(itemgetter(*indexes[num_val:].tolist())(all_imgs))
    train_batch = BatchGenerator(instances= train_imgs,labels= LABELS,batch_size= BATCH_SIZE,shuffle= True,jitter= 0.0,im_dir= train_image_folder)
    #valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)


    def yolo_loss(y_true, y_pred):
        # compute grid factor and net factor
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]

        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

        net_h       = IMAGE_H/grid_h
        net_w       = IMAGE_W/grid_w
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])

        pred_box_xy    = y_pred[..., 0:2]                                                       # t_wh
        pred_box_wh    = tf.log(y_pred[..., 2:4])                                                       # t_wh
        pred_box_conf  = tf.expand_dims(y_pred[..., 4], 4)
        pred_box_class = y_pred[..., 5:]                                            # adjust class probabilities      
        # initialize the masks
        object_mask     = tf.expand_dims(y_true[..., 4], 4)

        true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
        true_box_wh    = tf.where(y_true[...,2:4]>0, tf.log(tf.cast(y_true[..., 2:4],tf.float32)), y_true[...,2:4])
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = y_true[..., 5:]         

        xy_delta    = COORD_SCALE * object_mask   * (pred_box_xy-true_box_xy) #/net_factor #* xywh_scale
        wh_delta    = COORD_SCALE * object_mask   * (pred_box_wh-true_box_wh) #/ net_factor #* xywh_scale

        obj_delta  = OBJECT_SCALE * object_mask * (pred_box_conf-true_box_conf)  
        no_obj_delta = NO_OBJECT_SCALE * (1-object_mask) * pred_box_conf
        class_delta = CLASS_SCALE * object_mask * (pred_box_class-true_box_class)

        loss_xy = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5))) 
        loss_wh = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5))) 
        loss_obj= tf.reduce_sum(tf.square(obj_delta),     list(range(1,5))) 
        lossnobj= tf.reduce_sum(tf.square(no_obj_delta),     list(range(1,5))) 
        loss_cls= tf.reduce_sum(tf.square(class_delta),    list(range(1,5)))

        loss = loss_xy + loss_wh + loss_obj + lossnobj + loss_cls
        return loss



    print('Prepared bathes now we will load weights')
    wt_file=your_weights
    optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=yolo_loss, optimizer=optimizer)

    early_stop = EarlyStopping(monitor='loss', min_delta=0.001,patience=5,mode='min',verbose=1)
    checkpoint = ModelCheckpoint(wt_file,monitor='loss',verbose=1,save_best_only=True,mode='min',period=1)

    print('Training starts.')
    start = time.time()
    model.fit_generator(generator        = train_batch, 
                        steps_per_epoch  = len(train_batch), 
                        epochs           = EPOCHS, 
                        verbose          = 1,
                #        validation_data  = valid_batch,
                #        validation_steps = len(valid_batch),
                        callbacks        = [checkpoint, early_stop],#, tensorboard], 
                        max_queue_size   = 3)
    model.save_weights(trained_weights)
    end = time.time()
    print('Training took ' + str(end - start) + ' seconds')
    print('Weights saved to ' + trained_weights)
    print("Finished! :o)")

if __name__ == '__main__':
    main(sys.argv)
