from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
import numpy as np
import pickle
import os, sys, cv2
import time
from generator import BatchGenerator
from operator import itemgetter
import random
sys.path.append("..")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.yolo_models import get_yolo_model


FINE_TUNE=1

LABELS = ['aoi']
IMAGE_H, IMAGE_W = 864, 864
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 2.0
CLASS_SCALE      = 1.0

train_image_folder = 'horse_images/' #/home/ctorney/data/coco/train2014/'
train_annot_folder = 'train_images_1/'
valid_image_folder = train_image_folder#'/home/ctorney/data/coco/val2014/'
valid_annot_folder = train_annot_folder#'/home/ctorney/data/coco/val2014ann/'


if FINE_TUNE:
    BATCH_SIZE= 4
    EPOCHS=10
    model = get_yolo_model(IMAGE_W,IMAGE_H, num_class=1,headtrainable=True, trainable=True)
    model.load_weights('../weights/horses-yolo.h5')
else:
    BATCH_SIZE=32
    EPOCHS=500
    model = get_yolo_model(IMAGE_W,IMAGE_H, num_class=1,headtrainable=True)
    model.load_weights('../weights/yolo-v3-coco.h5', by_name=True)

### read saved pickle of parsed annotations
with open (train_image_folder + '/annotations-checked.pickle', 'rb') as fp:
    all_imgs = pickle.load(fp)


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




wt_file='../weights/horses-yolo.h5'
optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss=yolo_loss, optimizer=optimizer)

early_stop = EarlyStopping(monitor='loss', min_delta=0.001,patience=5,mode='min',verbose=1)
checkpoint = ModelCheckpoint(wt_file,monitor='loss',verbose=1,save_best_only=True,mode='min',period=1)


start = time.time()
model.fit_generator(generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = EPOCHS, 
                    verbose          = 1,
            #        validation_data  = valid_batch,
            #        validation_steps = len(valid_batch),
 #                   callbacks        = [checkpoint, early_stop],#, tensorboard], 
                    max_queue_size   = 3)
model.save_weights(wt_file)
end = time.time()
print('Training took ' + str(end - start) + ' seconds')

