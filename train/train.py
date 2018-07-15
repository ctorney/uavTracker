
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf
#import imgaug as ia
#from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, sys, cv2
import time
from generator import BatchGenerator
sys.path.append("..")

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.yolo_models import get_yolo_model


FINE_TUNE=0


LABELS = ['aoi']
IMAGE_H, IMAGE_W = 864, 864
GRID_H,  GRID_W  = 27, 27
# each cell is going to be 32x32
BOX              = 3
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3#0.5 # this must be for showing the object - should be lower??
NMS_THRESHOLD    = 0.3#0.45 # non max suppression - what does this do?

# this is the width/height of the anchor boxes - this will be 2,2 for all 5 - maybe - might be better to use pretrained
#ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ignore_thresh=0.8
# scales - for training maybe?? no idea
# all seem to be in the custom loss function - some method to weight the loss
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 2.0
CLASS_SCALE      = 1.0

if FINE_TUNE:
    BATCH_SIZE       = 4
else:
    BATCH_SIZE       = 4
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50
print(len(LABELS))


#true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))


train_image_folder = 'train_images_1/' #/home/ctorney/data/coco/train2014/'
train_annot_folder = 'train_images_1/'
valid_image_folder = train_image_folder#'/home/ctorney/data/coco/val2014/'
valid_annot_folder = train_annot_folder#'/home/ctorney/data/coco/val2014ann/'


model = get_yolo_model(IMAGE_W,IMAGE_H, num_class=1,trainable=True)

if FINE_TUNE:
    model.load_weights('../weights/wb-yolo.h5')
else:
    model.load_weights('../weights/yolo-v3-coco.h5', by_name=True)

print(model.summary())

def yolo_loss(y_true, y_pred):
    loss = tf.sqrt(tf.reduce_sum(y_pred[0]))
    # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
    loss = tf.Print(loss, [tf.shape(y_true)], message='prereshape  \t\t', summarize=1000)
    #return loss
 #   y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
 #   y_true = tf.reshape(y_true, tf.concat([tf.shape(y_true)[:3], tf.constant([3, -1])], axis=0))
    loss = tf.Print(loss, [tf.shape(y_pred)], message='shape  \t\t', summarize=1000)
 #   return loss

    # compute grid factor and net factor
    grid_h      = tf.shape(y_true)[1]
    grid_w      = tf.shape(y_true)[2]
    
    #net_factor = float32(IMAGE_H/grid_h) # number of pixels in grid cell

    # the variable to keep track of number of batches processed
  #  batch_seen = tf.Variable(0.)        

    grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

    net_h       = IMAGE_H/grid_h
    net_w       = IMAGE_W/grid_w
    net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
    
    """
    Adjust prediction
    """

 #   cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)))
 #   cell_y = tf.transpose(cell_x, (0,2,1,3,4))
 #   cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [BATCH_SIZE, 1, 1, 3, 1])
 #   pred_box_xy    = (cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy


    pred_box_xy    = y_pred[..., 0:2]                                                       # t_wh
    pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
    pred_box_conf  = tf.expand_dims(y_pred[..., 4], 4)
    pred_box_class = y_pred[..., 5:]                                            # adjust class probabilities      
    # initialize the masks
    object_mask     = tf.expand_dims(y_true[..., 4], 4)

    """
    Adjust ground truth
    """
    true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
    true_box_wh    = y_true[..., 2:4] # t_wh
    true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
    true_box_class = y_true[..., 5:]         

    #anc = tf.constant(ANCHORS, dtype='float', shape=[1,1,1,3,2])
    #true_xy = tf.expand_dims(true_box_xy / grid_factor,4)
    #true_wh = tf.expand_dims(tf.exp(true_box_wh) * anc / net_factor,4)
    """
    Compare each predicted box to all true boxes
    """        
    # initially, drag all objectness of all boxes to 0
    #conf_delta  = pred_box_conf 

    # then, ignore the boxes which have good overlap with some true box
    #true_xy = true_boxes[..., 0:2] / grid_factor
    #true_wh = true_boxes[..., 2:4] / net_factor
    #for b in range(BOX):

    #true_xy = y_true[..., 0:2]
    #true_wh = y_true[..., 2:4]
 #   ya = y_true[...,0,:]
    #ya = y_true[y_true[...,0,4],0,1]
    #ya = y_true[y_true[...,0,4]==1,0,1]
  #  ya = y_true[...,4]==1
   # ya=tf.where(y_true[...,4],true_wh)
    
  #  true_wh_half = true_wh / 2.
  #  true_mins    = true_xy - true_wh_half
  #  true_maxes   = true_xy + true_wh_half
    
    #pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
 #   pred_xy = pred_box_xy / grid_factor
 #   pred_wh = tf.exp(pred_box_wh) * anc / net_factor
    #pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * anc / net_factor, 4)
    
    #pred_wh_half = pred_wh / 2.
    #pred_mins    = pred_xy - pred_wh_half
    #pred_maxes   = pred_xy + pred_wh_half    

 #   loss = tf.Print(loss, [tf.shape(pred_maxes)], message='shape \t\t', summarize=1000)
 #   loss = tf.Print(loss, [tf.shape(true_maxes)], message='shape \t\t', summarize=1000)
    #intersect_mins  = tf.maximum(pred_mins,  true_mins)
    #intersect_maxes = tf.minimum(pred_maxes, true_maxes)

    #intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    #intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
  #  true_areas = true_wh[..., 0] * true_wh[..., 1]
  #  pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

 #   union_areas = pred_areas + true_areas - intersect_areas
#    iou_scores  = tf.truediv(intersect_areas, union_areas)

   # best_ious   = tf.reduce_max(iou_scores, axis=-1)        
#    conf_delta = tf.where(best_ious<ignore_thresh,conf_delta[...,0], tf.zeros_like(conf_delta[...,0]))
#    conf_delta  = tf.expand_dims(conf_delta, 4)
#    d_delta = tf.expand_dims(tf.to_float(best_ious < ignore_thresh), 4)
    #d_delta =tf.to_float(best_ious < ignore_thresh)
 #   d_delta = tf.expand_dims(tf.to_float(true_wh < ignore_thresh), 4)
    #d_delta = tf.to_float(best_ious < ignore_thresh)
  #  loss = tf.Print(loss, [tf.shape(best_ious<ignore_thresh)], message='shape \t\t', summarize=1000)
 #   loss = tf.Print(loss, [tf.shape(conf_delta)], message='shape \t\t', summarize=1000)
 #   return loss
#    conf_delta = d_delta
   # conf_delta *= tf.expand_dims(tf.to_float(best_ious < ignore_thresh), 4)
#    return loss

    """
    Compute some online statistics
    """            

#    true_wh_half = true_wh / 2.
#    true_mins    = true_xy - true_wh_half
#    true_maxes   = true_xy + true_wh_half
#
#    pred_xy = pred_box_xy / grid_factor
#    pred_wh = tf.exp(pred_box_wh) * anc / net_factor 
#    
#    pred_wh_half = pred_wh / 2.
#    pred_mins    = pred_xy - pred_wh_half
#    pred_maxes   = pred_xy + pred_wh_half     # 
#
#    intersect_mins  = tf.maximum(pred_mins,  true_mins)
#    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
#    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
#    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
#    
#    true_areas = true_wh[..., 0] * true_wh[..., 1]
#    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
#
#    union_areas = pred_areas + true_areas - intersect_areas
#    iou_scores  = tf.truediv(intersect_areas, union_areas)
#    iou_scores  = object_mask * tf.expand_dims(iou_scores, -1)
#    
#    count       = tf.reduce_sum(object_mask)
#    count_noobj = tf.reduce_sum(1-object_mask)
#    detect_mask = tf.to_float(pred_box_conf >= 0.5)
#    class_mask  = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), tf.argmax(true_box_class, -1))), 4)
#    recall50    = tf.to_float(iou_scores >= 0.5 ) * detect_mask
#    recall75    = tf.to_float(iou_scores >= 0.75) * detect_mask
#    recall50_c  = tf.reduce_sum(recall50  * class_mask) / (count + 1e-3)
#    recall75_c  = tf.reduce_sum(recall75  * class_mask) / (count + 1e-3)    
#    recall50    = tf.reduce_sum(recall50) / (count + 1e-3)
#    recall75    = tf.reduce_sum(recall75) / (count + 1e-3)        
#    avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
#    avg_obj     = tf.reduce_sum(detect_mask  * object_mask)  / (count + 1e-3)
#    avg_noobj   = tf.reduce_sum(detect_mask  * (1-object_mask))  / (count_noobj + 1e-3)
#    avg_cat     = tf.reduce_sum(pred_box_class * true_box_class) / (count + 1e-3) 
#
    """
    Warm-up training
    """
    
 #   true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1), 
  #                        lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
   #                                true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), 
    #                               tf.ones_like(object_mask)],
      #                    lambda: [true_box_xy, 
       #                            true_box_wh,
        #                           object_mask])
    """
    Compare each true box to all anchor boxes
    """      
#    xywh_scale = true_box_wh#) * anc / net_factor
 #   xywh_scale = tf.expand_dims(2 - xywh_scale[..., 0] * xywh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

    xy_delta    = COORD_SCALE * object_mask   * (pred_box_xy-true_box_xy) /net_factor #* xywh_scale
    wh_delta    = COORD_SCALE * object_mask   * (pred_box_wh-true_box_wh) / net_factor #* xywh_scale
 #   return loss
    #loss = tf.Print(loss, [tf.shape(object_mask)], message='shape \t\t', summarize=1000)
    #loss = tf.Print(loss, [tf.shape(pred_box_conf)], message='shape \t\t', summarize=1000)
    #loss = tf.Print(loss, [tf.shape(true_box_conf)], message='shape \t\t', summarize=1000)
    #conf_delta  = (object_mask * (pred_box_conf-true_box_conf) * 5) + ((1-object_mask) * conf_delta)
    obj_delta  = OBJECT_SCALE * object_mask * (pred_box_conf-true_box_conf)  
    no_obj_delta = NO_OBJECT_SCALE * (1-object_mask) * pred_box_conf
    class_delta = CLASS_SCALE * object_mask * (pred_box_class-true_box_class)
    #class_delta = object_mask * (pred_box_conf-true_box_conf)

 #   closs =       tf.reduce_sum(tf.square(conf_delta),     list(range(1,5))) #+ \
      #     tf.reduce_sum(tf.square(class_delta),    list(range(1,5)))
    loss_xy = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5))) 
    loss_wh = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5))) 
    loss_obj= tf.reduce_sum(tf.square(obj_delta),     list(range(1,5))) 
    lossnobj= tf.reduce_sum(tf.square(no_obj_delta),     list(range(1,5))) 
    loss_cls= tf.reduce_sum(tf.square(class_delta),    list(range(1,5)))

    loss = loss_xy + loss_wh + loss_obj + lossnobj + loss_cls
    #loss = loss_cls
 #   loss = tf.Print(loss, [loss_xy], message='\n\n avg_xy \t', summarize=1000)
 #   loss = tf.Print(loss, [loss_wh], message='\n\n avg_wh \t', summarize=1000)
 #   loss = tf.Print(loss, [tf.shape(xy_delta)], message='xy delta shape \t\t', summarize=1000)
 #   loss = tf.Print(loss, [loss_obj], message='\n\n avg_obj \t', summarize=1000)
 #   loss = tf.Print(loss, [lossnobj], message='\n\n avg_nobj \t', summarize=1000)
 #   loss = tf.Print(loss, [loss_cls], message='\n\n avg_cls \t', summarize=1000)
 #   noloss = tf.reduce_sum(tf.square(no_obj_delta),     list(range(1,5))) 
 #   loss = tf.Print(loss, [noloss], message='shape \t\t', summarize=1000)
 #   loss = tf.Print(loss, [tf.shape(closs)], message='conshape \t\t', summarize=1000)
    return loss

    #loss = tf.cond(tf.less(batch_seen, self.warmup_batches+1), # add 10 to the loss if this is the warmup stage
    #              lambda: loss + 10,
    #              lambda: loss)
    
 #   loss = tf.Print(loss, [avg_obj], message='\n\n avg_obj \t', summarize=1000)
 #   loss = tf.Print(loss, [avg_noobj], message='\n avg_noobj \t\n', summarize=1000)
#    loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
#    loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
#    loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
#    loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)   
#    loss = tf.Print(loss, [grid_h, recall50_c], message='recall50_cat \t', summarize=1000)
#    loss = tf.Print(loss, [grid_h, recall75_c], message='recall75_Cat \t', summarize=1000)          
#    loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)     
#    loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss)],  message='loss: \t',   summarize=1000)   
#



from operator import itemgetter
import random

### read saved pickle of parsed annotations
with open ('train_images_1/annotations-checked.pickle', 'rb') as fp:
    all_imgs = pickle.load(fp)

num_ims = len(all_imgs)
indexes = np.arange(num_ims)
#random.shuffle(indexes)

num_val = 0#num_ims//10

#valid_imgs = list(itemgetter(*indexes[:num_val].tolist())(all_imgs))
train_imgs = list(itemgetter(*indexes[num_val:].tolist())(all_imgs))

def normalize(image):
    image = image / 255.
    return image

train_batch = BatchGenerator(
        instances           = train_imgs, 
        labels              = LABELS,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = 1000,
        batch_size          = BATCH_SIZE,
        min_net_size        = IMAGE_H,
        max_net_size        = IMAGE_H,   
        shuffle             = False, 
        jitter              = 0.0, 
        norm                = normalize
)
print(len(train_batch))
#sys.exit('bye')
#train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize, jitter=False)
#valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)


# In[104]:




# **Setup a few callbacks and start the training**

# In[105]:



if FINE_TUNE:
    optimizer = Adam(lr=0.5e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    EPOCHS=200
else:
    optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    EPOCHS=1250
#  optimizer = SGD(lr=1e-5, decay=0.0005, momentum=0.9)
model.compile(loss=yolo_loss, optimizer=optimizer)
wt_file='../weights/balloon-yolo.h5'
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
early_stop = EarlyStopping(monitor='loss', 
                           min_delta=0.001, 
                           patience=5, 
                           mode='min', 
                           verbose=1)

checkpoint = ModelCheckpoint(wt_file, 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)


start = time.time()
model.fit_generator(generator        = train_batch, 
                    steps_per_epoch  = 10*len(train_batch), 
                    epochs           = EPOCHS, 
                    verbose          = 1,
            #        validation_data  = valid_batch,
            #        validation_steps = len(valid_batch),
                    callbacks        = [checkpoint, early_stop],#, tensorboard], 
                    max_queue_size   = 3)
end = time.time()
print('Training took ' + str(end - start) + ' seconds')
model.save_weights(wt_file)

