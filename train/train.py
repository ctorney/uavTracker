from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
import numpy as np
import yaml
import os, sys, cv2, argparse
import time
from generator import BatchGenerator
from operator import itemgetter
import random
sys.path.append('..')
from utils.utils import md5check

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models.yolo_models import get_yolo_model


def main(args):
    #Load data
    data_dir = args.ddir[0] + '/'  #in case we forgot '/'
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    #logging and debugging setup
    DEBUG = args.debug
    TEST_RUN = args.test_run
    training_phase = args.phase #if phase one, use generic pretrained weights. If phase two, use output from phase one
    print(config)

    train_dir = data_dir
    weights_dir = data_dir + config['weights_dir']
    annotations_dir = data_dir + config['annotations_dir']
    preped_images_dir = data_dir + config['preped_images_dir']

    training_setup = config['training_setup']

    list_of_train_files = annotations_dir + config[training_setup]['annotations_fname']
    LABELS = config['LABELS']
    IMAGE_H = config['IMAGE_H']
    IMAGE_W = config['IMAGE_W']
    NO_OBJECT_SCALE = config['NO_OBJECT_SCALE']
    OBJECT_SCALE = config['OBJECT_SCALE']
    COORD_SCALE = config['COORD_SCALE']
    CLASS_SCALE = config['CLASS_SCALE']
    BATCH_SIZE = config[training_phase]['BATCH_SIZE']
    EPOCHS = config[training_phase]['EPOCHS']
    LR = config[training_phase]['LR']
    trained_weights = weights_dir + config[training_phase]['trained_weights']

    if TEST_RUN:
        EPOCHS = 1
        BATCH_SIZE = 4

    #output trained weights:
    trained_weights = weights_dir + config[training_phase]['trained_weights']


    pretrained_weights = "" #weights will be initialised based on the training phase



    if training_phase == 'phase_one':
        print("Fine tuning phase 1. Training top layers.")
        pretrained_weights = weights_dir + config[training_setup]['weights']
        md5check(config[training_setup]['weights_md5'], pretrained_weights)
        model = get_yolo_model(
            IMAGE_W, IMAGE_H, num_class=len(LABELS), headtrainable=True, raw_features=False)
        print("Loading weights %s", pretrained_weights)
        model.load_weights(pretrained_weights, by_name=True)
    else:
        print(
            "Fine tuning phase 2. We retrain all layers with small learning rate"
        )
        pretrained_weights = weights_dir + config['phase_one']['trained_weights']
        model = get_yolo_model(
            IMAGE_W, IMAGE_H, num_class=len(LABELS), headtrainable=True, trainable=False, raw_features=False)
        print("Loading weights %s", pretrained_weights)
        model.load_weights(pretrained_weights)

    if DEBUG:
        print(model.summary())

    print("Loading images from %s", list_of_train_files)
    with open(list_of_train_files, 'r') as fp:
        all_imgs = yaml.safe_load(fp)

    print('Reading YaML file finished. Time to lock and load!\n')

    num_ims = len(all_imgs)
    indexes = np.arange(num_ims)
    random.shuffle(indexes)

    num_val = 0  #num_ims//10

    train_imgs = list(itemgetter(*indexes[num_val:].tolist())(all_imgs))
    train_batch = BatchGenerator(
        data_dir = data_dir,
        preped_images_dir = preped_images_dir,
        instances=train_imgs,
        labels=LABELS,
        objects=len(LABELS),
        batch_size=BATCH_SIZE,
        shuffle=True,
        jitter=0.0,
        im_dir=preped_images_dir,
        net_h=IMAGE_H,
        net_w=IMAGE_W)

    #   @tf.function
    def yolo_loss(y_true, y_pred):
        #grid factor and net factor are different at different scle levels (yolo has 3)
        grid_h = tf.shape(y_true)[1]
        grid_w = tf.shape(y_true)[2]
        grid_factor = tf.reshape(
            tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        net_h = IMAGE_H / grid_h
        net_w = IMAGE_W / grid_w
        net_factor = tf.reshape(
            tf.cast([net_w, net_h], tf.float32), [1, 1, 1, 1, 2])

        pred_box_xy = y_pred[..., 0:2]  # t_wh
        pred_box_wh = tf.math.log(y_pred[..., 2:4])  # t_wh
        pred_box_conf = tf.expand_dims(y_pred[..., 4], 4)
        pred_box_class = y_pred[..., 5:]  # adjust class probabilities
        # initialize the masks
        object_mask = tf.expand_dims(y_true[..., 4], 4)#zeroes outside object of interest mean that we only penalise incorrect box for current object.

        true_box_xy = y_true[..., 0:2]  # (sigma(t_xy) + c_xy)
        true_box_wh = tf.where(y_true[..., 2:4] > 0,
                               tf.math.log(tf.cast(y_true[..., 2:4], tf.float32)),
                               y_true[..., 2:4])
        true_box_conf = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = y_true[..., 5:]

        xy_delta = COORD_SCALE * object_mask * (pred_box_xy - true_box_xy
                                                )  #/net_factor #* xywh_scale
        wh_delta = COORD_SCALE * object_mask * (pred_box_wh - true_box_wh
                                                )  #/ net_factor #* xywh_scale

        obj_delta = OBJECT_SCALE * object_mask * (
            pred_box_conf - true_box_conf)
        no_obj_delta = NO_OBJECT_SCALE * (1 - object_mask) * pred_box_conf
        class_delta = CLASS_SCALE * object_mask * (
            pred_box_class - true_box_class)
        #tf.print(pred_box_class,summarize=-1,output_stream= "file:///home/ctorney/workspace/uavTracker/train/true_cls.out")
        #tf.print("============",output_stream= "file:///home/ctorney/workspace/uavTracker/train/true_cls.out")
        #tf.print(true_box_class,summarize=-1,output_stream= "file:///home/ctorney/workspace/uavTracker/train/true_cls.out")
        #tf.print("***************============",output_stream= "file:///home/ctorney/workspace/uavTracker/train/true_cls.out")


        loss_xy = tf.reduce_sum(tf.square(xy_delta), list(range(1, 5)))
        loss_wh = tf.reduce_sum(tf.square(wh_delta), list(range(1, 5)))
        loss_obj = tf.reduce_sum(tf.square(obj_delta), list(range(1, 5)))
        lossnobj = tf.reduce_sum(tf.square(no_obj_delta), list(range(1, 5)))
        loss_cls = tf.reduce_sum(tf.square(class_delta), list(range(1, 5)))

        loss = loss_xy + loss_wh + loss_obj + lossnobj + loss_cls
        return loss

    print('Prepared batches now we will compile')
    optimizer = Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=yolo_loss, optimizer=optimizer, metrics=['accuracy'])
    print("COMPILED")
    early_stop = EarlyStopping(
        monitor='loss', min_delta=0.001, patience=5, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(
        filepath = weights_dir + '/checkpoints',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1)

    print('Training starts.')
    start = time.time()
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    model.fit_generator(
        generator=train_batch,
        steps_per_epoch=len(train_batch),
        epochs=EPOCHS,
        verbose=1,
        callbacks=[checkpoint, early_stop, tensorboard],
        max_queue_size=3)

    model.save_weights(trained_weights)
    end = time.time()
    print('Training took ' + str(( end - start ) / 3600) + ' hours')
    print('Weights saved to ' + trained_weights)
    print("Finished! :o)")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine tune a yolo model. You have to run phase_one, or phase_one followed by phase_two. So basically you will run this program twice with different value for --phase flag. Good luck!',epilog='Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument('--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--ddir', '-d', required=True, nargs=1, help='Root of your data directory' )
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Set this flag to see a bit more wordy output')
    parser.add_argument('--test-run', default=False, action='store_true',
                        help='Set this flag to see meaningless output quicker. For instance in training it runs only for 1 epoch')
    parser.add_argument('--phase', required=True, choices=['phase_one','phase_two'],
                        help='phase_one: top layers, normal learning rate \n phase_two: all layers, small learning rate.')

    args = parser.parse_args()
    main(args)
