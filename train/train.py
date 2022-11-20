from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tqdm.keras import TqdmCallback
import tensorflow as tf
import numpy as np
import yaml
import os, sys, cv2, argparse, glob
import time
from generator import BatchGenerator
from operator import itemgetter
import random
sys.path.append('..')
from utils.utils import md5check
import datetime as dt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' ##TODO what does that do?

LABELS = ''
IMAGE_H = ''
IMAGE_W = ''
NO_OBJECT_SCALE = ''
OBJECT_SCALE = ''
COORD_SCALE = ''
CLASS_SCALE = ''

from models.yolo_models import get_yolo_model

def read_tsets(config, model_name, c_date, list_of_subsets):
    subsets = config['subsets']

    all_annotations = config['project_directory'] + config['annotations_dir'] + '/' + config['checked_annotations_fname']
    md5check(config['checked_annotations_md5'], all_annotations)

    print(f"Loading all images annotations from {all_annotations}")
    with open(all_annotations, 'r') as fp:
        all_imgs = yaml.safe_load(fp)

    ss_imgs_all = dict()
    for tset in list_of_subsets:
        tdir = config['project_directory'] + subsets[tset]['directory'] + '/'
        tnimg = subsets[tset]['number_of_images']
        if subsets[tset]['filelist']:
            with open(tdir + subsets[tset]['filelist'], 'r') as fl:
                ss_imgs = fl.read().splitlines()
            for fname in ss_imgs:
                if not os.exists(tdir + fname):
                    raise Exception(f'ERROR: In {tset} file {fname} is missing but according to the filelist it should be ther')
        elif subsets[tset]['regex']:
            ss_imgs_full = glob.glob(tdir + subsets[tset]['regex'])
            ss_imgs = [os.path.basename(x) for x in ss_imgs_full]
        else:
            raise Exception(f'ER R R R O RRRR: the subset {tset} has no filelist or regex defined. The simples regex would be \'*\' to include all files in the directory')

        if len(ss_imgs) != tnimg:
            raise Exception(f'Error: for the subset {tset} number of images defined is not matching the provided filename or regex')
        ss_imgs_all[tdir] = ss_imgs


    #Re-write existing annotations but with a subset path in a filename into a new file that will be used for this training
    annotations_subset = []
    for annotation_data in all_imgs:
        fname = annotation_data['filename']
        #if fname is in our set of filenames. But what if filenames are repeating?
        addit = False
        for ssdir, ssi in ss_imgs_all.items():
            if fname in ssi:
                annotation_data['filename'] = ssdir + fname
                addit = True
                break
        if addit:
            annotations_subset += [annotation_data]


    current_annotations = config['project_directory'] + config['annotations_dir'] + 'annotations_' + model_name + '_' + c_date + '.yml'
    with open(current_annotations, 'w') as handle:
        yaml.dump(annotations_subset, handle)

    return current_annotations



def run_full_training(model_name, config, data_dir, c_date, DEBUG, TEST_RUN):

    c_model = config['models'][model_name]
    project_name = config['project_name']

    train_dir = data_dir
    weights_dir = data_dir + config['weights_dir']
    annotations_dir = data_dir + config['annotations_dir']

    #The following are *I KID YOU NOT* sort of global variables. tf function yolo_loss has only two arguemts that are explicit x and y. the rest must be global...
    LABELS = config['common']['LABELS']
    IMAGE_H = config['common']['IMAGE_H']
    IMAGE_W = config['common']['IMAGE_W']
    NO_OBJECT_SCALE = config['common']['NO_OBJECT_SCALE']
    OBJECT_SCALE = config['common']['OBJECT_SCALE']
    COORD_SCALE = config['common']['COORD_SCALE']
    CLASS_SCALE = config['common']['CLASS_SCALE']

    #   @tf.function
    # Jesus Christ, this function has to be defined INSIDE another function so it can set those variables in bold which cannot be passed as normal variables becasue.. just because.
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

    #We are having potentially one large annotations file and physically separated subsets to be able to review them easily. But for the old code to work easily we need to have only the files we use in one annotation file. For this we are dynamically generate this file by reading in our existing annotation file and information about structure of the training sets we want to use
    list_of_train_files = read_tsets(config,model_name,c_date,c_model['training_sets'])

    trained_weights = dict()
    trained_weights['phase_one'] = f'{weights_dir}{project_name}_{model_name}_phase_one_{c_date}.h5'
    trained_weights['phase_two'] = f'{weights_dir}{project_name}_{model_name}_phase_two_{c_date}.h5'

    for iii in range(c_model['phases']):
        training_phase = 'phase_one' if iii == 0 else 'phase_two'

        #output trained weights:
        #TODO create a md5 of parameters? write down parameters used into results as well, as a whole run will result in writing to results
        BATCH_SIZE = c_model[training_phase]['BATCH_SIZE']
        EPOCHS = c_model[training_phase]['EPOCHS']
        LR = c_model[training_phase]['LR']
        if TEST_RUN:
            EPOCHS = 1
            BATCH_SIZE = 4

        pretrained_weights = "" #weights will be initialised based on the training phase, here just defining the variable

        if training_phase == 'phase_one':
            print("Fine tuning phase 1. Training top layers.")
            pretrained_weights = weights_dir + c_model['pretrained_weights']
            md5check(c_model['pretrained_weights_md5'], pretrained_weights)
            model = get_yolo_model(
                IMAGE_W, IMAGE_H, num_class=len(LABELS), headtrainable=True, raw_features=False)
            print("Loading weights %s", pretrained_weights)
            model.load_weights(pretrained_weights, by_name=True)
        else:
            print(
                "Fine tuning phase 2. We retrain all layers with small learning rate"
            )
            pretrained_weights = trained_weights['phase_one']
            model = get_yolo_model(
                IMAGE_W, IMAGE_H, num_class=len(LABELS), headtrainable=True, trainable=False, raw_features=False)
            print("Loading weights %s", pretrained_weights)
            model.load_weights(pretrained_weights)

        if DEBUG:
            print(model.summary())

        ##loading images
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
            preped_images_dir = data_dir,
            instances=train_imgs,
            labels=LABELS,
            objects=len(LABELS),
            batch_size=BATCH_SIZE,
            shuffle=True,
            jitter=0.0,
            im_dir=data_dir,
            net_h=IMAGE_H,
            net_w=IMAGE_W)


        print('Prepared batches now we will compile')
        optimizer = Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss=yolo_loss, optimizer=optimizer, metrics=['accuracy'])
        print("COMPILED")
        early_stop = EarlyStopping(
            monitor='loss', min_delta=0.001, patience=10, mode='min', verbose=1)
        checkpoint = ModelCheckpoint(
            filepath = weights_dir + '/checkpoints',
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min',
            save_freq='epoch')

        print('Training starts.')
        start = time.time()
        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

        model.fit(
            train_batch,
            steps_per_epoch=len(train_batch),
            epochs=EPOCHS,
            verbose=0,
            callbacks=[checkpoint, early_stop, tensorboard, TqdmCallback(verbose=1)],
            max_queue_size=3)

        model.save_weights(trained_weights[training_phase])
        end = time.time()
        print('Training of ' + training_phase + ' took ' + str(( end - start ) / 3600) + ' hours')
        print('Weights saved to ' + trained_weights[training_phase])
    print(f"Finished! with {model_name} :o)")

def main(args):
    c_date = dt.datetime.now().strftime('%Y%b%d') #we want all models in this run to have the same date
    #Load data
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    data_dir = config['project_directory'] + '/'

    #logging and debugging setup
    DEBUG = args.debug
    TEST_RUN = args.test_run
    n_models_to_train = len(config['models'].keys())
    print('Will train {n_models_to_train} different models for this experiment now')

    for model in config['models'].keys():
        run_full_training(model, config, data_dir, c_date, DEBUG, TEST_RUN)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine tune a yolo model. Run both phases of training depending how specified',epilog='Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument('--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Set this flag to see a bit more wordy output')
    parser.add_argument('--test-run', default=False, action='store_true',
                        help='Set this flag to see meaningless output quicker. For instance in training it runs only for 1 epoch')

    args = parser.parse_args()
    main(args)
