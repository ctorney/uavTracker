import cv2
import numpy as np
import sys
import os
from .bbox import BoundBox, bbox_iou
from scipy.special import expit
import hashlib, math, yaml, glob



"""
Run through the existing annotations and create a list of files without any annotations.
checked(manual) and pre(auto)

Returns a dictionary where keys are directory paths and values are lists of filenames from those directories that need to be annotated
"""
def filter_out_annotations(ss_imgs_all,some_exists,annotations_file):
    all_annot_imgs = []
    to_annot_dict = dict()

    if some_exists:
        print(f"Loading annotated images annotations from {annotations_file}")
        with open(annotations_file, 'r') as fp:
            all_annot_imgs = all_annot_imgs + yaml.safe_load(fp)

        annot_filenames = []
        for annotation_data in all_annot_imgs:
            annot_filenames.append(annotation_data['filename'])

        for ssdir, ssilist in ss_imgs_all.items():
            to_annot_dict[ssdir]=[]
            for ssi in ssilist:
                if not ssi in annot_filenames:
                    to_annot_dict[ssdir].append(ssi)
    else:
        to_annot_dict = ss_imgs_all

    return to_annot_dict


"""
Reads all subsets (defined as directories with file lists or regex) provided in a config providing a list of paths to the files.
Returns a list of strings
"""
def read_subsets(list_of_subsets,config):
    ss_imgs_all = dict()
    subsets = config['subsets']

    for tset in list_of_subsets:
        tdir = config['project_directory'] + subsets[tset]['directory'] + '/'
        tnimg = subsets[tset]['number_of_images']
        if subsets[tset]['filelist']:
            with open(tdir + subsets[tset]['filelist'], 'r') as fl:
                ss_imgs = fl.read().splitlines()
            for fname in ss_imgs:
                if not os.path.exists(tdir + fname):
                    raise Exception(f'ERROR: In {tset} file {fname} is missing but according to the filelist it should be ther')
        elif subsets[tset]['regex']:
            ss_imgs_full = glob.glob(tdir + subsets[tset]['regex'])
            ss_imgs = [os.path.basename(x) for x in ss_imgs_full]
        else:
            raise Exception(f'ER R R R O RRRR: the subset {tset} has no filelist or regex defined. The simples regex would be \'*\' to include all files in the directory')

        if len(ss_imgs) != tnimg:
            lss_imgs = len(ss_imgs)
            raise Exception(f'Error: for the subset {tset} number of images defined {tnimg} is not matching the provided filename or regex {lss_imgs}')
        ss_imgs_all[tdir] = ss_imgs

    return ss_imgs_all

"""
Read a subset of the training/testing set for the given model to train from the config. Following that prepare a temporary annotations file that contains those specified image files with their annotations only.

Returns:
Filename of temporary annotations file
"""
def read_tsets(config, model_name, c_date, list_of_subsets):
    subsets = config['subsets']

    all_annotations = config['project_directory'] + config['annotations_dir'] + '/' + config['checked_annotations_fname']
    md5check(config['checked_annotations_md5'], all_annotations)

    print(f"Loading all images annotations from {all_annotations}")
    with open(all_annotations, 'r') as fp:
        all_imgs = yaml.safe_load(fp)


    ss_imgs_all = read_subsets(list_of_subsets, config)

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

"""
Provide a checksum for an edited file
"""
def getmd5(my_file):
    if os.path.exists(my_file):
        read_file_hex = hashlib.md5(open(my_file,'rb').read()).hexdigest()
        print(f" (new) MD5 check on your file {my_file}: is {read_file_hex}, consider amending it in the config")
        return read_file_hex

"""
Return True if file exists, False if it isn't. Prints helpful information and throws and exception if the md5 checksum do not check out.
"""
def md5check(md5sum,my_file):
    if os.path.exists(my_file):
        if md5sum != "":
            read_file_hex = hashlib.md5(open(my_file,'rb').read()).hexdigest()
            if read_file_hex != md5sum:
                raise Exception(f"ERROR: md5 checksum of the provided {my_file} file doesn't match!")
            else:
                print(f"MD5 check on your file {my_file}: correct! :D ")
        else:
            print(":: Kind notice :: No md5 sum provided. Consider adding it once you have some trained weights / annotated files whatever it is that you are doing to avoid great confusion in the future")
        return True
    else:
        return False


def _sigmoid(x):
    return expit(x)

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def evaluate(model,
             generator,
             iou_threshold=0.5,
             obj_thresh=0.5,
             nms_thresh=0.45,
             net_h=416,
             net_w=416,
             save_path=None):
    """ Evaluate a given dataset using a given model.
    code originally from https://github.com/fizyr/keras-retinanet

    # Arguments
        model           : The model to evaluate.
        generator       : The generator that represents the dataset to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        obj_thresh      : The threshold used to distinguish between object and non-object
        nms_thresh      : The threshold used to determine whether two detections are duplicates
        net_h           : The height of the input image to the model, higher value results in better accuracy
        net_w           : The width of the input image to the model
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = [generator.load_image(i)]

        # make the boxes and the labels
        pred_boxes = get_yolo_boxes(model, raw_image, net_h, net_w, generator.get_anchors(), obj_thresh, nms_thresh)[0]

        score = np.array([box.get_score() for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])

        if len(pred_boxes) > 0:
            pred_boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes])
        else:
            pred_boxes = np.array([[]])

        # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes  = pred_boxes[score_sort]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = pred_boxes[pred_labels == label, :]

        annotations = generator.load_annotation(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

    # compute mAP by comparing all detections and all annotations
    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    return average_precisions

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]

            if(objectness <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row,col,b,:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height

            # last elements are class probabilities
            classes = netout[row,col,b,5:]

            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes

def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

def normalize(image):
    return image/255.

def get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
    image_h, image_w, _ = images[0].shape
    nb_images           = len(images)
    batch_input         = np.zeros((nb_images, net_h, net_w, 3))

    # preprocess the input
    for i in range(nb_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)

    # run the prediction
    batch_output = model.predict_on_batch(batch_input)
    batch_boxes  = [None]*nb_images

    for i in range(nb_images):
        yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2-j)*6:(3-j)*6] # config['model']['anchors']
            boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

        # correct the sizes of the bounding boxes
        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)

        batch_boxes[i] = boxes

    return batch_boxes

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


"""
Cut the image to a multiply of 32
"""
def makeYoloCompatible(image):
    im_height, im_width = image.shape[:2]
    new_width = 32 * math.floor(im_width / 32)
    new_height = 32 * math.floor(im_height / 32)
    im_yolo = image[0:new_height,0:new_width,:]
    return im_yolo

def pleaseCheckMyDirectories(config, data_dir):
    print('Checking if all output directories exist alright')

    groundtruths_dir = data_dir + config['groundtruths_dir']
    predictions_dir = data_dir + config['predictions_dir']
    annotations_dir = data_dir + config['annotations_dir']
    bbox_images_dir = data_dir + config['bbox_images_dir']
    tracks_dir = data_dir + config['tracks_dir']

    dirs = [
        groundtruths_dir, predictions_dir, annotations_dir, bbox_images_dir, tracks_dir
    ]

    for dir in dirs:
        if not os.path.isdir(dir):
            print('Creating ' + dir)
            os.makedirs(dir, exist_ok=True)
