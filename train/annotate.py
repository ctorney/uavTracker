import numpy as np
import pandas as pd
import os, sys, argparse
import cv2
import yaml  #replacing pickle with yaml
import datetime  #useful for debugging in notebooks
sys.path.append('../..')
sys.path.append('..')
from utils.utils import md5check, filter_out_annotations, getmd5, read_subsets, init_config

# Defining helper functions

# initialize the list of points for the rectangle bbox,
# the temporaray endpoint of the drawing rectangle
# the list of all bounding boxes of selected rois
# and boolean indicating wether drawing of mouse
# is performed or not

#WARNING: The mouse button might be the MIDDLE ONE!!!
rect_endpoint_tmp = []
rect_bbox = []

drawing = False  #this have to be False for drawing to work properly!!!


def check_boxes(img_clean, bbox_list):
    def draw_all_boxes():
        img = img_clean.copy()
        for b in bbox_list:
            #we have to make sure the the b are int. otherwise opencv thinks there is a problem with number of arguments
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),(0, 255, 0),2)
        cv2.imshow('image', img)

    # mouse callback function
    def draw_rect_roi(event, x, y, flags, param):
        # grab references to the global variables
        global rect_bbox, rect_endpoint_tmp, drawing

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that drawing is being
        # performed. set rect_endpoint_tmp empty list.
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_endpoint_tmp = []
            rect_bbox = [(x, y)]
            drawing = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # drawing operation is finished
            rect_bbox.append((x, y))
            drawing = False

            # draw a rectangle around the region of interest
            p_1, p_2 = rect_bbox

            # for bbox find upper left and bottom right points
            p_1x, p_1y = p_1
            p_2x, p_2y = p_2

            lx = min(p_1x, p_2x)
            ty = min(p_1y, p_2y)
            rx = max(p_1x, p_2x)
            by = max(p_1y, p_2y)

            # add bbox to list if both points are different
            if (lx, ty) != (rx, by):
                if abs(lx - rx) > 5:
                    if abs(ty - by) > 5:
                        bbox = [lx, ty, rx, by]
                        bbox_list.append(bbox)

        # if mouse is drawing set tmp rectangle endpoint to (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect_endpoint_tmp = [(x, y)]
        elif event == cv2.EVENT_RBUTTONDOWN:
            npbx = np.asarray(bbox_list)
            selected_box = ((x > npbx[:, 0]) & (y > npbx[:, 1]) &
                            (x < npbx[:, 2]) & (y < npbx[:, 3]))
            if np.sum(selected_box) == 1:
                bbox_list.remove(npbx[selected_box].tolist()[0])
            if np.sum(selected_box) > 1:
                potentials = npbx[selected_box]
                areas = (potentials[:, 2] - potentials[:, 0]) * (
                    potentials[:, 3] - potentials[:, 1])
                bbox_list.remove(potentials[np.argmin(areas)].tolist())
            draw_all_boxes()

    cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow('image', 1900, 1100)
    cv2.moveWindow('image', 20,20)
    cv2.setMouseCallback('image', draw_rect_roi)
    draw_all_boxes()
    # keep looping until the 'c' key is pressed
    stop = False
    while True:
        # display the image and wait for a keypress
        if not drawing:
            draw_all_boxes()
            #cv2.imshow('image', img)
        elif drawing and rect_endpoint_tmp:
            rect_cpy = img_clean.copy()
            start_point = rect_bbox[0]
            end_point_tmp = rect_endpoint_tmp[0]
            cv2.rectangle(rect_cpy, start_point, end_point_tmp, (0, 255, 0), 2)
            cv2.imshow('image', rect_cpy)

        key = cv2.waitKey(1)  #& 0xFF
        # if the 'c' key is pressed, break from the loop
        if key == ord('c'):
            break
        if key == ord('q'):
            stop = True
            break
    # close all open windows
    cv2.destroyAllWindows()
    #cv2.waitKey(1)
    return stop


def main(args):
    #Load data
    config = init_config(args)
    data_dir = config['project_directory']

    resume = config['args_resume']
    from_scratch = config['args_from_scratch']

    checked_annotations = os.path.join(config['project_directory'],config['annotations_dir'],config['checked_annotations_fname'])
    autogen_annotations = os.path.join(config['project_directory'],config['annotations_dir'],config['autogen_annotations_fname'])
    some_checked = md5check(config['checked_annotations_md5'], checked_annotations)
    some_autogen = md5check(config['autogen_annotations_md5'], autogen_annotations)

    obj_label = config['common']['LABELS'][0]



    if resume:
        print('Restoring previous session!')

    if some_autogen:
        with open(autogen_annotations, 'r') as fp:
            all_imgs = yaml.safe_load(fp)
    else:
        print('no automatically generate annotations provided, will have to annotate from scratch')
        from_scratch = True
        raise Exception('BUG, currently you have to run prepTrain.py first and generate automatic annotations in order to annotate from scratch. Sorry!')

    if from_scratch:
        print('Not showing any bounding boxes. You are annotating from scratch. Remove relevant flag to use ourput of your previous training (or generic model)')

    #Get a list of all the files that we need annotations for to do all those trainings and testings in the config.
    subsets = config['subsets']
    list_of_subsets = [x for x in subsets]
    ss_imgs_all = read_subsets(list_of_subsets,config)

    #run through the existing annotations and create a list of files without any annotations
    to_annot_imgs = filter_out_annotations(ss_imgs_all,some_checked,checked_annotations)

    if some_checked:
        with open(checked_annotations, 'r') as fp:
            new_imgs = yaml.safe_load(fp)
    else:
        new_imgs = []

    annot_filenames = []
    for annotation_data in all_imgs:
        annot_filenames.append(annotation_data['filename'])

    for ssdir, sslist in to_annot_imgs.items():
        for fname in sslist:
            fullname = os.path.join(ssdir, fname)
            if resume:
                if any(doneimg['filename'] == fname for doneimg in new_imgs):
                    continue
            i = annot_filenames.index(fname)

            img_data = {'object': []}
            img_data['filename'] = fname
            img_data['width'] = all_imgs[i]['width']
            img_data['height'] = all_imgs[i]['height']
            fullfile = fullname
            sys.stdout.write('\r')
            sys.stdout.write(fullfile + ", " + str(i) + ' of ' + str(
                len(all_imgs)) + " \n=====================================\n")
            sys.stdout.flush()

            boxes = []
            if not from_scratch:
                for obj in all_imgs[i]['object']:
                    boxes.append(
                        [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])

            #do box processing
            img = cv2.imread(fullfile)
            if check_boxes(img, boxes):
                break
            for b in boxes:
                obj = {}
                if ((b[2] - b[0]) * (b[3] - b[1])) < 10:
                    continue
                obj['name'] = obj_label
                obj['xmin'] = int(b[0])
                obj['ymin'] = int(b[1])
                obj['xmax'] = int(b[2])
                obj['ymax'] = int(b[3])
                img_data['object'] += [obj]

            new_imgs += [img_data]

    #print(all_imgs)
    with open(checked_annotations, 'w') as handle:
        yaml.dump(new_imgs, handle)

    getmd5(checked_annotations)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate or correct annotations from non-domain specific model. To remove annotation double left click. To add one, Middle Click and move. \'c\' accepts changes and goes to the next image, \'q\' ends the session and saves files done so far (resume option is used to continue this work).',epilog='Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument('--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--resume', '-r', default=False, action='store_true',
                        help='Continue a session of annotation (it will access output file and re-start when you left off)')
    parser.add_argument('--from-scratch', '-f', default=False, action='store_true',
                        help='Annotate files from scratch without any existing bounding boxes')

    args = parser.parse_args()
    main(args)
