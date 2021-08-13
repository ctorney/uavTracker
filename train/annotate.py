import numpy as np
import pandas as pd
import os, sys, argparse
import cv2
import yaml  #replacing pickle with yaml
import datetime  #useful for debugging in notebooks

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


def check_boxes(img_clean, bbox_list, im_width, im_height):
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
        elif event == cv2.EVENT_LBUTTONDBLCLK:
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
    #cv2.resizeWindow('image', im_width, im_height)
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
    data_dir = args.ddir[0] + '/'  #in case we forgot '/'
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    trained_annotations_fname = data_dir + config['annotations_dir'] + config['pretrained_annotations_fname']
    checked_annotations_fname = data_dir + config['annotations_dir'] +  config['checked_annotations_fname']
    im_width = config['IMAGE_W']  #size of training images for yolo
    im_height = config['IMAGE_H']

    resume = args.resume
    from_scratch = args.from_scratch

    if from_scratch:
        print('Not showing any bounding boxes. You are annotating from scratch. Remove relevant flag to use ourput of your previous training (or generic model)')

    if resume:
        print('Restoring previous session!')

    preped_images_dir = data_dir + config['preped_images_dir']

    with open(trained_annotations_fname, 'r') as fp:
        all_imgs = yaml.safe_load(fp)

    if not resume:
        new_imgs = []
    else:
        with open(checked_annotations_fname, 'r') as fp:
            new_imgs = yaml.safe_load(fp)

    for i in range(len(all_imgs)):
        fname = all_imgs[i]['filename']
        if resume:
            if any(doneimg['filename'] == fname for doneimg in new_imgs):
                continue
        img_data = {'object': []}
        img_data['filename'] = fname
        img_data['width'] = all_imgs[i]['width']
        img_data['height'] = all_imgs[i]['height']
        fullfile = preped_images_dir + fname
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
        if check_boxes(img, boxes, im_width, im_height):
            break
        for b in boxes:
            obj = {}
            if ((b[2] - b[0]) * (b[3] - b[1])) < 10:
                continue
            obj['name'] = 'aoi'
            obj['xmin'] = int(b[0])
            obj['ymin'] = int(b[1])
            obj['xmax'] = int(b[2])
            obj['ymax'] = int(b[3])
            img_data['object'] += [obj]

        new_imgs += [img_data]

    #print(all_imgs)
    with open(checked_annotations_fname, 'w') as handle:
        yaml.dump(new_imgs, handle)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate or correct annotations from non-domain specific model. To remove annotation double left click. To add one, Middle Click and move. \'c\' accepts changes and goes to the next image, \'q\' ends the session and saves files done so far (resume option is used to continue this work).',epilog='Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument('--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--ddir', '-d', required=True, nargs=1, help='Root of your data directory' )
    parser.add_argument('--resume', '-r', default=False, action='store_true',
                        help='Continue a session of annotation (it will access output file and re-start when you left off)')
    parser.add_argument('--from-scratch', '-f', default=False, action='store_true',
                        help='Annotate files from scratch without any existing bounding boxes')

    args = parser.parse_args()
    main(args)
