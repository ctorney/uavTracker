import os, sys, glob, argparse
import csv
import cv2
import yaml
import numpy as np

import time


def main(args):
    #Load data
    data_dir = args.ddir[0] + '/'  #in case we forgot '/'
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    tracking_setup = config["tracking_setup"]
    np.set_printoptions(suppress=True)

    data_dir = root_dir + config['movie_dir']

    videos_name_regex_short = config[tracking_setup]['videos_name_regex']
    videos_list = data_dir + config[tracking_setup]['videos_list']
    videos_info = []  #this list will be saved into videos_list file

    im_width = config['IMAGE_W']  #size of training imageas for yolo
    im_height = config['IMAGE_H']

    filelist = glob.glob(data_dir + videos_name_regex_short)
    print(filelist)
    prefix_pos = len(data_dir)
    scalefact = 4.0

    for input_file in filelist:

        input_file_short = input_file[prefix_pos:]
        direct, ext = os.path.split(input_file_short)
        noext, _ = os.path.splitext(ext)
        tr_file = data_dir + '/tracks/' + noext + '_MAT.npy'
        tr_file_short = '/tracks/' + noext + '_MAT.npy'
        if os.path.isfile(tr_file):
            print("transformation file already exists")
            continue
        print(input_file)

        ##########################################################################
        ##          open the video file for inputs
        ##########################################################################
        cap = cv2.VideoCapture(input_file)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ##########################################################################
        ##          corrections for camera motion
        ##########################################################################
        warp_mode = cv2.MOTION_EUCLIDEAN
        number_of_iterations = 20
        # Specify the threshold of the increment in the correlation coefficient between two iterations
        termination_eps = 1e-6
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    number_of_iterations, termination_eps)

        im1_gray = np.array([])
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        full_warp = np.eye(3, 3, dtype=np.float32)

        save_warp = np.zeros((
            nframes,
            3,
            3,
        ))
        np.save(tr_file, save_warp)

        vid_info = {}
        vid_info['periods'] = []
        vid_info['periods'].append({"start": 0, "stop": 0, "clipname": "all"})
        vid_info['filename'] = input_file_short
        vid_info['transforms'] = tr_file_short

        frame_idx = 0
        for i in range(nframes):
            ret, frame = cap.read()
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%% %d/%d" %
                             ('=' * int(20 * i / float(nframes)),
                              int(100.0 * i / float(nframes)), i, nframes))
            sys.stdout.flush()

            if not ret:
                continue
            frame2 = cv2.resize(
                frame, None, fx=1.0 / scalefact, fy=1.0 / scalefact)
            if not (im1_gray.size):
                # enhance contrast in the image
                im1_gray = cv2.equalizeHist(
                    cv2.cvtColor(frame2,
                                 cv2.COLOR_BGR2GRAY))  #[200:-200,200:-200])

            im2_gray = cv2.equalizeHist(
                cv2.cvtColor(frame2,
                             cv2.COLOR_BGR2GRAY))  #[200:-200,200:-200])

            #           start=time.time()
            try:
                # find difference in movement between this frame and the last frame
                (cc, warp_matrix) = cv2.findTransformECC(
                    im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None,
                    5)  #opencv bug requires adding those two last arguments
                # (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria, np.zeros((im_height,im_width,3), np.uint8), 5) #opencv bug requires adding those two last arguments
                # this frame becames the last frame for the next iteration
                im1_gray = im2_gray.copy()
            except cv2.error as e:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            warp_matrix[:, 2] = warp_matrix[:, 2] * scalefact
            # all moves are accumulated into a matrix
            full_warp = np.dot(full_warp, np.vstack((warp_matrix, [0, 0, 1])))
            save_warp[i, :, :] = full_warp

#          print('ecc ', time.time()-start)

        print("saving tr")
        np.save(tr_file, save_warp)
        videos_info.append(vid_info)
        print("vid_info")
        print(vid_info)
        print("videos_info")
        print(videos_info)

    print("appending existing file" + videos_list)
    with open(videos_list, 'a') as handle:
        yaml.dump(videos_info, handle)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'This file prepares transform files for files in a specified directory with movies to track specified in config.',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument(
        '--ddir',
        '-d',
        required=True,
        nargs=1,
        help='Root of your data directory')

    args = parser.parse_args()
    main(args)
