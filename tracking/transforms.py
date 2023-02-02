import os, sys, glob, argparse
import csv
import cv2
import yaml
import numpy as np
from pathlib import Path
import time



def main(args):

    padding = 1000
    shift_display = np.array([1,0,padding/2,0,1,padding/2,0,0,1])
    shift_display = shift_display.reshape((3,3))

    #Load data
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)

    data_dir = config['project_directory']
    tracks_dir = os.path.join(data_dir,config['tracks_dir'])
    os.makedirs(tracks_dir, exist_ok=True)
    tracking_setup = config["tracking_setup"]
    np.set_printoptions(suppress=True)

    videos_name_regex_short = config[tracking_setup]['videos_name_regex']
    videos_list = os.path.join(data_dir,config[tracking_setup]['videos_list'])
    videos_info = []  #this list will be saved into videos_list file

    im_width = config['common']['IMAGE_W']  #size of training imageas for yolo
    im_height = config['common']['IMAGE_H']

    filelist = glob.glob(os.path.join(data_dir, videos_name_regex_short))
    print(filelist)
    scalefact = 4.0

    for input_file in filelist:

        full_i_path = Path(input_file)
        input_file_short = str(full_i_path.relative_to(data_dir))
        direct, ext = os.path.split(input_file_short)
        noext, _ = os.path.splitext(ext)
        tr_file = os.path.join(tracks_dir, noext + '_MAT.npy')
        tr_file_short = os.path.join(config['tracks_dir'], noext + '_MAT.npy')
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
        if args.visual:
            cv2.namedWindow('image transformed', cv2.WINDOW_GUI_EXPANDED)
            cv2.moveWindow('image transformed', 20,20)
        for i in range(nframes):
            if args.static: #non-moving camera.
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                full_warp = np.dot(full_warp, np.vstack((warp_matrix, [0, 0, 1])))
                save_warp[i, :, :] = full_warp
                continue

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

            if args.visual:
                inv_warp = np.linalg.inv(full_warp)
                shifted = shift_display.dot(inv_warp)
                warped_frame = cv2.warpPerspective(frame, shifted,(padding+frame.shape[1],padding+frame.shape[0]))
                cv2.imshow('image transformed', warped_frame)
                key = cv2.waitKey(1)  #& 0xFF

#          print('ecc ', time.time()-start)

        print("saving tr")
        np.save(tr_file, save_warp)
        videos_info.append(vid_info)
        print("vid_info")
        print(vid_info)
        print("videos_info")
        print(videos_info)

    if len(videos_info)!=0:
        print("appending existing file" + videos_list)
        with open(videos_list, 'a') as handle:
            yaml.dump(videos_info, handle)
    else:
        print('No videos to add to file')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'This file prepares transform files for files in a specified directory with movies to track specified in config.',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--static', '-s', default=False, action='store_true',
                        help='Static camera. The program will create videos file for you but will set all transformations to identity. It will be quick, I promise.')
    parser.add_argument('--visual', '-v', default=False, action='store_true',
                        help='Show camera transformations (for debugging)')

    args = parser.parse_args()
    main(args)
