import os, sys, glob, argparse
import csv
import cv2
import yaml
import numpy as np
from pathlib import Path
import time
sys.path.append('..')
from utils.utils import init_config

dx=0
dy=0
pa = (0,0)
pb = (0,0)

def onmouse(event, x, y, flags, param):
    global pa, pb, dx, dy

    if(event == cv.EVENT_LBUTTONDOWN):
        pa = (x,y)
        dx = pb[0]-pa[0]
        dy = pb[1]-pa[1]
        print(f'pa is {pa}')
    if(event == cv.EVENT_RBUTTONDOWN):
        pb = (x,y)
        dx = pb[0]-pa[0]
        dy = pb[1]-pa[1]
        print(f'pb is {pb}')

def main(args):

    padding = 1000
    shift_display = np.array([1,0,padding/2,0,1,padding/2,0,0,1])
    shift_display = shift_display.reshape((3,3))

    #Load data
    config = init_config(args)
    args_static = config['args_static']
    args_visual = config['args_visual']

    args_manual = config['args_manual']

    if args_manual:
        args_static = True
        args_visual = True

    data_dir = config['project_directory']
    tracks_dir = os.path.join(data_dir,config['tracks_dir'])
    os.makedirs(tracks_dir, exist_ok=True)
    tracking_setup = config["tracking_setup"]
    np.set_printoptions(suppress=True)
    key = ord('c')

    videos_name_regex_short = config[tracking_setup]['videos_name_regex']
    videos_list = os.path.join(data_dir,config[tracking_setup]['videos_list'])
    videos_info = []  #this list will be saved into videos_list file

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

        (h, w) = frame.shape[:2]
        (cX, cY) = (w // 2, h // 2)
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
        if args_visual:
            cv2.namedWindow('image_transformed', cv2.WINDOW_GUI_EXPANDED)
            cv2.moveWindow('image_transformed', 20,20)
            if args_manual:
                cv2.setMouseCallback("image_transformed", onmouse, param = None)

        for i in range(nframes):
            if args_static: #non-moving camera.
                warp_matrix = np.eye(2, 3, dtype=np.float32)
                #or: correct a non-moving camera
                if args_manual:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    while key != ord('q'):
                        cv2.imshow('image_transformed',frame)
                        key = cv2.waitKey(0)
                        print(f'dx and dy is {dx} and {dy}')
                        degrot = - np.degrees(np.arctan2(dx,dy))
                        M = cv2.getRotationMatrix2D((cX, cY), degrot, 1.0)
                        rotated = cv.warpAffine(frame, M, (w, h))
                        cv2.imshow('image_transformed',rotated)
                        key = cv2.waitKey(0)

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

            if args_visual:
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
    parser.add_argument('--manual', '-m', default=False, action='store_true',
                        help='Provide a manual correction for a statically misaligned camera')

    args = parser.parse_args()
    main(args)
