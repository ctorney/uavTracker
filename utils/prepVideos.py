import os, sys, glob, argparse
import csv

import cv2
import yaml
import numpy as np

sys.path.append('..')
import time
from utils import md5check, init_config


def main(args):
    #Load data
    config = init_config(args)

    data_dir = config['project_directory']

    key = ord('c') #by default, continue

    np.set_printoptions(suppress=True)
    tracking_setup = config['tracking_setup']
    videos_list = os.path.join(data_dir, config[tracking_setup]['videos_list'])

    tracks_dir = os.path.join(data_dir,config['tracks_dir'])

    print('Remember to run `transforms` script before this file so, yml file with a list of videos is created')
    with open(videos_list, 'r') as video_config_file_h:
        video_config = yaml.safe_load(video_config_file_h)

    filelist = video_config
    print(yaml.dump(filelist))

    for input_file_dict in filelist:

        if key==ord('q'):
            break
        input_file = os.path.join(data_dir, input_file_dict["filename"])

        direct, ext = os.path.split(input_file)
        noext, _ = os.path.splitext(ext)

        print("Loading " + str(len(input_file_dict["periods"])) +
              " predefined periods for tracking...")
        cap = cv2.VideoCapture(input_file)
        #checking fps, mostly because we're curious
        fps = round(cap.get(cv2.CAP_PROP_FPS))

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        S = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if width == 0:
            print('WIDTH is 0. It is safe to assume that the file doesn\'t exist or is corrupted. Hope the next one isn\'t... Skipping - obviously. ')
            break

        ##########################################################################
        ##          corrections for camera motion
        ##########################################################################
        tr_file = os.path.join(data_dir, input_file_dict["transforms"])
        print("Loading transformations from " + tr_file)
        if os.path.isfile(tr_file):
            saved_warp = np.load(tr_file)
            print("done!")
        else:
            saved_warp = None
            print(":: oh dear! :: No transformations found.")

        nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.namedWindow('tracker', cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow('tracker', 20,20)

        #skip first 20 frames in case camera is warming up
        for i in range(20):
            ret, frame = cap.read() #we have to keep reading frames

        ret, frame = cap.read() #we have to keep reading frames
        if saved_warp is None:
            full_warp = np.eye(3, 3, dtype=np.float32)
        else:
            full_warp = saved_warp[i]

        #avoid crash when matrix is singular (det is 0 and cannot invert, crashes instead, joy!
        try:
            inv_warp = np.linalg.inv(full_warp)
        except:
            print('Couldn\'t invert matrix, not transforming this frame')
            inv_warp = np.linalg.inv(np.eye(3, 3, dtype=np.float32))

        im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
        cv2.putText(frame, str(i),  (30,60), cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2);

        frame = cv2.resize(frame, S)
        cv2.imshow('tracker', frame)
        cv2.imshow('aligned', im_aligned)
        key = cv2.waitKey(0)  #& 0xFF

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Run tracker. It uses specified yml file whic defines beginnings and ends of files to analyse',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--visual', '-v', default=False, action='store_true',
                        help='Display tracking progress')
    parser.add_argument('--step', '-s', default=False, action='store_true',
                        help='Do it step by step')


    args = parser.parse_args()
    main(args)
