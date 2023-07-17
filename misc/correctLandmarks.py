import os, sys, glob, argparse
import csv
import pytesseract
import cv2
import yaml
import numpy as np

sys.path.append('..')
import time, datetime
from utils.utils import md5check, init_config

pa = (0,0)
landmarks_list = []

def onmouse(event, x, y, flags, param):
    global landmarks_list

    if(event == cv2.EVENT_LBUTTONDOWN):
        pa = (x,y)
        print(f'Added point {pa}')
        landmarks_list.append(pa)

    if(event == cv2.EVENT_RBUTTONDOWN):
        print(f'Clearing landmark')
        landmarks_list = []

def main(args):
    global landmarks_list
    #Load data
    config = init_config(args)

    data_dir = config['project_directory']

    key = ord('c') #by default, continue

    np.set_printoptions(suppress=True)
    tracking_setup = config['tracking_setup']
    videos_list = os.path.join(data_dir, config[tracking_setup]['videos_list'])

    tracks_dir = os.path.join(data_dir,config['tracks_dir'])

    with open(videos_list, 'r') as video_config_file_h:
        video_config = yaml.safe_load(video_config_file_h)

    filelist = video_config
    print(yaml.dump(filelist))

    for input_file_dict in filelist:
        input_file = os.path.join(data_dir, input_file_dict["filename"])

        direct, ext = os.path.split(input_file)
        noext, _ = os.path.splitext(ext)
        landmarks_file = os.path.join(tracks_dir, noext + '_landmarks.yml')

        timestamps = []

        timestamps_out_file = os.path.join(tracks_dir, noext + '_timestamps.txt')
        with open(timestamps_out_file, "r") as input:
            timestamps = input.readlines()

        landmarks_dict = dict()
        with open(landmarks_file, 'r') as input:
            landmarks_dict = yaml.full_load(input)

        landmarks_list = []
        landmarks_dict = dict()
        if key==ord('q'):
            break
        landmarks_list = landmarks_dict['landmarks']
        cameraname = landmarks_dict['camera']

        #
        ch_width_px = np.sqrt(
            np.power(landmarks_list[0][1]-landmarks_list[1][1],2) +
            np.power(landmarks_list[0][2]-landmarks_list[1][2],2))

        px_to_cm = reallocs['ch_width'][cameraname] / ch_width_px

        #landmark A is the reference point
        ax = landmarks_list[0][1]
        ay = landmarks_list[0][2]

        print("Loading " + str(len(input_file_dict["periods"])) +
              " predefined periods for tracking...")
        cap = cv2.VideoCapture(input_file)
        #checking fps, mostly because we're curious
        fps = round(cap.get(cv2.CAP_PROP_FPS))

        print(f'Video setting fps is {fps}')
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        S = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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


        if os.path.isfile(timestamps_out_file):
            print(
                "File already analysed, dear sir. Remove output timestamp/landmark files to redo"
            )
            continue

        nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.namedWindow('tracker', cv2.WINDOW_GUI_EXPANDED)
        cv2.moveWindow('tracker', 20,20)

        ret, frame = cap.read() #we have to keep reading frames
        i = 0
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

        frame = cv2.warpPerspective(frame, full_warp, (S[0],S[1]))
        timestamp = timestamps[i]
        ts_clean = timestamp.replace('\n','')
        ts_clean = datetime.datetime.strptime(ts_clean,"%d-%b-%Y %H:%M:%S.%f")
        # im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
        cameraname_timestamp = f'{cameraname}: {timestamp}'
        #display timestamp
        frame = cv2.resize(frame, S)

        frame_disp = frame.copy()
        cv2.imshow('frame', frame_disp)
        cv2.putText(frame_disp, cameraname_timestamp, (120,50), cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2)

        for iii, landmark in enumerate(landmarks_list):
            lx = landmark[1]
            ly = landmark[2]
            lm = landmark[0]
            cv2.circle(frame_disp, (lx,ly),5,(0,0,230), -1)
            cv2.putText(frame_disp, f'{lm}:[{lx},{ly}]',  (lx,ly), cv2. FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0,170,0), 2)

        clear_al = im_aligned.copy()
        while key != ord('l'):
            if not landmarks_list: #empty
                im_aligned = clear_al.copy()

            for iii, landmark in enumerate(landmarks_list):
                cv2.circle(im_aligned, landmark,5,(0,0,230), -1)
                cv2.putText(im_aligned, chr(65+iii),  landmark, cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2)
            cv2.imshow('aligned', im_aligned)
            cv2.setMouseCallback("aligned", onmouse, param = None)
            key = cv2.waitKey(100)  #& 0xFF

        #Saving the landmarks

        #open a file
        new_landmarks_dict = {'camera': camera_name,
                              'datetime': landmarks_dict['datetime']}
        landmarks_list_named = []
        for iii, landmark in enumerate(landmarks_list):
            landmarks_list_named.append((chr(65+iii),landmark[0],landmark[1]))

        new_landmarks_dict['landmarks'] = landmarks_list_named

        with open(landmarks_file, 'w') as handle:
            yaml.dump(landmarks_dict, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'This is a script for extracting additional information from Miks\' salmon phd videos. Before running this script, run transforms.py with --manual flag to straighten my videos. In this scripts we perform following steps: 1. click on top-left and bottom right corner of time on the image displayed, 2. click tp and br on the date information. If it doesnt look right click again. Press `q` to continue. 3. Enter the name of the camera alpha/beta/gamma which will be used to match the landmarks. 4. On the aligned video frame click on the landmarks in alphabetical order. Press `q` to quonfirm',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--config', '-c', required=True, nargs=1, help='Your yml config file')


    args = parser.parse_args()
    main(args)
