import os, sys, glob, argparse
import csv
import pytesseract
import cv2
import yaml
import numpy as np

sys.path.append('..')
import time, datetime
from utils import md5check, init_config

pa = (0,0)
landmarks_list = []

def onmouse(event, x, y, flags, param):
    global landmarks_list

    if(event == cv2.EVENT_LBUTTONDOWN):
        pa = (x,y)
        print(f'Added point {pa}')
        landmarks_list.append(pa)

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

        im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]))
        # im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
        cv2.putText(frame, str(i),  (30,60), cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2);

        frame = cv2.resize(frame, S)
        cv2.imshow('frame', frame)
        frame_disp = frame.copy()
        #First ask to provide a rectangle for time reading.
        while key != ord('q'):
            cv2.imshow('frame', frame_disp)
            cv2.setMouseCallback("frame", onmouse, param = None)
            for iii, landmark in enumerate(landmarks_list):
                cv2.circle(frame_disp, landmark,5,(0,0,230), -1)
            if len(landmarks_list) == 4:
                cv2.rectangle(
                    frame_disp, landmarks_list[0],
                    landmarks_list[1], (200, 0, 0), 1)
                cv2.rectangle(
                    frame_disp, landmarks_list[2],
                    landmarks_list[3], (0, 200, 0), 1)
            if len(landmarks_list) > 4:
                landmarks_list = []
                frame_disp = frame.copy()
            key = cv2.waitKey(100)  #& 0xFF

        timebox = frame[landmarks_list[0][1]:landmarks_list[1][1],
                        landmarks_list[0][0]:landmarks_list[1][0],
                        :]
        datebox = frame[landmarks_list[2][1]:landmarks_list[3][1],
                        landmarks_list[2][0]:landmarks_list[3][0],
                        :]

        mytime = pytesseract.image_to_string(timebox,config="-c tessedit_char_whitelist=123456789: --psm 6") # only digits
        mydate = pytesseract.image_to_string(datebox,config="--psm 6")

        current_date_str = f'{mydate} {mytime}'.replace('\n','')
        current_date = datetime.datetime.strptime(current_date_str, "%d-%b-%Y %H:%M:%S")
        print(current_date)

        ####
        ## Landmarks
        ####

        camera_name = input("what is this camera name?\n")
        key = ord('c')
        landmarks_list = [] # clear landmark list after getting date/time rectangles
        while key != ord('q'):
            for iii, landmark in enumerate(landmarks_list):
                cv2.circle(im_aligned, landmark,5,(0,0,230), -1)
                cv2.putText(im_aligned, chr(65+iii),  landmark, cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2)
            cv2.imshow('aligned', im_aligned)
            cv2.setMouseCallback("aligned", onmouse, param = None)
            key = cv2.waitKey(100)  #& 0xFF

        #Saving the landmarks

        #open a file
        landmarks_dict = {'camera': camera_name,
                          'datetime': current_date_str}
        landmarks_list_named = []
        for iii, landmark in enumerate(landmarks_list):
            landmarks_list_named.append((chr(65+iii),landmark[0],landmark[1]))

        landmarks_dict['landmarks'] = landmarks_list_named

        landmarks_file = os.path.join(tracks_dir, noext + '_landmarks.yml')
        with open(landmarks_file, 'w') as handle:
            yaml.dump(landmarks_dict, handle)

        #read in timestamps file
        noextwithdir, _ = os.path.splitext(input_file)
        timestamps_file = os.path.join(noextwithdir + '.txt')
        print(timestamps_file)

        with open(timestamps_file) as f:
            timestamps = [(int(line.split(', ')[0]),
                           line.split(', ')[1].replace('\n',''))
                          for line in f]
        print(timestamps[:10])
        # print(datetime.datetime.strptime(x,"%H%M%S%f").strftime('%Y-%m-%d %H:%M:%S'))

        #Run through the video and provide the time for each frame.
        #TODO read/check existing frames
        #now we will read a second frame...
        ret, frame = cap.read() #we have to keep reading frames

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
