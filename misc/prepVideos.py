import os, sys, glob, argparse
import cv2, yaml
import numpy as np
import pandas as pd

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

def deal_with_timestamps(tracks_dir, noext, input_file, first_frame_file, vmf, filename):
    global landmarks_list
    cap = cv2.VideoCapture(input_file)
    nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        ret, frame = cap.read() #we have to keep reading frames
    except:
        print('Video is empty. Using first frame if available')
        frame - cv2.imread(first_frame_file)
        print('success!')

    timestamps_out_file = os.path.join(tracks_dir, noext + '_timestamps.txt')
    if os.path.isfile(timestamps_out_file):
        print(f"Timestamp outputs already exist in {timestamps_out_file}")
        return 0
    #
    # In either case let's first ask for current date/time
    #
    print(input_file)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(20)  #& 0xFF

    #Provided csv file with some goodies    
    if vmf is not None and filename in vmf['name'].values:
        print('Using video meta file to get the date')
        current_date_str = vmf.loc[vmf['name'] == filename, 'startdate'].values[0]
        print(current_date_str)
        current_date = datetime.datetime.strptime(current_date_str, "%d-%b-%Y %H:%M:%S")
    else:
        current_date_str = input("What is the datetime in the format %d-%b-%Y %H:%M:%S?\n")
        mydate = current_date_str.split(' ')[0]

        current_date = datetime.datetime.strptime(current_date_str, "%d-%b-%Y %H:%M:%S")


    #
    #Best option, I have a full timestamped file!
    #
    #read in timestamps file
    noextwithdir, _ = os.path.splitext(input_file)
    timestamps_file = os.path.join(noextwithdir + '.txt')
    print(timestamps_file)

    if os.path.exists(timestamps_file):
        use_timestamp_file = True
        with open(timestamps_file) as f:
            timestamps = [(int(line.split(', ')[0]),
                        line.split(', ')[1].replace('\n',''))
                        for line in f]
    else:
        use_timestamp_file = False
        timestamps = []

    #rewrite all the datetime as the correct time
    timestamps_out = []
    nextDay = False #that happens only once as my recordings are never more than 24hrs
    if use_timestamp_file:
        for iii, tt in enumerate(timestamps):
            if iii != tt[0]:
                raise ValueError(f'a frame {iii} is skipped in the timestamp list!')
            cdt = datetime.datetime.strptime(tt[1],"%H%M%S%f")
            if (cdt.hour == 0) and (not nextDay):
                current_date = current_date.replace(day=current_date.day+1)
                nextDay = True
            cdt = cdt.replace(year=current_date.year, month=current_date.month, day = current_date.day)
            timestamps_out.append(cdt.strftime("%d-%b-%Y %H:%M:%S.%f") + '\n')

    #
    #Second best, we have a first frame date and know the fps
    #
    else:
        if vmf is not None and filename in vmf['name'].values:
            print('Using video meta file to get the fps')
            pfps = vmf.loc[vmf['name'] == filename, 'fps'].values[0]
        else:
            pfps_str = input("What is the fps?\n")
            pfps = int(pfps_str)

        timestamps_out.append(current_date.strftime("%d-%b-%Y %H:%M:%S.%f") + '\n')
        for iii in range(nframes-1):
            current_date = current_date + datetime.timedelta(milliseconds=int(1000/pfps))
            timestamps_out.append(current_date.strftime("%d-%b-%Y %H:%M:%S.%f") + '\n')

    with open(timestamps_out_file, "w") as output:
        output.writelines(timestamps_out)
    #You might be excused to think that we can just read the timestamp from the screen or the Exif data. Sadly text recognition is not nearly quality enough, and since we are recording in h264 we don't really have precise file end information.
    cv2.destroyAllWindows()
    return current_date_str

def deal_with_landmarks(data_dir, input_file_dict, tracks_dir, noext, input_file, current_date_str, first_frame_file, vmf, filename):
    global landmarks_list
    landmarks_file = os.path.join(tracks_dir, noext + '_landmarks.yml')
    #Check and read in landmarks file if exists
    #
    landmarks_dict = dict()
    try:
        with open(landmarks_file, 'r') as landinput:
            landmarks_dict = yaml.full_load(landinput)
    except:
        print(f'No landmarks yet exist. Starting from scratch!')

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
        return 1

    tr_file = os.path.join(data_dir, input_file_dict["transforms"])
    print("Loading transformations from " + tr_file)
    if os.path.isfile(tr_file):
        saved_warp = np.load(tr_file)
        print("done!")
    else:
        saved_warp = None
        print(":: oh dear! :: No transformations found.")
    i = 0
    if saved_warp is None:
        full_warp = np.eye(3, 3, dtype=np.float32)
    else:
        full_warp = saved_warp[i]

    try:
        ret, frame = cap.read() #we have to keep reading frames
    except:
        print('Video is empty. Using first frame if available')
        frame - cv2.imread(first_frame_file)
        print('success!')

    frame = cv2.resize(frame, S)
    frame = cv2.warpPerspective(frame, full_warp, (S[0],S[1]))
    im_aligned = frame.copy()

    if not 'camera' in landmarks_dict.keys():
        key = ord('n')
    else:
        camera_name = landmarks_dict['camera']
        key = ord('x')

    if 'landmarks' in landmarks_dict.keys():
        landmarks_list = [(a[1],a[2]) for a in landmarks_dict['landmarks']]
    else:
        landmarks_list = [] #clear the list from the previous video

    print('add landmarks, and press \'c\' to continue')
    while key != ord('c'):
        if key == ord('n'):
            if vmf is not None and filename in vmf['name'].values:
                print('Using video meta file to get the camera name')
                camera_name = vmf.loc[vmf['name'] == filename, 'camera'].values[0]
                key = ord('x')
            else:
                camera_name = input("But first! What is this camera name?\n")
        if not landmarks_list: #empty
            im_aligned = frame.copy()
        for iii, landmark in enumerate(landmarks_list):
            cv2.circle(im_aligned, landmark,5,(0,0,230), -1)
            cv2.putText(im_aligned, chr(65+iii),  landmark, cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2)
        cv2.imshow(camera_name, im_aligned)
        cv2.setMouseCallback(camera_name, onmouse, param = None)
        key = cv2.waitKey(100)  #& 0xFF

    print('Saving landmarks!')
    landmarks_dict = {'camera': camera_name,
                        'datetime': current_date_str}
    landmarks_list_named = []
    for iii, landmark in enumerate(landmarks_list):
        landmarks_list_named.append((chr(65+iii),landmark[0],landmark[1]))

    landmarks_dict['landmarks'] = landmarks_list_named

    print(landmarks_dict)

    with open(landmarks_file, 'w') as handle:
        yaml.dump(landmarks_dict, handle)

    cv2.destroyAllWindows()
    return 0

def main(args):
    global landmarks_list
    #Load data
    config = init_config(args)
    DEBUG = config['args_debug']

    data_dir = config['project_directory']

    key = ord('c') #by default, continue

    np.set_printoptions(suppress=True)
    tracking_setup = config['tracking_setup']
    videos_list = os.path.join(data_dir, config[tracking_setup]['videos_list'])

    tracks_dir = os.path.join(data_dir,config['tracks_dir'])

    print('Remember to run `transforms` script before this file so, yml file with a list of videos is created')
    print('Opening ' + videos_list)
    with open(videos_list, 'r') as video_config_file_h:
        video_config = yaml.safe_load(video_config_file_h)

    filelist = video_config
    if DEBUG:
        print(yaml.dump(filelist))

    vmf = None
    if 'video_meta_file' in config[tracking_setup]:
        print('Found video meta file')
        vmfile = os.path.join(data_dir, config[tracking_setup]['video_meta_file'])
        vmf = pd.read_csv(vmfile)


    for input_file_dict in filelist:
        input_file = os.path.join(data_dir, input_file_dict["filename"])
        direct, ext = os.path.split(input_file)
        noext, fext = os.path.splitext(ext)

        try: 
            first_frame_file = os.path.join(data_dir, input_file_dict["first_frame"])
            #if first frame is not available we need to crete it from the video:
            if not os.path.isfile(first_frame_file):
                print('First frame not found. Creating it from the video')
                cap = cv2.VideoCapture(input_file)
                ret, frame = cap.read()
                cv2.imwrite(first_frame_file, frame)
        except:
            first_frame_file = os.path.join(tracks_dir, noext + '_first_frame.png')
            print('First frame not defined. Creating it from the video')
            print(f'First frame file is {first_frame_file}')
            cap = cv2.VideoCapture(input_file)
            ret, frame = cap.read()
            cv2.imwrite(first_frame_file, frame)


        print("Loading " + str(len(input_file_dict["periods"])) +
              " predefined periods for tracking...")

        #Deal with timestamps
        current_date_str = deal_with_timestamps(tracks_dir, noext, input_file, first_frame_file, vmf, ext)
        #Deal with getting landmarks
        deal_with_landmarks(data_dir, input_file_dict, tracks_dir, noext, input_file, current_date_str, first_frame_file, vmf, ext)

    print('Done!')

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
