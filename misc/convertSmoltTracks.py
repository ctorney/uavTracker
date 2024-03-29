import os, sys, glob, argparse
import csv
import cv2
import yaml
import numpy as np
import pandas as pd

sys.path.append('..')
import time, datetime
from utils.utils import md5check, init_config
from utils.yolo_detector import showTracks

#Channel width is 63 cm... in all of the recordings ;)
reallocs = {
    'ch_width':{
        'alpha':63,
        'beta':63,
        'gamma':63}
    }


def main(args):
    #Load data
    config = init_config(args)

    args_visual = config['args_visual']
    data_dir = config['project_directory']

    np.set_printoptions(suppress=True)
    tracking_setup = config['tracking_setup']
    transform_before_track = config[tracking_setup]['transform_before_track']
    videos_list = os.path.join(data_dir, config[tracking_setup]['videos_list'])
    tracks_dir = os.path.join(data_dir,config['tracks_dir'])

    with open(videos_list, 'r') as video_config_file_h:
        video_config = yaml.safe_load(video_config_file_h)

    filelist = video_config
    print(yaml.dump(filelist))

    for input_file_dict in filelist:

        input_file = os.path.join(data_dir, input_file_dict["filename"])
        cap = cv2.VideoCapture(input_file)
        #checking fps, mostly because we're curious
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        videos_fps = config[tracking_setup]['videos_fps']
        step_frames = config[tracking_setup]['step_frames']
        if fps != videos_fps:
            print(f'WARNING! frame rate of videos set in config file ({videos_fps}) doesn\'t much one read by opencv CAP_PROP_FPS property ({fps}). That happens but you should be aware we use whatever config file specified!')
            fps = videos_fps

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        S = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if width == 0:
            print('WIDTH is 0. It is safe to assume that the file doesn\'t exist or is corrupted. Hope the next one isn\'t... Skipping - obviously. ')
            break

        direct, ext = os.path.split(input_file)
        noext, _ = os.path.splitext(ext)

        timestamps_out_file = os.path.join(tracks_dir, noext + '_timestamps.txt')
        landmarks_file = os.path.join(tracks_dir, noext + '_landmarks.yml')

        timestamps = []
        with open(timestamps_out_file, "r") as input:
            timestamps = input.readlines()

        landmarks_dict = dict()
        with open(landmarks_file, 'r') as input:
            landmarks_dict = yaml.full_load(input)

        nframes = len(timestamps)
        if args_visual:
            cv2.namedWindow('tracker', cv2.WINDOW_GUI_EXPANDED)
            cv2.moveWindow('tracker', 20,20)
        #we are starting from the second frame as we want to see two frames at once

        print("Loading " + str(len(input_file_dict["periods"])) +
              " predefined periods for tracking...")
        for period in input_file_dict["periods"]:
            sys.stdout.write('\n')
            sys.stdout.flush()
            print(period["clipname"], period["start"], period["stop"])
            data_file_corrected = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_POS_corrected.txt')
            data_file_reallife = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_POS_reallife.txt')
            if not os.path.isfile(data_file_corrected):
                print(
                    "This file has not been yet tracked **and validated/corrected**, there is nothing to convert :/ run runTracker and correctTracks first. Skipping!!!"
                )
                continue

            if os.path.isfile(data_file_reallife):
                print(
                    f"This file already have a file with real life tracks {data_file_reallife} Skipping!!!"
                )
                continue

            reallife_tracks = pd.DataFrame(columns = ['frame_number','timestamp','track_id', 'dx','dy','w','h','long_score','score'])
            nloc = 0 #for a new df

            #######################################################################
            ##          corrections for camera motion
            #######################################################################
            tr_file = os.path.join(data_dir, input_file_dict["transforms"])
            print("Loading transformations from " + tr_file)
            if os.path.isfile(tr_file):
                saved_warp = np.load(tr_file)
                print("done!")
            else:
                saved_warp = None
                print(":: oh dear! :: No transformations found.")

            corrected_tracks = pd.read_csv(data_file_corrected,header=None)
            corrected_tracks.columns = ['frame_number','corrected_track_id','c0','c1','c2','c3','long_score','score']
            #display landmarks
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

            #Ok, now display everything with all the calculated data.
            #We will assume a real-life location of 0.0 for landmark A of gamma (a bit of a bottom left corner
            key = ord('c')
            failed_read = False
            for i in range(nframes):

                if args_visual:
                    ret, frame = cap.read()
                    failed_read = not ret

                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%% %d/%d" %
                                 ('=' * int(20 * i / float(nframes)),
                                  int(100.0 * i / float(nframes)), i, nframes))
                sys.stdout.flush()

                #jump frames
                if (i > period["stop"] and period["stop"] != 0) or failed_read:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    print("::Hey, hey! :: The end of the defined period, skipping to the next file or period (or maybe the file was shorter than you think")
                    print(f"nothing read from file: failed_read: {failed_read} ")
                    sys.stdout.flush()
                    if args_visual:
                        cap.release()
                    break
                if i < period["start"]:
                    continue
                if ((i - period["start"]) % step_frames):
                    continue

                #GET TIMESTAMP
                timestamp = timestamps[i]
                ts_clean = timestamp.replace('\n','')
                ts_clean = datetime.datetime.strptime(ts_clean,"%d-%b-%Y %H:%M:%S.%f")

                if args_visual:
                    if saved_warp is None:
                        full_warp = np.eye(3, 3, dtype=np.float32)
                    else:
                        full_warp = saved_warp[i]
                    try:
                        inv_warp = np.linalg.inv(full_warp)
                    except:
                        print('Couldn\'t invert matrix, not transforming this frame')
                        inv_warp = np.linalg.inv(np.eye(3, 3, dtype=np.float32))

                    #transform frame
                    if transform_before_track:
                        frame = cv2.warpPerspective(frame, full_warp, (S[0],S[1]))
                        full_warp = None
                        inv_warp = None


                for _, track in corrected_tracks[corrected_tracks['frame_number']==i].iterrows():
                    #calculate distance in pixel for each track
                    #for each track calculate coordinates of the centre
                    #for each track calculate length of fish from aspect ratio and longer length

                    mx = track['c0'] + (track['c2'] - track['c0'])/2
                    my = track['c1'] + (track['c3'] - track['c1'])/2
                    dx = (mx - ax) * px_to_cm #TODO change dx to x, it isn't a dx
                    dy = - (my - ay) * px_to_cm
                    w = (track['c2'] - track['c0']) * px_to_cm
                    h = (track['c3'] - track['c1']) * px_to_cm

                    position = f'[{dx:.0f},{dy:.0f}]'
                    reallife_tracks.loc[nloc] = {
                        'frame_number':i,
                        'timestamp':ts_clean,
                        'track_id':int(track['corrected_track_id']),
                        'dx':dx,
                        'dy':dy,
                        'w':w,
                        'h':h,
                        'long_score':track['long_score'],
                        'score':track['score'],
                    }
                    nloc += 1
                    if args_visual:
                        frame =showTracks(track,frame,i,full_warp, True, position)

                if args_visual:
                    #display tracks
                    #display cameraname

                    cameraname_timestamp = f'{cameraname}: {timestamp}'
                    #display timestamp
                    cv2.putText(frame, cameraname_timestamp, (120,50), cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2)
                    for iii, landmark in enumerate(landmarks_list):
                        lx = landmark[1]
                        ly = landmark[2]
                        lm = landmark[0]
                        cv2.circle(frame, (lx,ly),5,(0,0,230), -1)
                        cv2.putText(frame, f'{lm}:[{lx},{ly}]',  (lx,ly), cv2. FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0,170,0), 2)
                    cv2.imshow('tracker',frame)
                    key = cv2.waitKey(0)  #& 0xFF

                if key == ord('q'):
                    break

            reallife_tracks.to_csv(data_file_reallife,header=True,index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'This is a script specifically for Miks\' salmon tracking research. It takes all of the tracker outputs, landmarks and timestamp file to provide a real-life locations and timings of all the fish',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--visual', '-v', default=False, action='store_true',
                        help='Display tracking progress')

    args = parser.parse_args()
    main(args)
