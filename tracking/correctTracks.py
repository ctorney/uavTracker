"""
Display two frames of a tracker with the tracks to edit the corrections file manually. The correction file is read by this program and show your corrected tracks.
A - previous frame
D - next frame
Q - next movie
"""
import os, sys, glob, argparse
import csv
import cv2
import yaml
import numpy as np
import time
import pandas as pd
sys.path.append('..')
from utils.utils import init_config
from utils.yolo_detector import showTracks

"""
Simple video buffer to be able to go back a few frames when watching the track
"""
class Vidbu:
    def __init__(self,_cap,_nframes, _buflen=200):
        self.filo_frames = []
        self.first_frame = 0
        self.bufnlen = _buflen
        self.nframes = _nframes
        self.cap = _cap

    def get_i_frame(self,i):

        if len(self.filo_frames)>self.bufnlen:
            self.filo_frames.pop(0)
            self.first_frame = self.first_frame+1

        j = i - self.first_frame

        if j < 0:
            return False, self.filo_frames[0].copy()

        while j >= len(self.filo_frames):
            ret, frame = self.cap.read()
            self.filo_frames.append(frame)

        return True, self.filo_frames[j].copy()

def main(args):
    #Load data
    config = init_config(args)
    args_visual = config['args_visual']

    data_dir = config['project_directory']
    tracking_setup = config["tracking_setup"]
    buffer_size = config['common']["corrections_buffer_size"]

    np.set_printoptions(suppress=True)

    videos_list = os.path.join(data_dir, config[tracking_setup]['videos_list'])

    tracks_dir = os.path.join(data_dir,config['tracks_dir'])

    #frame rate calculus:
    step_frames = config[tracking_setup]['step_frames']
    videos_fps = config[tracking_setup]['videos_fps']
    output_vid_fps = round(videos_fps/step_frames)

    obj_thresh = config[tracking_setup]['obj_thresh']
    nms_thresh = config[tracking_setup]['nms_thresh']
    max_age_val = config[tracking_setup]['max_age']
    track_thresh_val = config[tracking_setup]['track_thresh']
    init_thresh_val = config[tracking_setup]['init_thresh']
    init_nms_val = config[tracking_setup]['init_nms']
    link_iou_val = config[tracking_setup]['link_iou']

    max_l = config['common']['MAX_L']  #maximal object size in pixels
    min_l = config['common']['MIN_L']

    transform_before_track = config[tracking_setup]['transform_before_track']
    save_output = True #corrections of tracks need to be visual and save output...
    showDetections = config['common']['show_detections']

    with open(videos_list, 'r') as video_config_file_h:
        video_config = yaml.safe_load(video_config_file_h)

    filelist = video_config
    print(yaml.dump(filelist))


    for input_file_dict in filelist:

        input_file = os.path.join(data_dir, input_file_dict["filename"])

        direct, ext = os.path.split(input_file)
        noext, _ = os.path.splitext(ext)

        print("Loading " + str(len(input_file_dict["periods"])) +
              " predefined periods for tracking...")
        for period in input_file_dict["periods"]:
            sys.stdout.write('\n')
            sys.stdout.flush()
            print(period["clipname"], period["start"], period["stop"])
            data_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_POS.txt')
            data_file_corrected = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_POS_corrected.txt')
            corrections_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_corrections.csv')
            transitions_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_transitions.csv')
            switches_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_switches.csv')
            false_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_false.csv')
            video_file_corrected = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_corrected.avi')
            print(input_file, video_file_corrected)
            if not os.path.isfile(data_file):
                print(
                    "This file has not been yet tracked, there is nothing to correct :/ run runTracker first. Skipping!!!"
                )
                continue

            cap = cv2.VideoCapture(input_file)
            #checking fps, mostly because we're curious
            fps = round(cap.get(cv2.CAP_PROP_FPS))
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

            results = []

            ##########################################################################
            ##          open the video file for inputs and outputs
            ##########################################################################
            if save_output:
                fourCC = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                out = cv2.VideoWriter(video_file_corrected, fourCC, output_vid_fps, S, True)

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
            if args_visual:
                cv2.namedWindow('tracker', cv2.WINDOW_GUI_EXPANDED)
                cv2.moveWindow('tracker', 20,20)
            #we are starting from the second frame as we want to see two frames at once

            ret, frame1 = cap.read()
            filo_frames = []
            i = 0
            key = ord('d')

            filof = Vidbu(cap,nframes,buffer_size)

            messy_tracks = pd.read_csv(data_file,header=None)
            messy_tracks.columns = ['frame_number','track_id','c0','c1','c2','c3','long_score','score']

            #load in corrections
            # corrected_tracks0 = pd.read_csv(corrections_file,header=None)
            # corrected_tracks0.columns = ['frame_number','track_id','corrected_track_id']
            # corrected_tracks = pd.merge(corrected_tracks0,messy_tracks,on=['frame_number','track_id'],how='left')
            corrected_tracks = messy_tracks.copy()
            corrected_tracks['corrected_track_id']=corrected_tracks['track_id']

            #load in transitions
            with open(transitions_file) as f:
                tracks_transitions = [line for line in f]
            for tt in tracks_transitions:
                for tt_id in tt.split(',')[1:]:
                    corrected_tracks['corrected_track_id'] = corrected_tracks['corrected_track_id'].replace({int(tt_id):int(tt.split(',')[0])})

            #load in switches and false tracks
            with open(switches_file) as f:
                tracks_switches = [line for line in f]
            for tt in tracks_switches:
                tt = tt.split(',')
                frame_start = int(tt[0])
                track_wrong = int(tt[1])
                track_right = int(tt[2])
                corrected_tracks['corrected_track_id'][corrected_tracks['frame_number']>=frame_start] =  corrected_tracks['corrected_track_id'].replace({track_wrong:track_right, track_right:track_wrong})
            with open(false_file) as f:
                tracks_false = [line for line in f]
            for tt in tracks_false:
                corrected_tracks =  corrected_tracks[corrected_tracks['corrected_track_id']!=int(tt)]

            while i < nframes:
                if key == ord('q'):
                    break

                #append previous frame into buffer and get a next frame
                try:
                    if key == ord('d'):
                        i=i+1
                        avaf1, frame1 = filof.get_i_frame(i)
                        avaf0, frame0 = filof.get_i_frame(i-1)

                    if key == ord('a'):
                        i=i-1
                        avaf1, frame1 = filof.get_i_frame(i)
                        avaf0, frame0 = filof.get_i_frame(i-1)
                except:
                    print('breaking out!')
                    break

                if key == ord('l'):
                    corrected_tracks = messy_tracks.copy()
                    corrected_tracks['corrected_track_id']=corrected_tracks['track_id']

                    #TODO remove all tracks with avg long_score under a threshold
                    corrected_tracks = corrected_tracks

                    #load in transitions
                    with open(transitions_file) as f:
                        tracks_transitions = [line for line in f]
                    for tt in tracks_transitions:
                        for tt_id in tt.split(',')[1:]:
                            corrected_tracks['corrected_track_id'] = corrected_tracks['corrected_track_id'].replace({int(tt_id):int(tt.split(',')[0])})

                    #load in switches and false tracks
                    with open(switches_file) as f:
                        tracks_switches = [line for line in f]
                    for tt in tracks_switches:
                        tt = tt.split(',')
                        frame_start = int(tt[0])
                        track_wrong = int(tt[1])
                        track_right = int(tt[2])
                        corrected_tracks['corrected_track_id'][corrected_tracks['frame_number']>=frame_start] =  corrected_tracks['corrected_track_id'].replace({track_wrong:track_right, track_right:track_wrong})
                    with open(false_file) as f:
                        tracks_false = [line for line in f]
                    for tt in tracks_false:
                        corrected_tracks =  corrected_tracks[corrected_tracks['corrected_track_id']!=int(tt)]

                #####
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%% %d/%d" %
                                 ('=' * int(20 * i / float(nframes)),
                                  int(100.0 * i / float(nframes)), i, nframes))
                sys.stdout.flush()

                #jump frames
                if (i > period["stop"] and period["stop"] != 0) or not ret:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    print("::Hey, hey! :: The end of the defined period, skipping to the next file or period (or maybe the file was shorter than you think")
                    print("nothing read from file: ")
                    sys.stdout.write('It is ')
                    sys.stdout.write(str( not ret))
                    sys.stdout.write(' that it is the end of the file :)')
                    sys.stdout.flush()
                    cap.release()
                    break
                if i < period["start"]:
                    continue
                if ((i - period["start"]) % step_frames):
                    continue

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

                if transform_before_track:
                    frame0 = cv2.warpPerspective(frame0, full_warp, (frame0.shape[1],frame0.shape[0]))
                    frame1 = cv2.warpPerspective(frame1, full_warp, (frame1.shape[1],frame1.shape[0]))
                    full_warp = None
                    inv_warp = None

                frame0c= frame0.copy()
                frame1c= frame1.copy()
                if avaf1: #only draw on available frames
                    for _, track in messy_tracks[messy_tracks['frame_number']==i].iterrows():
                        frame1 =showTracks(track,frame1,i,full_warp)

                    for _, track in corrected_tracks[corrected_tracks['frame_number']==i].iterrows():
                        frame1c =showTracks(track,frame1c,i,full_warp, corrected=True)

                if avaf0: #only draw on available frames
                    for _, track in messy_tracks[messy_tracks['frame_number']==(i-1)].iterrows():
                        frame0 = showTracks(track,frame0,i-1,full_warp,corrected=False)

                    for _, track in corrected_tracks[corrected_tracks['frame_number']==i].iterrows():
                        frame0c =showTracks(track,frame0c,i-1,full_warp, corrected=True)

                if save_output:
                    #       cv2.imshow('', frame)
                    #       cv2.waitKey(10)
                    framex1 = cv2.resize(frame1, S)
                    #   im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
                    out.write(framex1)

                if args_visual:
                    framex0 = cv2.resize(frame0, S)
                    framex1 = cv2.resize(frame1, S)
                    framey0 = cv2.resize(frame0c, S)
                    framey1 = cv2.resize(frame1c, S)
                    img_pair = np.concatenate((framex0, framex1), axis=1)
                    img_pair_corrected = np.concatenate((framey0, framey1), axis=1)
                    img_quart = np.concatenate((img_pair, img_pair_corrected), axis=0)
                    cv2.imshow('tracker', img_quart)
                    key = cv2.waitKey(0)  #& 0xFF


                corrected_tracks[['frame_number','corrected_track_id', 'c0','c1','c2','c3','long_score','score']].to_csv(data_file_corrected,header=None,index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Validates the tracks from runTracker. It uses specified yml file whic defines beginnings and ends of files to analyse',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--config', '-c', required=True, nargs=1, help='Your yml config file')
    parser.add_argument('--visual', '-v', default=False, action='store_true',
                        help='Display tracking progress')

    args = parser.parse_args()
    main(args)
