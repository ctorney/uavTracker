"""

"""
import os, sys, glob, argparse
import csv
import cv2
import yaml
import numpy as np
import time
import pandas as pd

def main(args):
    #Load data
    print('Opening file' + args.config[0])
    with open(args.config[0], 'r') as configfile:
        config = yaml.safe_load(configfile)
    args_visual = True

    data_dir = config['project_directory']
    tracking_setup = config["tracking_setup"]

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
    im_width = config['common']['IMAGE_W']  #size of training imageas for yolo
    im_height = config['common']['IMAGE_H']

    save_output = True #corrections of tracks need to be visual and save output...
    showDetections = config['common']['showDetections']

    with open(videos_list, 'r') as video_config_file_h:
        video_config = yaml.safe_load(video_config_file_h)

    filelist = video_config
    print(yaml.dump(filelist))

    key = ord('m')

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
            corrections_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_corrections.txt')
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

            ret, frame0 = cap.read()
            for i in range(1,nframes):

                ret, frame1 = cap.read() #we have to keep reading frames
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

                messy_tracks = pd.read_csv(data_file,header=None)
                messy_tracks.columns = ['frame_number','track_id','c0','c1','c2','c3']

                for _, track in messy_tracks[messy_tracks['frame_number']==i].iterrows():

                    bbox = [track['c0'],
                            track['c1'],
                            track['c2'],
                            track['c3']]
                    t_id = int(track['track_id'])

                    if save_output:
                        iwarp = (full_warp)

                        corner1 = np.expand_dims([bbox[0], bbox[1]], axis=0)
                        corner1 = np.expand_dims(corner1, axis=0)
                        corner1 = cv2.perspectiveTransform(corner1,
                                                           iwarp)[0, 0, :]
                        corner2 = np.expand_dims([bbox[2], bbox[3]], axis=0)
                        corner2 = np.expand_dims(corner2, axis=0)
                        corner2 = cv2.perspectiveTransform(corner2,
                                                           iwarp)[0, 0, :]
                        corner3 = np.expand_dims([[bbox[0], bbox[3]]], axis=0)
                        #               corner3 = np.expand_dims(corner3,axis=0)
                        corner3 = cv2.perspectiveTransform(corner3,
                                                           iwarp)[0, 0, :]
                        corner4 = np.expand_dims([bbox[2], bbox[1]], axis=0)
                        corner4 = np.expand_dims(corner4, axis=0)
                        corner4 = cv2.perspectiveTransform(corner4,
                                                           iwarp)[0, 0, :]
                        maxx = max(corner1[0], corner2[0], corner3[0],
                                   corner4[0])
                        minx = min(corner1[0], corner2[0], corner3[0],
                                   corner4[0])
                        maxy = max(corner1[1], corner2[1], corner3[1],
                                   corner4[1])
                        miny = min(corner1[1], corner2[1], corner3[1],
                                   corner4[1])

                        np.random.seed(
                            t_id
                        )  # show each track as its own colour - note can't use np random number generator in this code
                        r = np.random.randint(256)
                        g = np.random.randint(256)
                        b = np.random.randint(256)
                        cv2.rectangle(frame1, (int(minx), int(miny)),
                                      (int(maxx), int(maxy)), (r, g, b), 4)
                        cv2.putText(frame1, str(t_id),
                                    (int(minx) - 5, int(miny) - 5), 0,
                                    5e-3 * 200, (r, g, b), 2)
                        cv2.putText(frame1, str(i),  (30,60), cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2);

                if save_output:
                    #       cv2.imshow('', frame)
                    #       cv2.waitKey(10)
                    framex1 = cv2.resize(frame1, S)
                    #   im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
                    out.write(framex1)

                if args_visual:
                    framex0 = cv2.resize(frame0, S)
                    framex1 = cv2.resize(frame1, S)
                    img_pair = np.concatenate((framex0, framex1), axis=1)
                    cv2.imshow('tracker', img_pair)
                    key = cv2.waitKey(0)  #& 0xFF

                if key == ord('q'):
                    break

                frame0 = frame1.copy()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Run tracker. It uses specified yml file whic defines beginnings and ends of files to analyse',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--config', '-c', required=True, nargs=1, help='Your yml config file')

    args = parser.parse_args()
    main(args)
