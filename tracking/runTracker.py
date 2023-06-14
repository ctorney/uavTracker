import os, sys, glob, argparse
import csv

import cv2
import yaml
import numpy as np

sys.path.append('..')
sys.path.append('.')
import time
from utils.yolo_detector import yoloDetector, showDetections
from yolo_tracker import yoloTracker
from utils.utils import md5check, init_config

def main(args):
    #Load data
    config = init_config(args)

    args_visual = config['args_visual']
    data_dir = config['project_directory']
    tracking_setup = config["tracking_setup"]
    args_step = config['args_step']

    key = ord('c') #by default, continue
    if args_step:
        wk_length = 0
    else:
        wk_length = 20

    np.set_printoptions(suppress=True)

    videos_list = os.path.join(data_dir, config[tracking_setup]['videos_list'])

    weights_dir = os.path.join(data_dir, config['weights_dir'])
    weights = os.path.join(weights_dir, config[tracking_setup]['weights'])
    md5check(config[tracking_setup]['weights_md5'], weights)

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
    hold_without = config[tracking_setup]['hold_without']

    max_l = config['common']['MAX_L']  #maximal object size in pixels
    min_l = config['common']['MIN_L']

    save_output = config['common']['save_output']
    transform_before_track = config[tracking_setup]['transform_before_track']
    show_detections = config['common']['show_detections']

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
        for period in input_file_dict["periods"]:
            if key==ord('q'):
                break
            sys.stdout.write('\n')
            sys.stdout.flush()
            print(period["clipname"], period["start"], period["stop"])
            data_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_POS.txt')
            corrections_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_corrections.csv')
            transitions_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_transitions.csv')
            switches_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_switches.csv')
            false_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_false.csv')

            video_file = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_TR.avi')
            print(input_file, video_file)
            if os.path.isfile(data_file):
                print(
                    "File already analysed, dear sir. Remove output files to redo"
                )
                continue

            cap = cv2.VideoCapture(input_file)
            #checking fps, mostly because we're curious
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            if fps != videos_fps:
                print('WARNING! frame rate of videos set in config file doesn\'t much one read by opencv CAP_PROP_FPS property. That happens but you should be aware we use whatever config file specified!')
                fps = videos_fps

            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            S = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            if width == 0:
                print('WIDTH is 0. It is safe to assume that the file doesn\'t exist or is corrupted. Hope the next one isn\'t... Skipping - obviously. ')
                break

            ##########################################################################
            ##          set-up yolo detector and tracker
            ##########################################################################
            #detector = yoloDetector(width, height, wt_file = weights, obj_threshold=0.05, nms_threshold=0.5, max_length=100)
            #tracker = yoloTracker(max_age=30, track_threshold=0.5, init_threshold=0.9, init_nms=0.0, link_iou=0.1)
            print("Loading YOLO models")
            print("We will use the following model for testing: ")
            print(weights)
            detector = yoloDetector(
                width,
                height,
                wt_file=weights,
                obj_threshold=obj_thresh,
                nms_threshold=nms_thresh,
                max_length=max_l)

            tracker = yoloTracker(
                max_age=max_age_val,
                track_threshold=track_thresh_val,
                init_threshold=init_thresh_val,
                init_nms=init_nms_val,
                link_iou=link_iou_val,
                hold_without = hold_without)

            results = []
            corrections_template = []

            ##########################################################################
            ##          open the video file for inputs and outputs
            ##########################################################################
            if save_output:
                fourCC = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                out = cv2.VideoWriter(video_file, fourCC, output_vid_fps, S, True)

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
            for i in range(nframes):

                ret, frame = cap.read() #we have to keep reading frames
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
                    # print('skipping frame!')
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

                #For simplicity in smolt tracking we can first convert the frame and track in the converted image.
                if transform_before_track:
                    frame = cv2.warpPerspective(frame, full_warp, (frame.shape[1],frame.shape[0]))
                    full_warp = None
                    inv_warp = None


                # Run detector
                detections = detector.create_detections(
                    frame, inv_warp )                 # Update tracker
                tracks = tracker.update(np.asarray(detections))

                if show_detections:
                    frame = showDetections(detections, frame, full_warp)

                cv2.putText(frame, str(i),  (30,60), cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2);

                for track in tracks:
                    bbox = track[0:4]
                    if save_output:
                        if full_warp == None:
                            minx = bbox[0]
                            miny = bbox[1]
                            maxx = bbox[2]
                            maxy = bbox[3]
                        else:
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
                            int(track[4])
                        )  # show each track as its own colour - note can't use np random number generator in this code
                        r = np.random.randint(256)
                        g = np.random.randint(256)
                        b = np.random.randint(256)
                        cv2.rectangle(frame, (int(minx), int(miny)),
                                      (int(maxx), int(maxy)), (r, g, b), 4)
                        cv2.putText(frame, str(int(track[4])),
                                    (int(minx) - 5, int(miny) - 5), 0,
                                    5e-3 * 200, (r, g, b), 2)

                        cv2.putText(frame, str(int(100*track[5])),
                                    (int(maxx) + 5, int(miny) - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (r, g, b), 1)
                        #include long score and score
                    results.append([
                        i, int(track[4]), bbox[0], bbox[1], bbox[2], bbox[3], track[5], track[6]
                    ])
                    corrections_template.append([
                        i, int(track[4]), int(track[4])
                    ])

                if save_output:
                    #       cv2.imshow('', frame)
                    #       cv2.waitKey(10)
                    frame = cv2.resize(frame, S)
                    #   im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
                    out.write(frame)

                if args_visual:
                    frame = cv2.resize(frame, S)
                    cv2.imshow('tracker', frame)
                    key = cv2.waitKey(wk_length)  #& 0xFF
                #cv2.imwrite('pout' + str(i) + '.jpg',frame)
                if key in [ord('q'), ord('s')]:
                    break


            with open(data_file, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(results)

            with open(corrections_file, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows(corrections_template)

            with open(transitions_file, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows([])

            with open(switches_file, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows([])

            with open(false_file, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                writer.writerows([])

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
