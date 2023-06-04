import os, sys, glob, argparse
import csv
import pytesseract
import cv2
import yaml
import numpy as np

sys.path.append('..')
import time, datetime
from utils import md5check, init_config

def main(args):
    #Load data
    config = init_config(args)

    data_dir = config['project_directory']

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

        timestamps_out_file = os.path.join(tracks_dir, noext + '_timestamps.txt')
        landmarks_file = os.path.join(tracks_dir, noext + '_landmarks.yml')

        timestamps = []
        with open(timestamps_out_file, "r") as input:
            timestamps = input.readlines()

        landmarks_dict = dict()
        with open(landmarks_file, 'r') as input:
            landmarks_dict = yaml.safe_load(input)

        print("Loading " + str(len(input_file_dict["periods"])) +
              " predefined periods for tracking...")
        for period in input_file_dict["periods"]:
            sys.stdout.write('\n')
            sys.stdout.flush()
            print(period["clipname"], period["start"], period["stop"])
            data_file_corrected = os.path.join(tracks_dir, noext + "_" + period["clipname"] + '_POS_corrected.txt')
            if not os.path.isfile(data_file_corrected):
                print(
                    "This file has not been yet tracked **and validated/corrected**, there is nothing to convert :/ run runTracker and correctTracks first. Skipping!!!"
                )
                continue

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

            ####TODO
            ## To Be Continued

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'This is a script specifically for Miks\' salmon tracking research. It takes all of the tracker outputs, landmarks and timestamp file to provide a real-life locations and timings of all the fish',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--config', '-c', required=True, nargs=1, help='Your yml config file')

    args = parser.parse_args()
    main(args)
