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
        old_smolt = False
        landmarks_list = []
        landmarks_dict = dict()
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

        timestamps_out_file = os.path.join(tracks_dir, noext + '_timestamps.txt')

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

        im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]))
        # im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
        cv2.putText(frame, str(i),  (30,60), cv2. FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0,170,0), 2);

        frame = cv2.resize(frame, S)
        cv2.imshow('frame', frame)
        frame_disp = frame.copy()
        #First ask to provide a rectangle for time reading.
        while key != ord('n'):
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

        cv2.destroyWindow('frame')
        tblimits = [landmarks_list[0][1],
                    landmarks_list[1][1],
                    landmarks_list[0][0],
                    landmarks_list[1][0]]

        timebox = frame[tblimits[0]:tblimits[1],tblimits[2]:tblimits[3],:]
        # timebox = frame[landmarks_list[0][1]:landmarks_list[1][1],
        #                 landmarks_list[0][0]:landmarks_list[1][0],
        #                 :]
        datebox = frame[landmarks_list[2][1]:landmarks_list[3][1],
                        landmarks_list[2][0]:landmarks_list[3][0],
                        :]

        mytime = pytesseract.image_to_string(timebox,config="-c tessedit_char_whitelist=123456789: --psm 6").replace(' ','') # only digits
        mydate = pytesseract.image_to_string(datebox,config="--psm 6").replace(' ','')

        current_date_str = f'{mydate} {mytime}'.replace('\n','')
        try:
            current_date = datetime.datetime.strptime(current_date_str, "%d-%b-%Y %H:%M:%S")
        except:
            print(f'can\'t read it! If it is just a date that is incorrect, write it down. If you need to read time from the screen and it is incorrent maybe correct the box?')
            print(f'I read \"{current_date_str}\" - does it look good?')
            current_date_str = input("What is the datetime in the format %d-%b-%Y %H:%M:%S?\n")
            #HACK - for reading files without any displayed image
            if current_date_str == 'old':
                print(f'A ha! Goth ya! I guess you have an old video with a 10fps without any additional timestamp information :) SPECIAL mode is ON!')
                old_smolt = True
                current_date_str = input("What is the datetime in the format %d-%b-%Y %H:%M:%S?\n")
            mydate = current_date_str.split(' ')[0]

            current_date = datetime.datetime.strptime(current_date_str, "%d-%b-%Y %H:%M:%S")

        print(current_date)

        ####
        ## Landmarks
        ####

        camera_name = input("what is this camera name?\n")
        key = ord('c')
        landmarks_list = [] # clear landmark list after getting date/time rectangles
        while key != ord('l'):
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

        if os.path.exists(timestamps_file):
            use_timestamp_file = True
            with open(timestamps_file) as f:
                timestamps = [(int(line.split(', ')[0]),
                            line.split(', ')[1].replace('\n',''))
                            for line in f]
        else:
            use_timestamp_file = False
            timestamps = []

        # print(datetime.datetime.strptime(x,"%H%M%S%f").strftime('%Y-%m-%d %H:%M:%S'))
        #check if the hour of the timestamp matches that one read from the image.
        #rewrite all the datetime as the correct time
        timestamps_out = []
        nextDay = False #that happens only once as my recordings are never more than 24hrs
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
        # Our fantastico algorithm for supplementing on-screen date with frames calculation
        #
        if not use_timestamp_file and not old_smolt:

            timestamps_out.append(current_date.strftime("%d-%b-%Y %H:%M:%S.%f") + '\n')
            f_in_f=1
            for i in range(nframes-1):#we already read the first one
                ret, frame = cap.read()
                if not ret:
                    continue
                timebox = frame[tblimits[0]:tblimits[1],tblimits[2]:tblimits[3],:]
                mytime = pytesseract.image_to_string(timebox,config="-c tessedit_char_whitelist=1234567890: --psm 6").replace(' ','') # only digits
                if len(mytime)==8: #somehow missed one colon, happens often
                    if mytime[2]!=':':
                        mytime = f'{mytime[:2]}:{mytime[2:]}'
                    elif mytime[5]!=':':
                        mytime = f'{mytime[:5]}:{mytime[5:]}'
                    print('upsi')

                current_date_str = f'{mydate} {mytime}'.replace('\n','')
                pdate = current_date
                try:
                    current_date = datetime.datetime.strptime(current_date_str, "%d-%b-%Y %H:%M:%S")
                    print(current_date)
                except:
                    print(f'cant read: {current_date_str}, matching minutes and seconds...')
                    try:
                        c_second = int(mytime[6:])
                        c_minute = int(mytime[3:5])
                        current_date = current_date.replace(second=c_second,
                                                            minute=c_minute)
                    except:
                        print(f'cant even do that...')

                #clean and prepare the time
                if (current_date.hour == 0) and (not nextDay):
                    current_date = current_date.replace(day=current_date.day+1)
                    nextDay = True

                if pdate != current_date:
                    print(f'datecting framerate {f_in_f}')
                    for iii in range(f_in_f):
                        fragpdate = pdate + datetime.timedelta(milliseconds=1000 * iii/(f_in_f+1))
                        timestamps_out.append(fragpdate.strftime("%d-%b-%Y %H:%M:%S.%f") + '\n')
                    f_in_f=1
                else:
                    f_in_f += 1

            #last frame
            print(f'datecting framerate {f_in_f}')
            for iii in range(f_in_f):
                fragpdate = pdate + datetime.timedelta(milliseconds=1000 * iii/(f_in_f+1))
                timestamps_out.append(fragpdate.strftime("%d-%b-%Y %H:%M:%S.%f") + '\n')

            with open(timestamps_out_file, "w") as output:
                output.writelines(timestamps_out)

        #Old smolt - 10fps and hope for the best!
        if old_smolt:
            timestamps_out.append(current_date.strftime("%d-%b-%Y %H:%M:%S.%f") + '\n')
            for iii in range(nframes-1):
                current_date = current_date + datetime.timedelta(milliseconds=100)
                timestamps_out.append(current_date.strftime("%d-%b-%Y %H:%M:%S.%f") + '\n')
            with open(timestamps_out_file, "w") as output:
                output.writelines(timestamps_out)


        #Run through the video and provide the time for each frame.
        #now we will read a second frame...
        # ret, frame = cap.read() #we have to keep reading frames

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
