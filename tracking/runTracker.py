import os, sys, glob
import csv
import cv2
import yaml
import numpy as np

from yolo_detector import yoloDetector
from yolo_tracker import yoloTracker


def main(argv):
    if(len(sys.argv) != 3):
        print('Usage ./runTracker.py [root_dir] [config.yml]')
        sys.exit(1)
    #Load data
    root_dir = argv[1]  + '/' #in case we forgot
    print('Opening config file' + root_dir + argv[2])
    with open(root_dir + argv[2], 'r') as configfile:
        config = yaml.safe_load(configfile)
    print('Read config file.')

    np.set_printoptions(suppress=True)
    data_dir = root_dir + config['data_dir']
    video_name_regex = config['test_videos_name_regex']
    weights_dir = root_dir + config['weights_dir']
    tracks_dir = root_dir + config['tracks_dir']
    your_weights = weights_dir + config['specific_weights']
    generic_weights = weights_dir + config['generic_weights']
    trained_weights = weights_dir + config['trained_weights']
    step_frames_n = config['step_frames']
    #train_images =  glob.glob( image_dir + '*.png' )
    #video_file1 = 'out.avi'

    #TODO allow different resolution
    width = 1920
    height = 1080

    display = config['display']
    showDetections = config['showDetections']

    filelist = glob.glob(data_dir + video_name_regex)
    print('There are ' + str( len(filelist) ) + ' files to process.\n')

    for input_file in filelist:
        print('Input vid:' + input_file)
        direct, ext = os.path.split(input_file)
        noext, _ = os.path.splitext(ext)
        data_file = tracks_dir + '/' +  noext + '_POS.txt'
        video_file = tracks_dir + '/' +  noext + '_TR.avi'
        print('Tracks file: ' + str(data_file))
        print('Videos file: ' + str(video_file))
        if os.path.isfile(data_file):
            print('There is already output for this file. Skipping...')
            continue
        print(input_file, video_file)
        ##########################################################################
        ##          set-up yolo detector and tracker
        ##########################################################################
        detector = yoloDetector(width, height, wt_file = trained_weights, obj_threshold=0.05, nms_threshold=0.5, max_length=100)
        tracker = yoloTracker(max_age=30, track_threshold=0.5, init_threshold=0.9, init_nms=0.0, link_iou=0.1)
        results = []


        ##########################################################################
        ##          open the video file for inputs and outputs
        ##########################################################################
        cap = cv2.VideoCapture(input_file)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        S = (1920,1080)
        if display:
            fourCC = cv2.VideoWriter_fourcc('X','V','I','D')
            out = cv2.VideoWriter(video_file, fourCC, fps, S, True)


        ##########################################################################
        ##          corrections for camera motion
        ##########################################################################
        warp_mode = cv2.MOTION_HOMOGRAPHY
        number_of_iterations = 20
        # Specify the threshold of the increment in the correlation coefficient between two iterations
        termination_eps = 1e-6;
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        im1_gray = np.array([])
        warp_matrix = np.eye(3, 3, dtype=np.float32) 
        full_warp = np.eye(3, 3, dtype=np.float32)


        frame_idx=0
        nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('frames total: ' + str(nframes) +' frames')
        print('step: ' + str(step_frames_n) + ' frame(s)')

        for i in range(1,nframes,step_frames_n):

            sys.stdout.write('\r')
            sys.stdout.write('[%-20s] %d%% %d/%d' % ('='*int(20*i/float(nframes)), int(100.0*i/float(nframes)), i,nframes))
            sys.stdout.flush()
            for sf in range(step_frames_n):
                #make sure we do not try to read over the frame number
                if (i + sf) == nframes:
                    break;
                ret, frame = cap.read()

            if not(im1_gray.size):
                # enhance contrast in the image
                im1_gray = cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))

            im2_gray =  cv2.equalizeHist(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))


            try:
                # find difference in movement between this frame and the last frame
                (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
                # this frame becames the last frame for the next iteration
                im1_gray = im2_gray.copy()
            except cv2.error as e:
                warp_matrix = np.eye(3, 3, dtype=np.float32)

            # all moves are accumulated into a matrix
            full_warp = np.dot(warp_matrix,full_warp)

            # Run detector
            detections = detector.create_detections(frame, np.linalg.inv(full_warp))
            # Update tracker
            tracks = tracker.update(np.asarray(detections))

            if showDetections:
                for detect in detections:
                    bbox = detect[0:4]
                    if display:
                        iwarp = (full_warp)
                        corner1 = np.expand_dims([bbox[0],bbox[1]], axis=0)
                        corner1 = np.expand_dims(corner1,axis=0)
                        corner1 = cv2.perspectiveTransform(corner1,iwarp)[0,0,:]
                        minx = corner1[0]
                        miny = corner1[1]
                        corner2 = np.expand_dims([bbox[2],bbox[3]], axis=0)
                        corner2 = np.expand_dims(corner2,axis=0)
                        corner2 = cv2.perspectiveTransform(corner2,iwarp)[0,0,:]
                        maxx = corner2[0]
                        maxy = corner2[1]

                        cv2.rectangle(frame, (int(minx)-2, int(miny)-2), (int(maxx)+2, int(maxy)+2),(0,0,0), 1)


            for track in tracks:
                bbox = track[0:4]
                if display:
                    iwarp = (full_warp)

                    corner1 = np.expand_dims([bbox[0],bbox[1]], axis=0)
                    corner1 = np.expand_dims(corner1,axis=0)
                    corner1 = cv2.perspectiveTransform(corner1,iwarp)[0,0,:]
                    corner2 = np.expand_dims([bbox[2],bbox[3]], axis=0)
                    corner2 = np.expand_dims(corner2,axis=0)
                    corner2 = cv2.perspectiveTransform(corner2,iwarp)[0,0,:]
                    corner3 = np.expand_dims([[bbox[0],bbox[3]]], axis=0)
    #               corner3 = np.expand_dims(corner3,axis=0)
                    corner3 = cv2.perspectiveTransform(corner3,iwarp)[0,0,:]
                    corner4 = np.expand_dims([bbox[2],bbox[1]], axis=0)
                    corner4 = np.expand_dims(corner4,axis=0)
                    corner4 = cv2.perspectiveTransform(corner4,iwarp)[0,0,:]
                    maxx = max(corner1[0],corner2[0],corner3[0],corner4[0]) 
                    minx = min(corner1[0],corner2[0],corner3[0],corner4[0]) 
                    maxy = max(corner1[1],corner2[1],corner3[1],corner4[1]) 
                    miny = min(corner1[1],corner2[1],corner3[1],corner4[1]) 

                    np.random.seed(int(track[4])) # show each track as its own colour - note can't use np random number generator in this code
                    r = np.random.randint(256)
                    g = np.random.randint(256)
                    b = np.random.randint(256)
                    cv2.rectangle(frame, (int(minx), int(miny)), (int(maxx), int(maxy)),(r,g,b), 4)
                    cv2.putText(frame, str(int(track[4])),(int(minx)-5, int(miny)-5),0, 5e-3 * 200, (r,g,b),2)

                results.append([frame_idx, track[4], bbox[0], bbox[1], bbox[2], bbox[3]])
            frame_idx+=1

            if display:
        #       cv2.imshow('', frame)
        #       cv2.waitKey(10)
        #       frame = cv2.resize(frame,S)
            #   im_aligned = cv2.warpPerspective(frame, full_warp, (S[0],S[1]), borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
                out.write(frame)

            #cv2.imwrite('pout' + str(i) + '.jpg',frame)
    #   break

        with open(data_file, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(results)
    #   break
        #   for val in results:
        #      writer.writerow([val])    



if __name__ == '__main__':
    main(sys.argv)
