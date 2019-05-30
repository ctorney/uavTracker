import os, sys, glob
import csv
import cv2
import yaml
import numpy as np

import time


def main(argv):
    if(len(sys.argv) != 3):
        print('Usage ./transforms.py [root_dir] [config.yml]')
        sys.exit(1)
    #Load data
    root_dir = argv[1]  + '/' #in case we forgot
    print('Opening config file' + root_dir + argv[2])
    with open(root_dir + argv[2], 'r') as configfile:
        config = yaml.safe_load(configfile)

    np.set_printoptions(suppress=True)
    data_dir = config['movie_dir']
    video_name_regex = data_dir + "/*.avi"
    weights_dir = root_dir + config['weights_dir']
    your_weights = weights_dir + config['specific_weights']
    generic_weights = weights_dir + config['generic_weights']
    trained_weights = weights_dir + config['trained_weights']
    #train_images =  glob.glob( image_dir + "*.png" )
    #video_file1 = 'out.avi'

    #TODO allow different resolution
    width = 3840#1920
    height = 2176#1080

    display = config['display']
    showDetections = config['showDetections']

    filelist = glob.glob(video_name_regex)

    scalefact=4.0

    for input_file in filelist:

        direct, ext = os.path.split(input_file)
        noext, _ = os.path.splitext(ext)
        data_file = data_dir + '/tracks/' +  noext + '_POS.txt'
        video_file = data_dir + '/tracks/' +  noext + '_TR.avi'
        tr_file = data_dir + '/tracks/' +  noext + '_MAT.npy'
        if os.path.isfile(tr_file):
            continue
        print(input_file, video_file)

        ##########################################################################
        ##          open the video file for inputs 
        ##########################################################################
        cap = cv2.VideoCapture(input_file)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        nframes = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        ##########################################################################
        ##          corrections for camera motion
        ##########################################################################
        warp_mode = cv2.MOTION_EUCLIDEAN
        number_of_iterations = 20
        # Specify the threshold of the increment in the correlation coefficient between two iterations
        termination_eps = 1e-6;
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        im1_gray = np.array([])
        warp_matrix = np.eye(2, 3, dtype=np.float32) 
        full_warp = np.eye(3, 3, dtype=np.float32)

        save_warp = np.zeros((nframes,3,3,))
        np.save(tr_file,save_warp)



        frame_idx=0
        for i in range(nframes):

            ret, frame = cap.read() 
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%% %d/%d" % ('='*int(20*i/float(nframes)), int(100.0*i/float(nframes)), i,nframes)) 
            sys.stdout.flush()

            frame2 =cv2.resize(frame,None, fx=1.0/scalefact, fy=1.0/scalefact)
            if not(im1_gray.size):
                # enhance contrast in the image
                im1_gray = cv2.equalizeHist(cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY))#[200:-200,200:-200])

            im2_gray =  cv2.equalizeHist(cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY))#[200:-200,200:-200])


 #           start=time.time()
            try:
                # find difference in movement between this frame and the last frame
                (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
                # this frame becames the last frame for the next iteration
                im1_gray = im2_gray.copy()
            except cv2.error as e:
                warp_matrix = np.eye(2, 3, dtype=np.float32)

            warp_matrix[:,2]=warp_matrix[:,2]*scalefact
            # all moves are accumulated into a matrix
            full_warp = np.dot(full_warp,np.vstack((warp_matrix,[0,0,1])))
            save_warp[i,:,:] = full_warp
  #          print('ecc ', time.time()-start)

        np.save(tr_file,save_warp)



if __name__ == '__main__':
    main(sys.argv)
