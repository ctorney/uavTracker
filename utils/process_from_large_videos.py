'''
For example:

python utils/process_from_large_videos.py -d data/wilderbeest/images -f data/wilderbeest/videos/DJI_0095.MP4
'''
import os, sys, glob, argparse, cv2

def main(args):
    # Open the video file
    cap = cv2.VideoCapture(args.file[0])
    outdir = args.dir[0]

    # Create a directory to store the images
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Initialize frame counter
    count = 0

    # Loop through frames
    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        # Check if we have reached the end of the video
        if not ret:
            break
        #cut to yolo shape
        frame = frame[:(frame.shape[0] // 32) * 32,
:(frame.shape[1] // 32) * 32]
        cv2.imshow('video',frame)
        # Increment frame counter
        count += 1

        # Check if it's the tenth frame
        if count % 10 == 0:
            # Split the frame into four images
            h, w, _ = frame.shape
            half_h = h // 2
            half_w = w // 2
            images = [
                frame[:half_h, :half_w],
                frame[:half_h, half_w:],
                frame[half_h:, :half_w],
                frame[half_h:, half_w:]
            ]

            # Resize each image to be a multiple of 32
            for i, image in enumerate(images):
                filename = os.path.join(outdir, f'frame_{count}_part_{i}.jpg')
                cv2.imwrite(filename, image)

        # Check if the user has pressed 'q' to quit
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        'Take a video and break it down into smaller bits',
        epilog=
        'Any issues and clarifications: github.com/ctorney/uavtracker/issues')
    parser.add_argument(
        '--file', '-f', required=True, nargs=1, help='Your video')
    parser.add_argument(
        '--dir', '-d', required=True, nargs=1, help='output directory')

    args = parser.parse_args()
    main(args)
