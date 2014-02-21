#!/usr/bin/python
import SimpleCV
import time
from SimpleCV import Image,  VirtualCamera, Display
import sys

if __name__ == "__main__":
3
    matches = 0
    green = (0, 255, 0)
    sleeptime = 2

    display = Display()

    cam = SimpleCV.Camera(0, {"width": 240, "height": 240})
    while display.isNotDone():
        tstart = time.time()


        frame = cam.getImage()
        #frame.save(display)

        facedetect = frame.findHaarFeatures('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

        # Only count if we find a face
        if facedetect:
            # Count all the matches
            for f in facedetect:
                matches += 1
                # Draw boxes around matches
                facedetect.sortColorDistance(green)[0].draw(green)
                frame.save(display)
                #time.sleep(sleeptime)
        else:
            frame.save(display)
        if display.mouseRight: display.done = True

    sys.exit()