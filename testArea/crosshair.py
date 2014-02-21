from SimpleCV import Camera, Color, Display, RunningSegmentation



cam = Camera(0, {"width": 100, "height": 140})
rs = RunningSegmentation(.5)
size = (cam.getImage().size())
display = Display(size)
# Start the crosshairs in the center of the screen
center = (size[0] / 2, size[1] / 2)
while display.isNotDone():
    input1 = cam.getImage()
    # Assume using monitor mounted camera, so flip to create mirror image
    input1 = input1.flipHorizontal()
    rs.addImage(input1) #
    if(rs.isReady()):
        # Get the object that moved
        img = rs.getSegmentedImage(False)
        blobs = img.dilate(3).findBlobs()
        #
        # If a object in motion was found
        if( blobs is not None ):
            blobs = blobs.sortArea()
            # Update the crosshairs onto the object in motion
            center = (int(blobs[-1].minRectX()),
                        int(blobs[-1].minRectY())) #
            # Inside circle
            input1.dl().circle(center, 50, Color.BLACK, width = 3) #
            # Outside circle
            input1.dl().circle(center, 200, Color.BLACK, width = 6)
            # Radiating lines
            input1.dl().line((center[0], center[1] - 50),
            (center[0], 0), Color.BLACK, width = 2)
            input1.dl().line((center[0], center[1] + 50),
            (center[0], size[1]), Color.BLACK, width = 2)
            input1.dl().line((center[0] - 50, center[1]),
            (0, center[1]), Color.BLACK, width = 2)
            input1.dl().line((center[0] + 50, center[1]),
            (size[0], center[1]), Color.BLACK, width = 2)
            input1.save(display)
