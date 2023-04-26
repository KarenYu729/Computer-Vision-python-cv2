# https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
import cv2

# a video capture object
cam = cv2.VideoCapture(0)

while(True):
    # read a video frame by frame
    ret, frame = cam.read()

    color = (0, 255, 0)

    """
    *-----------------------
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
    |                      |
    -----------------------#
    *: start_point
    #: end_point
    """

    # Start coordinate, here (100, 100)
    # represents the top left corner of rectangle
    start_point = (100, 100)
    # Ending coordinate, here (300, 300)
    # represents the bottom right corner of rectangle
    end_point = (300, 300)
    thickness = 2

    # add rectangle
    # Draw a rectangle with green line borders of thickness of 2 px
    rect_frame = cv2.rectangle(frame, start_point, end_point, color, thickness)


    cv2.imshow('frame', rect_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# release the capture object
cam.release()
# destroy all windows
cv2.destroyAllWindows()
