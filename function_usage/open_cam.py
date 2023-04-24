# https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
import cv2

# a video capture object
cam = cv2.VideoCapture(0)
"""
cv2.VideoCapture
1. camera index (mainly 0)
2. video directions, for example:
    cam = cv2.VideoCapture(video_dir)
"""

while(True):
    # read a video frame by frame
    ret, frame = cam.read()
    '''
    read() returns tuple in which 1st item is boolean value 
    either True or False and 2nd item is frame of the video.
    read() returns False when video is ended so 
    no frame is readed and error will be generated.
    '''
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# release the capture object
cam.release()
# destroy all windows
# cv2.destroyAllWindows()
# or destroy by name
cv2.destroyWindow('frame')

