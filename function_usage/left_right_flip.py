import cv2

# a video capture object
cam = cv2.VideoCapture(0)

while(True):
    # read a video frame by frame
    ret, frame = cam.read()
    # flip horizontally
    flip_frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    cv2.imshow('left_right_flip', flip_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# release the capture object
cam.release()
# destroy all windows
cv2.destroyAllWindows()
