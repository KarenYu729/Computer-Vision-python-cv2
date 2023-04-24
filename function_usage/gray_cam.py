# https://www.geeksforgeeks.org/python-grayscaling-of-images-using-opencv/
# https://www.geeksforgeeks.org/converting-color-video-to-grayscale-using-opencv-in-python/
# import the opencv library
import cv2

# define a video capture object
cam = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = cam.read()
    # rgb to gray
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('gray_frame', gray_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()

cv2.destroyAllWindows()