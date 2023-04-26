# canny
# https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
# https://www.geeksforgeeks.org/real-time-edge-detection-using-opencv-python/
# sobel, laplacian
# https://www.geeksforgeeks.org/python-program-to-detect-the-edges-of-an-image-using-opencv-sobel-edge-detection/
import cv2

# a video capture object
cam = cv2.VideoCapture(0)


while(True):
    # read a video frame by frame
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # This stage decides which are all edges are really edges and which are not.
    # For this, we need two threshold values, minVal and maxVal.
    # Any edges with intensity gradient more than maxVal are sure to be edges and
    # those below minVal are sure to be non-edges, so discarded.
    Canny_edge = cv2.Canny(gray, 50, 200)

    # Calculation of Sobelx
    sobelx = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=5)

    # Calculation of Sobely
    sobely = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=5)

    # sobel x + y
    sobelxy = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=1, ksize=5)

    # Calculation of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    cv2.imshow('frame', frame)
    cv2.imshow('Canny Edge', Canny_edge)
    cv2.imshow('sobel_x', sobelx)
    cv2.imshow('sobel_y', sobely)
    cv2.imshow('sobel_xy', sobelxy)
    cv2.imshow('laplacian', laplacian)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# release the capture object
cam.release()
# destroy all windows
cv2.destroyAllWindows()


