import cv2

# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
# https://www.datacamp.com/tutorial/face-detection-python-opencv
# Object Detection using Haar feature-based cascade classifiers is an effective object detection method
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)
# a video capture object
cam = cv2.VideoCapture(0)

while(True):
    # read a video frame by frame
    ret, frame = cam.read()

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    eye = eye_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    for (x, y, w, h) in eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)

    cv2.imshow('frame', frame)


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# release the capture object
cam.release()
# destroy all windows
cv2.destroyAllWindows()

