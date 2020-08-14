import cv2
from random import randrange


#img_file = 'peeps.jpg'
#img = cv2.imread(img_file)

webcam = cv2.VideoCapture(0)

face_tracker = cv2.CascadeClassifier('face_detector.xml')

while True:

    successful_read, frame = webcam.read()

    if successful_read:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    face_coordinates = face_tracker.detectMultiScale(gray_img)

    for x,y,w,h in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 1)


    cv2.imshow('Arzz Face Detector', frame)
    key = cv2.waitKey(1) #1 milisecond

    if key == ord('Q') or key == ord('q'):
        break

webcam.release()

print('Code completed.')
