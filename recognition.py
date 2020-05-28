import cv2
import numpy as np
import os

face_casecade = cv2.CascadeClassifier("G:/Opencv/face_rec/haarcascade_frontalface_default.xml")

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

rec = cv2.face.LBPHFaceRecognizer_create()

rec.read("recognizer/trainingData.yml")

id = 0

###font = cv2.FONT_HERSHEY_SIMPLEX(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)

font = cv2.FONT_HERSHEY_SIMPLEX
     
while (True):
    ret,img = cam.read()
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = face_casecade.detectMultiScale(
            gray,
            scaleFactor = 1.3,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
            )

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        id,conf = rec.predict(gray[y:y+h, x:x+w])
        
        if (id == 1):
            id = "rahul"
        elif(id == 2):
            id = "srknth"
      
        cv2.putText(img, str(id), (x, y + h), font,1, (0, 255, 0), 1, cv2.LINE_AA)
        ###cv2.cv.putText(cv2.cv.fromarray(img),str(id),(x,y+h), font,(0,255,0), 2)
        
    cv2.imshow("Face", img)
    
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

    #if (cv2.waitKey(100)== ord("q")) :
        #break

cam.release()
cv2.destroyAllWindows() 
    