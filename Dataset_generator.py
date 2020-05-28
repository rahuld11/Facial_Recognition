import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cam.set(3, 640) # set video width

cam.set(4, 480) # set video height

#make sure 'haarcascade_frontalface_default.xml' is in the same folder as this code
face_cascade = cv2.CascadeClassifier('G:/Opencv/Face_Recognition/Haar cascade/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id (must enter number start from 1, this is the lable of person 1)
id = input('enter user id')

# Initialize individual sampling face count
sampleNum = 0

#here we start detect your face and take 47 pictures
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        sampleNum = sampleNum+1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
        
    cv2.imshow("Face",img)
    cv2.waitKey(1)
    
    if(sampleNum>46): # Take 47 face sample and stop video
        break
    
cam.release() 
cv2.destroyAllWindows()    