import os
import cv2
import numpy as np
from PIL import Image  #pillow package

#recognizer = cv2.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()


path = 'dataSet'

# function to get the images and label data
def getImagesWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePaths in imagePaths:
        faceImg = Image.open(imagePaths).convert('L') # convert it to grayscale
        faceNp = np.array(faceImg ,'uint8')
        ID = int(os.path.split(imagePaths)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
        
    return IDs,faces

Ids, faces = getImagesWithID(path)
recognizer.train(faces,np.array(Ids))

# Here we Save the model into trainer/trainer.yml
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
        
    