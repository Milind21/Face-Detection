import cv2
import numpy as np
import matplotlib.pyplot as plt

#create clasiifier
face_classifier=cv2.CascadeClassifier(r"C:\Users\milin\Desktop\Projects\ObjectDetection OpenCV\haarcascade_frontalface_default.xml")
#open image
image=cv2.imread(r"C:\Users\milin\Desktop\Projects\ObjectDetection OpenCV\Friends_4926700-FRIENDS._V392939166_SX1080_.jpg")
#fix_img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) (The orignal picture is in RGB)
#cv2.imshow("Window",image)
#cv2.waitKey(0)

#helper function to find face rectangle in fixed img

def detect_face(fix_img):
    face_rect=face_classifier.detectMultiScale(fix_img,1.7,4)
    for (x,y,w,h) in face_rect:
        cv2.rectangle(fix_img,(x,y),(x+w,y+h),(255,0,0),10)
    return fix_img
result=detect_face(image)
#once output is showing please close img window to view output.
cv2.imshow("Window2",result)
cv2.waitKey(0)
