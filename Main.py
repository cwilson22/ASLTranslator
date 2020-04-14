import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    print("Hello, World")
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    #First parameter is videocapture property, accessed by number
    cap.set(3, 1280) #set Width = 300
    cap.set(4, 720) #set Height = 300
    #cap.set(12, 0.1)

    #resnext = torchvision.models.resnext101_32x8d(pretrained=True,progress=True)
    
    #This displays on the screen what the predicted sign is
    sign = 'DEFAULT'
    while(True):
        ret, frame = cap.read()
        #Gets rid of the info on the bottom bar
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        frame = cv2.putText(frame, sign, (10,30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if(cv2.waitKey(1) & 0xFF == ord(' ')): #27= esc key but I don't know where the constant is being kept  #ord(' ')):
            out = cv2.imwrite('capture.jpg',frame)
            break

    cap.release()
    cv2.destroyAllWindows()
