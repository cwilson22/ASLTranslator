import numpy as np
import cv2
import torch
import torchvision

if __name__ == '__main__':
    print("Hello, World")
    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    #subtractor = cv2.createBackgroundSubtractorKNN()
    #subtractor = cv2.createBackgroundSubtractorMOG2()

    #First parameter is videocapture property, accessed by number
    cap.set(3, 1280) #set Width = 300
    cap.set(4, 720) #set Height = 300
    #cap.set(12, 0.1)

    torchvision.model
    resnext = torchvision.models.resnext101_32x8d(pretrained=True,progress=True)


    sign = 'DEFAULT'

    while(True):
        ret, frame = cap.read()
        #Gets rid of the info on the bottom bar
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        #maskedFrame = subtractor.apply(frame)

        frame = cv2.putText(frame, sign, (10,30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if(cv2.waitKey(1) & 0xFF == ord(' ')): #27= esc key but I don't know where the constant is being kept  #ord(' ')):
            out = cv2.imwrite('capture.jpg',frame)
            break

    cap.release()
    cv2.destroyAllWindows()
