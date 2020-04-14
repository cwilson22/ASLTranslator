import cv2
import os

cap = cv2.VideoCapture(0)

#The amount of data already generated aka the next number to be added to the dataset
counter = 0

#dimension x dimension is the size of the training set we are using, we may want to shrink or grow this for performance reasons in the future
dimension = 200

#Set sign that you are creating for data currently
sign = "q"

cap.set(3, 480)  # set Width = 300
cap.set(4, 360)  # set Height = 300
# Actual width and height are determined by the camera of the device
width = cap.get(3)
print(width)
height = cap.get(4)
print(height)
save = False

while (True):
    ret, frame = cap.read()
    # Gets rid of the info on the bottom bar
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    left_bound = int(width/2 - dimension/2)
    right_bound = left_bound + dimension
    top_bound = int(height/2 - dimension/2)
    bottom_bound = top_bound + dimension
    croppedFrame = frame[top_bound:bottom_bound, left_bound:right_bound] #x and y are flipped idk why

    cv2.imshow('frame', croppedFrame)

    pressed = cv2.waitKey(1)
    if save == True:
        path = "./data/%s/%s%d.jpg" % (sign,sign,counter)
        print(path)
        counter += 1
        out = cv2.imwrite(path, croppedFrame)

    if(pressed & 0xFF == ord(' ')):
        save = True
    elif (pressed & 0xFF == 101):
        save = False
    elif(pressed & 0XFF == 27):
        break


    # if (cv2.waitKey(1) & 0xFF == ord(' ')):  # 27= esc key but I don't know where the constant is being kept  #ord(' ')):
    #     out = cv2.imwrite('capture.jpg', frame)
    #     #print(os.listdir(directory))
    #     break

cap.release()
cv2.destroyAllWindows()