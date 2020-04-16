import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import time
from torchvision import datasets, models
from torch.autograd import Variable
import copy
from PIL import Image
from skimage import io
from torch.utils.data import Dataset, DataLoader
from TestDataset import TestDataset

if __name__ == '__main__':
    print("Hello, World")
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX



    signs = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

    #First parameter is videocapture property, accessed by number
    cap.set(3, 1280) #set Width = 300
    cap.set(4, 720) #set Height = 300
    #cap.set(12, 0.1)

    bounded_box_height = cap.get(4)
    bounded_box_width = cap.get(3)
    dimension = 224

    #resnext = torchvision.models.resnext101_32x8d(pretrained=True,progress=True)


    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 26)
    input_size = 224

    #model.load_state_dict(torch.load('resnet_weight.pt'))
    #model_dict = torch.load('resnet_weight.pt')]
    model.load_state_dict(torch.load('resnet_weight.pt'))

    model.eval()

    #print(model)

    last = time.time()

    #This displays on the screen what the predicted sign is
    sign = 'DEFAULT'
    while(True):
        ret, frame = cap.read()


        left_bound = int(bounded_box_width / 2 - dimension / 2)
        right_bound = left_bound + dimension
        top_bound = int(bounded_box_height / 2 - dimension / 2)
        bottom_bound = top_bound + dimension
        croppedFrame = frame[top_bound:bottom_bound, left_bound:right_bound]  # x and y are flipped idk why

        #print('{} to {} out of {}'.format(left_bound, right_bound, bounded_box_width))
        #print('{} to {} out of {}'.format(top_bound, bottom_bound, bounded_box_height))

        frame = cv2.rectangle(frame,(left_bound,top_bound),(right_bound,bottom_bound),0,5,8,0)

        #Gets rid of the info on the bottom bar
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        frame = cv2.putText(frame, sign, (10,30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if(cv2.waitKey(1) & 0xFF == ord(' ')): #27= esc key but I don't know where the constant is being kept  #ord(' ')):
            out = cv2.imwrite('capture.jpg',frame)
            break

        if time.time()-last > 1:
            last = time.time()
            input = croppedFrame
            #input = Image.fromarray(input)

            #input = io.imread('.\\data\\training_images\\combined\\a0.jpg')

            dataloader = DataLoader(TestDataset(input), batch_size=1, shuffle=False, num_workers=0)
            for i in dataloader:
                outputs = model(i)
                _, preds = torch.max(outputs, 1)
                print(outputs)
            sign = signs[preds.int()].upper()
            # input = cv2.resize(input,dsize=(224,224))
            # transform = transforms.ToTensor()
            # input = transform(input)
            # transform = transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
            # input = transform(input)

            #input = torch.from_numpy(input)

            # input = input.unsqueeze(0)
            # print(input.size())
            # print(input)
            # outputs = model(input)

            #print(outputs)
            #print(preds)
            #sign = str(torch.max(outputs, 1))

    cap.release()
    cv2.destroyAllWindows()
