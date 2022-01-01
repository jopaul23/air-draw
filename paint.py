import cv2 as cv
import numpy as np
import mediapipe as mp

capture  =cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands =  mpHands.Hands()

mpDraw =  mp.solutions.drawing_utils

isTrue,frame =  capture.read()
frame_width  = frame.shape[0]
frame_height = frame.shape[1]

blank = np.zeros((frame_width,frame_height),dtype='uint8')

four_cordinates = np.array([0,0],dtype='uint8')
eight_cordinates =np.array([frame_width,frame_height],dtype='int')

while True:
    isTrue, frame  = capture.read()
    frame_rgb  = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    results =  hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            
            for point , lm in enumerate(landmark.landmark):
                if point==4:
                    four_cordinates = lm.x*frame_width,lm.y*frame_height
                if point==8:
                    eight_cordinates = lm.x*frame_width,lm.y*frame_height
                    
                square_diff = np.square(np.array([four_cordinates[0]-eight_cordinates[0],four_cordinates[1]-eight_cordinates[1]]))
                displacement  =  square_diff[0]+square_diff[1]
                if displacement<3000:
                    blank[int(four_cordinates[1]):int(four_cordinates[1])+5,int(four_cordinates[0]):int(four_cordinates[0])+5] = 255
                print(displacement)
            mpDraw.draw_landmarks(frame,landmark)
    cv.imshow('artboard',cv.flip(blank,1))
    cv.imshow('image',cv.flip(frame,1))
    cv.waitKey(10)