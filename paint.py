import cv2 as cv
import numpy as np
import mediapipe as mp
from numpy.lib.type_check import imag

def findDisplacement(p1,p2):
    square_diff = np.square(np.array([p1[0]-p2[0],p1[1]-p2[1]]))
    displacement  =  square_diff[0]+square_diff[1]
    return displacement
capture  =cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands =  mpHands.Hands()

mpDraw =  mp.solutions.drawing_utils

isTrue,frame =  capture.read()
frame_width  = frame.shape[1]
frame_height = frame.shape[0]
print(frame_width,frame_height)
blank = np.zeros((frame_height,frame_width),dtype='uint8')

four_cordinates = np.array([0,0])
previous_cordinates =np.array([0,0])
eight_cordinates =np.array([frame_width,frame_height])

while True:
    isTrue, frame  = capture.read()
    frame_rgb  = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    results =  hands.process(frame_rgb)
    flag = False
    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            
            for point , lm in enumerate(landmark.landmark):
                if point==4:
                    four_cordinates =lm.x*frame_width,lm.y*frame_height
                if point==8:
                    eight_cordinates = lm.x*frame_width,lm.y*frame_height
                finger_displacement = findDisplacement(four_cordinates,eight_cordinates)
                if finger_displacement<2500:
                    blank[int(four_cordinates[1]):int(four_cordinates[1])+5,int(four_cordinates[0]):int(four_cordinates[0])+5] = 255
                    if flag:
                        cv.line(blank,(np.uint(previous_cordinates[0]),np.uint(previous_cordinates[1])),(np.uint(four_cordinates[0]),np.uint(four_cordinates[1])),(255),thickness=2)
                    flag =  True
                    previous_cordinates  = four_cordinates
            mpDraw.draw_landmarks(frame,landmark)
    blank_bgr =  cv.cvtColor(blank,cv.COLOR_GRAY2BGR)
    frame =  cv.bitwise_or(frame,blank_bgr)
    cv.imshow('artboard',cv.flip(blank,1))
    cv.imshow('image',cv.flip(frame,1))
    cv.waitKey(10)

