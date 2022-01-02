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
frame_width  = 700
frame_height = 500
print(frame_width,frame_height)
blank = np.zeros((frame_height*5,frame_width,3),dtype='uint8')
page_height = 0

four_cordinates = np.array([0,0])
previous_cordinates =np.array([0,0])
eight_cordinates =np.array([frame_width,frame_height])

color = [0,255,255]
marker_size =  6

text =  "yellow marker selected"

free_mode = False


while True:
    isTrue, frame  = capture.read()
    frame = cv.resize(frame,(700,500))
    frame_rgb  = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    results =  hands.process(frame_rgb)
    flag = False
    if not free_mode:
        if results.multi_hand_landmarks:
            for landmark in results.multi_hand_landmarks:
                
                for point , lm in enumerate(landmark.landmark):
                    if point==4:
                        four_cordinates =lm.x*frame_width,lm.y*frame_height
                    if point==8:
                        eight_cordinates = lm.x*frame_width,lm.y*frame_height
                    finger_displacement = findDisplacement(four_cordinates,eight_cordinates)
                    if finger_displacement<2000:
                        cv.circle(frame,np.uint(tuple(four_cordinates)),15,color=tuple(color),thickness=4 )
                        blank[int(four_cordinates[1]+page_height):int(four_cordinates[1])+marker_size+page_height,int(four_cordinates[0]):int(four_cordinates[0])+marker_size] = color
                        if flag:
                            cv.line(blank,(np.uint(previous_cordinates[0]),np.uint(previous_cordinates[1]+page_height)),(np.uint(four_cordinates[0]),np.uint(four_cordinates[1]+page_height)),color,thickness=int(marker_size/2))
                        flag =  True
                        previous_cordinates  = four_cordinates
                mpDraw.draw_landmarks(frame,landmark)
    frame =  cv.bitwise_or(frame,blank[page_height:page_height+500,:])
    blank_fliped = cv.flip(blank,1)
    
    frame = cv.flip(frame,1)
    cv.putText(frame,text,(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2)

    cv.imshow('artboard',blank_fliped[int(page_height/2):700+int(page_height/2),:])
    cv.imshow('image',frame)
    key = cv.waitKey(10)

    if key == ord('c'):#clear
        text =  "frame cleared"
        blank = np.zeros((frame_height*5,frame_width,3),dtype='uint8')
    elif key == ord('r'):#red
        text =  "red marker selected"
        marker_size =  6
        color = [0,0,255]
    elif key == ord('w'):#wite
        text =  "white marker selected"
        marker_size =  6
        color = [255,255,255]
    elif key == ord('g'):#green
        text =  "green marker selected"
        marker_size =  6
        color = [0,255,0]
    elif key == ord('b'):#blue
        marker_size =  6
        text =  "blue marker selected"
        color = [255,0,0]
    elif key == ord('y'):#yellow
        marker_size =  6
        text =  "yellow marker selected"
        color = [0,255,255]
    elif key == ord('e'):#eraser
        marker_size =  30
        text =  "eraser selected"
        color = [0,0,0]
    elif key == ord('f'):#eraser
        free_mode = not free_mode
        if free_mode:
            text =  "free mode selected"
        else:
            text =  "free mode turned off"
    if key == ord('d'):#scroll down artboard
        if page_height<frame_height*4:
            page_height+=10
    elif key == ord('u'):#scroll up artboard
        if page_height>0:
            page_height-=10
    elif key == ord('s'):#save as jpg
        try: 
            cv.imwrite('out.jpg',blank_fliped[0:page_height+500])
            text =  "successfully saved"
            print("saved successfully")
        except:
            print("saving failed")