import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import keyboard as kb

#Create video capture
capture = cv.VideoCapture(0)

#Set Window variables
WINDOW_NAME = "Hand Recognition"

#MediaPipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

#FPS set times
fpsStartTime = 0
fpsEndTime = 0


#Loop while running capture program
while True:
    #Save image to variable
    isSuccess, image = capture.read()


    #Sharpen Image
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1], 
        [0, -1, 0]
    ])
    sharpenImage = cv.filter2D(image, -1, kernel)


    #Convert image to RGB for easier hand tracking
    imageRGB = cv.cvtColor(sharpenImage, cv.COLOR_BGR2RGB)
    imageProcess = hands.process(imageRGB)
    results = imageProcess.multi_hand_landmarks


    if results:
        #for each point on the hand
        for handLandmarks in results:
            #for each 
            for id, landmark in enumerate(handLandmarks.landmark):
                height, width, coord = sharpenImage.shape
                coordX, coordY = int(landmark.x * width), int(landmark.y * height)
                
                cv.circle(sharpenImage, (coordX, coordY), 10, (0, 0, 0), cv.FILLED)
                
            mpDraw.draw_landmarks(sharpenImage, handLandmarks, mpHands.HAND_CONNECTIONS)


    fpsEndTime = time.time()
    fps = 1 / (fpsEndTime - fpsStartTime)
    fpsStartTime = fpsEndTime


    cv.rectangle(sharpenImage, (2, 2), (50, 30), (0, 0, 0), cv.FILLED)
    cv.putText(sharpenImage, (str(int(fps))), (5, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)


    cv.imshow(WINDOW_NAME, sharpenImage)
    cv.moveWindow(WINDOW_NAME, 650, 250)


    if cv.waitKey(1) & kb.is_pressed(" "):
        break


capture.release()
cv.destroyAllWindows()
