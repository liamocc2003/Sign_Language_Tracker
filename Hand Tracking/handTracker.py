import cv2 as cv
import mediapipe as mp
import time
import datetime as dt
import numpy as np
import pyautogui as pagui
import keyboard as kb


class HandTracker:

    def runTracker(createDataset, letter):
        #Create video capture
        capture = cv.VideoCapture(0)

        #Set Window variables
        WINDOW_NAME = "Hand Recognition"

        #MediaPipe hands
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(False)
        mpDraw = mp.solutions.drawing_utils

        #Set Dataset Creation variables
        count = 0

        #Set FPS times
        fpsStartTime = 0
        fpsEndTime = 0


        #Loop while running capture program
        while True:
            #Save image to variable
            isSuccess, image = capture.read()


            #Sharpen Image
            sharpenImage = HandTracker.sharpenImage(image)


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


                #Take Image for Creating Dataset
                count = HandTracker.createDataset(createDataset, letter, count)

            
            #Show FPS
            fpsStartTime = HandTracker.FPSCounter(sharpenImage, fpsStartTime)


            cv.imshow(WINDOW_NAME, sharpenImage)
            cv.moveWindow(WINDOW_NAME, 650, 250)
            cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_TOPMOST, 1)


            if cv.waitKey(1) & kb.is_pressed(" "):
                break


        capture.release()
        cv.destroyAllWindows()


    def sharpenImage(image):
        #Sharpen Image
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1], 
            [0, -1, 0]
        ])

        sharpenImage = cv.filter2D(image, -1, kernel)

        return sharpenImage
    

    def FPSCounter(image, fpsStartTime):
        #Get FPS
        fpsEndTime = time.time()
        fps = 1 / (fpsEndTime - fpsStartTime)
        fpsStartTime = fpsEndTime

        #Create Visual pieces to show fps
        cv.rectangle(image, (2, 2), (50, 30), (0, 0, 0), cv.FILLED)
        cv.putText(image, (str(int(fps))), (5, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        return fpsStartTime
    

    def createDataset(createDataset, letter, count):
        #Get current time
        currentDatetime = dt.datetime.now()
        currentSecond = int(currentDatetime.strftime("%S"))

        if (createDataset == 'y'):
            if (currentSecond % 5 == 0):
                pagui.screenshot(letter + str(count) + ".png")
                count = count + 1
                time.sleep(1)
            
            return count