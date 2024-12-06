import cv2 as cv
import imutils as imu
import mediapipe as mp
import time
import datetime as dt
import numpy as np
import pyautogui as pagui
import keyboard as kb
from openpyxl import load_workbook as load_wb


class HandTracker:

    def runTracker(createDataset, letter):
        #Create video capture
        capture = cv.VideoCapture(1)
        if not capture.isOpened():
            print("External camera unable to connect. Connecting to internal webcam...")
            capture = cv.VideoCapture(0)
            if not capture.isOpened():
                print("Internal camera not available. Please connect a camera to continue. \nExiting...")
                return 0
            cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)


        #Set Window variables
        WINDOW_NAME = "Hand Recognition"
        
        #MediaPipe hands
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(False)
        mpDraw = mp.solutions.drawing_utils

        #Set Dataset Creation variables
        dataset_count = 0

        #Set FPS times
        fpsStartTime = 0


        #Loop while running capture program
        while True:
            #Save image to variable
            isSuccess, image = capture.read()

            #Resize image
            image = imu.resize(image, width = 500)


            #Sharpen Image
            sharpenImage = HandTracker.sharpenImage(image)


            #Convert image to RGB for easier hand tracking
            imageRGB = cv.cvtColor(sharpenImage, cv.COLOR_BGR2RGB)
            imageProcess = hands.process(imageRGB)
            results = imageProcess.multi_hand_landmarks


            if results:
                #for each point on the hand
                for handLandmarks in results:
                    #init coord array
                    list_of_coords = []

                    #for each 
                    for id, landmark in enumerate(handLandmarks.landmark):
                        height, width, coord = sharpenImage.shape
                        coordX, coordY = int(landmark.x * width), int(landmark.y * height)

                        list_of_coords.extend([coordX, coordY])
                        
                        cv.circle(sharpenImage, (coordX, coordY), 10, (0, 0, 0), cv.FILLED)
                        
                    mpDraw.draw_landmarks(sharpenImage, handLandmarks, mpHands.HAND_CONNECTIONS)


                #Take Image for Creating Dataset
                if (createDataset == True):
                    dataset_count = HandTracker.createDataset(letter, dataset_count, list_of_coords)

            
            #Show FPS
            fpsStartTime = HandTracker.FPSCounter(sharpenImage, fpsStartTime)

            
            cv.imshow(WINDOW_NAME, sharpenImage)
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
    

    def createDataset(letter, count, list_of_coords):
        #Get current time
        currentDatetime = dt.datetime.now()
        currentSecond = int(currentDatetime.strftime("%S"))

        if (currentSecond % 5 == 0):
            pagui.screenshot(r"C:\Users\liamo\Documents\FYP\Training NN\Datasets\Personal Dataset\\" + letter + str(count) + ".png")
            count = count + 1
            HandTracker.addToExcel(letter, list_of_coords)
            cv.waitKey(1000)
        
        return count
        

    def addToExcel(letter, list_of_coords):
        dataset_path = r"Training NN\Datasets\Personal Dataset"
        excel_file = dataset_path + r"\personal_dataset.xlsx"

        workbook = load_wb(excel_file)
        worksheet = workbook.active

        list_of_coords.insert(0, letter)
        list_of_coords.append(dataset_path + "\\" + letter + "1.png")

        worksheet.append(list_of_coords)
        workbook.save(excel_file)
