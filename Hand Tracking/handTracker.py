import cv2 as cv
import imutils as imu
import mediapipe as mp
import time
import numpy as np
import pygetwindow as pygw
import pyscreenshot as pyss
import keyboard as kb
from openpyxl import load_workbook as load_wb


class HandTracker:

    def runTracker(createDataset, letter):
        #Set Window variables
        WINDOW_NAME = "Hand Recognition"
        
        #Open video capture
        capture = cv.VideoCapture(0)
        if not capture.isOpened():
            print("\nExternal camera unable to connect.\nConnecting to internal webcam")
            capture = cv.VideoCapture(1)
            cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        
        #MediaPipe hands
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(False)
        mpDraw = mp.solutions.drawing_utils

        #Set Dataset Creation variables
        dataset_count = 0

        #Set FPS times
        fpsStartTime = 0
        
        #Timer for taking pictures to add to database
        end_timer_for_dataset = time.time() + 7 #Add 7 seconds to current time


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
                    current_second = time.time()

                    if (current_second >= end_timer_for_dataset):
                        dataset_count = HandTracker.createDataset(WINDOW_NAME, letter, dataset_count, list_of_coords)

            
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
    

    def createDataset(window_name, letter, count, list_of_coords):
        ##Screenshot screen
        #Get app window
        app_window = pygw.getWindowsWithTitle(window_name)[0]

        #Application window coordinates
        x1 = app_window.left
        y1 = app_window.top
        x2 = app_window.left + app_window.width
        y2 = app_window.top + app_window.height

        #Get picture count
        excel_file = r"Training NN\Datasets\Personal Dataset\personal_dataset.xlsx"
        workbook = load_wb(excel_file)
        worksheet = workbook[letter]
        row_count = str(worksheet.max_row - 1)

        #Screenshot window
        ss = pyss.grab(bbox = (x1, y1, x2, y2))
        ss.save(r"C:\Users\liamo\Documents\FYP\Training NN\Datasets\Personal Dataset\\" + letter + "\\" + letter + row_count + ".png")

        #Add to excel sheet
        HandTracker.addToExcel(letter, row_count, list_of_coords, excel_file)
    
        return count
        

    def addToExcel(letter, row_count, list_of_coords, excel_file_path):
        #Ensure Excel file is closed as permission error occurs otherwise

        dataset_path = r"Training NN\Datasets\Personal Dataset"

        # Keep all data on one workbook
        workbook = load_wb(excel_file_path)
        
        # Each letter gets individual sheet
        worksheet = workbook[letter]

        #Insert new coords
        list_of_coords.append(dataset_path + "\\" + letter + row_count + ".png")

        worksheet.append(list_of_coords)
        workbook.save(excel_file_path)
