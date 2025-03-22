from os import environ
environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sys import path
path.append('C:/Users/liamo/Documents/FYP/Training NN/Keras')
from cv2 import VideoCapture, setWindowProperty, cvtColor, imshow, waitKey, destroyAllWindows, filter2D, rectangle, putText, circle
from cv2 import WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN, COLOR_BGR2RGB, WND_PROP_TOPMOST, FILLED, FONT_HERSHEY_PLAIN
from mediapipe import solutions
from time import time
from imutils import resize
from keyboard import is_pressed
from numpy import array
from pygetwindow import getWindowsWithTitle
from pyscreenshot import grab
from openpyxl import load_workbook as load_wb
from keras_deep_learning import KerasDeepLearning


class HandTracker:
    def runTracker(createDataset, useKeras, letter):
        # Set Window variables
        WINDOW_NAME = "Hand Recognition"
        window_width = 500
        
        # Open video capture
        capture = VideoCapture(0)
        if not capture.isOpened():
            print("\nExternal camera unable to connect.\nConnecting to internal webcam")
            capture = VideoCapture(1)
            setWindowProperty(WINDOW_NAME, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN)
        
        # MediaPipe hands
        mpHands = solutions.hands
        hands = mpHands.Hands(False)
        mpDraw = solutions.drawing_utils

        # Set Dataset Creation variables
        dataset_count = 0
        
        # Timer for taking pictures to add to database
        end_timer_for_dataset = time() + 7 #Add 7 seconds to current time

        # Load Keras model
        model = KerasDeepLearning.load_model()


        # Loop while running capture program
        while True:
            # Save image to variable
            isSuccess, image = capture.read()

            # Resize image
            image = resize(image, width = window_width)


            # Sharpen Image
            sharpenImage = HandTracker.sharpenImage(image)


            # Convert image to RGB for easier hand tracking
            imageRGB = cvtColor(sharpenImage, COLOR_BGR2RGB)
            imageProcess = hands.process(imageRGB)
            results = imageProcess.multi_hand_landmarks


            if results:
                # for each point on the hand
                for handLandmarks in results:
                    #init coord array
                    list_of_coords = []

                    # for each 
                    for id, landmark in enumerate(handLandmarks.landmark):
                        height, width, coord = sharpenImage.shape
                        coordX, coordY = int(landmark.x * width), int(landmark.y * height)

                        list_of_coords.extend([coordX, coordY])
                        
                        # circle(sharpenImage, (coordX, coordY), 10, (0, 0, 0), FILLED)
                        
                    # mpDraw.draw_landmarks(sharpenImage, handLandmarks, mpHands.HAND_CONNECTIONS)


                # Check coords on model to predict letter
                if useKeras == True:
                    letter = HandTracker.predictUsingKeras(window_width, sharpenImage, model, list_of_coords, letter)


                # Take Image for Creating Dataset
                if createDataset == True:
                    current_second = time()

                    if (current_second >= end_timer_for_dataset):
                        dataset_count = HandTracker.createDataset(WINDOW_NAME, letter, dataset_count, list_of_coords)

            
            imshow(WINDOW_NAME, sharpenImage)
            setWindowProperty(WINDOW_NAME, WND_PROP_TOPMOST, 1)


            if waitKey(1) & is_pressed(" "):
                break


        capture.release()
        destroyAllWindows()


    def sharpenImage(image):
        # Sharpen Image
        kernel = array([
            [0, -1, 0],
            [-1, 5, -1], 
            [0, -1, 0]
        ])

        sharpenImage = filter2D(image, -1, kernel)

        return sharpenImage
    

    def createDataset(window_name, letter, count, list_of_coords):
        ## Screenshot screen
        # Get app window
        app_window = getWindowsWithTitle(window_name)[0]

        # Application window coordinates
        x1 = app_window.left
        y1 = app_window.top
        x2 = app_window.left + app_window.width
        y2 = app_window.top + app_window.height

        # Get picture count
        excel_file = r"Training NN\Datasets\Personal Dataset\personal_dataset.xlsx"
        workbook = load_wb(excel_file)
        worksheet = workbook[letter]
        row_count = str(worksheet.max_row - 1)

        # Screenshot window
        ss = grab(bbox = (x1, y1, x2, y2))
        ss.save(r"C:\Users\liamo\Documents\FYP\Training NN\Datasets\Personal Dataset\\" + letter + "\\" + letter + row_count + ".png")

        # Add to excel sheet
        HandTracker.addToExcel(letter, row_count, list_of_coords, excel_file)
    
        return count
        

    def addToExcel(letter, row_count, list_of_coords, excel_file_path):
        # Ensure Excel file is closed as permission error occurs otherwise

        dataset_path = r"Training NN\Datasets\Personal Dataset"

        # Keep all data on one workbook
        workbook = load_wb(excel_file_path)
        
        # Each letter gets individual sheet
        worksheet = workbook[letter]

        # Insert new coords
        list_of_coords.append(dataset_path + "\\" + letter + row_count + ".png")

        worksheet.append(list_of_coords)
        workbook.save(excel_file_path)


    def predictUsingKeras(window_width, image, model, list_of_coords, letter):
        predict_letter_timer = int(time())

        letter_box_width = 50
        letter_box_height = 50
        letter_box_x_point = int((window_width - letter_box_width) / 2)
        letter_box_y_point = 800

        if predict_letter_timer % 2 == 0:
            letter = KerasDeepLearning.predict_sign(model, list_of_coords)
        
        letter_box = rectangle(image, (letter_box_x_point, letter_box_y_point), ((letter_box_x_point + letter_box_width), (letter_box_y_point + letter_box_height)), (0, 0, 0), -1)
        putText(letter_box, letter, ((letter_box_x_point + 5), (letter_box_y_point + 47)), FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5)

        return letter
