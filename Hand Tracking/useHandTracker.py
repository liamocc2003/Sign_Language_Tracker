import sys
sys.path.append('C:/Users/liamo/Documents/FYP/Hand Tracking')
from handTracker import HandTracker

createDataset = input("Do you want to expand the database? \n'y' or 'n': ")
letter = ''

if (createDataset == 'y'):

    while (len(letter) != 1):
        letter = input("What letter do you want to expand? \nLetter: ")

    
HandTracker.runTracker(createDataset, letter)