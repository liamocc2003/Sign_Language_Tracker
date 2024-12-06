import sys
sys.path.append('C:/Users/liamo/Documents/FYP/Hand Tracking')
from handTracker import HandTracker
import tkinter as tk


### tkinter functions
def dataset_toggle():
    if add_dataset.get() == 1:
        #Checkbox is selected
        letter_label.config(state = "normal")
        letter_entry.config(state = "normal")
    else:
        #Checkbox is not selected
        letter_label.config(state = "disabled")
        letter_entry.config(state = "disabled")


def runTracker():
    if add_dataset.get() == 1:
        create_dataset = True
        letter = letter_entry.get()
        letter = letter.capitalize()

    else:
        create_dataset = False
        letter = ""

    exitstatus = HandTracker.runTracker(create_dataset, letter)
    if exitstatus == 0:
        form.destroy()


### Create tkinter form
form = tk.Tk()
form.title("Hand Tracker")
form.geometry("500x500")

##Create Form Objects
#Title
title_label = tk.Label(form, text = "Use hand Tracker", font = "50")

#Expand the Database
add_dataset = tk.IntVar()
expand_dataset_checkbox = tk.Checkbutton(
    form, 
    text = "Do you want to expand the dataset?", 
    variable = add_dataset, 
    command = dataset_toggle
)

#Choose Letter
letter_label = tk.Label(form, text = "Enter the letter you want", state = "disabled")
letter_entry = tk.Entry(form, textvariable = "", state = "disabled")

#Run Tracker Btn
tracker_btn = tk.Button(form, text = "Run Tracker", command = runTracker)


##Add Form Objects
title_label.pack()
expand_dataset_checkbox.pack()
letter_label.pack()
letter_entry.pack()
tracker_btn.pack()


#Run loop
form.mainloop()
