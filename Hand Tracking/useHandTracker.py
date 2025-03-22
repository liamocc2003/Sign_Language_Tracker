from os import environ
environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from handTracker import HandTracker
from tkinter import Tk, messagebox
from tkinter import Label, IntVar, Checkbutton, Entry, Button
from screeninfo import get_monitors


### tkinter functions
# Choose whether to add to dataset
def dataset_toggle():
    if add_dataset.get() == 1:
        # Checkbox is selected
        use_keras_model_checkbox.config(state = "disabled")

        letter_label.config(state = "normal")
        letter_entry.config(state = "normal")
        letter_entry.focus()
    else:
        # Checkbox is not selected
        use_keras_model_checkbox.config(state = "normal")

        letter_label.config(state = "disabled")
        letter_entry.config(state = "disabled")
        tracker_btn.focus()


def toggle_keras_model():
    if use_keras_model.get() == 1:
        # Checkbox is selected
        expand_dataset_checkbox.config(state = "disabled")

    else:
        # Checkbox is not selected
        expand_dataset_checkbox.config(state = "normal") 


# Run the hand tracker
def runTracker():
    create_dataset = False
    use_keras = False
    letter = ""

    if add_dataset.get() == 1:
        create_dataset = True
        letter = letter_entry.get()
        letter = letter.capitalize()

        letter_length = len(letter)
        if letter_length < 1:
            messagebox.showwarning("No letter provided", "Please enter a letter when adding to the dataset.")
            return
        
        elif letter_length > 1:
            messagebox.showwarning("More than one character entered", "Please enter only one character when adding to the dataset. ")
            return
        

    elif use_keras_model.get() == 1:
        use_keras = True


    exitstatus = HandTracker.runTracker(create_dataset, use_keras, letter)
    if exitstatus == 0:
        form.destroy()


### Create tkinter form
form = Tk()
form.title("Hand Tracker")

# Get screen size
for monitor in get_monitors():
    if monitor.is_primary == True:
        screen_width = monitor.width
        screen_height = monitor.height

form_width = 400
form_height = 500
form_x_position = int((screen_width - form_width) / 2)
form_y_position = int((screen_height - form_height) / 2)

form.geometry(str(form_width) + "x" + str(form_height) + "+" + str(form_x_position) + "+" + str(form_y_position))


## Create Form Objects
font_name = "Helvetica"
# Title
title_label = Label(form, text = "Hand Tracker", font = (font_name, 20, 'bold'))


# Use Keras Model
use_keras_model = IntVar()
use_keras_model_checkbox = Checkbutton(
    form, 
    text = "Do you want to use the Keras model?", 
    font = (font_name, 14),
    variable = use_keras_model, 
    command = toggle_keras_model
)


# Expand the Database
add_dataset = IntVar()
expand_dataset_checkbox = Checkbutton(
    form, 
    text = "Do you want to expand the dataset?",
    font = (font_name, 14),
    variable = add_dataset, 
    command = dataset_toggle
)

# Choose Letter
letter_label = Label(form, text = "Enter the letter you want:", state = "disabled", font = (font_name, 12))
letter_entry = Entry(form, textvariable = "", state = "disabled", bd = 3, font = (font_name, 12), width = 10)


# Run Tracker Btn
tracker_btn = Button(form, text = "Run Tracker", command = runTracker, width = 15, height = 2, bd = 5, font = (font_name, 10, 'bold'))


## Add Form Objects
title_label.place(x = 110, y = 10)
use_keras_model_checkbox.place(x = 20, y = 70)
expand_dataset_checkbox.place(x = 20, y = 185)
letter_label.place(x = 40, y = 225)
letter_entry.place(x = 50, y = 250)
tracker_btn.place(x = 140, y = 430)


# Run loop
form.mainloop()
