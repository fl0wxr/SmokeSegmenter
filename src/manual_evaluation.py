import data_tools
import json
import numpy as np
import visuals
import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import tkinter.font as font


class manual_evaluation_ui:

    def __init__(self, BLACKLIST_FP):

        self.BLACKLIST_FP = BLACKLIST_FP

        self.window = tk.Tk()

        # Set window dimensions (width x height)
        self.window_width = 1410
        self.window_height = 700

        # Get the screen width and height
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()

        # Calculate the x and y position to center the window on the screen
        self.x_position = (self.screen_width - self.window_width) // 2
        self.y_position = (self.screen_height - self.window_height) // 2

        # Set the window size and position
        self.window.geometry(f"{self.window_width}x{self.window_height}+{self.x_position}+{self.y_position}")

    def build(self, img, mask, combined, res, img_fp, mask_fp, n_smoke_pixels, contains_smoke, completion_status):

        def top_text():

            top_margin = left_margin = right_margin = bottom_margin = 10
            text_frame_width = self.window_width - right_margin - left_margin
            text_frame_height = 90 - bottom_margin - top_margin

            text_frame = tk.Frame(self.window, width = text_frame_width, height = text_frame_height, bg = 'white')
            text_frame.place(x = left_margin, y = top_margin)

            text_label = tk.Label(text_frame, text = 'Image path: %s\nMask path: %s'%(img_fp, mask_fp), wraplength = text_frame_width, anchor = 'w', justify = 'left')
            text_label.pack()

        def left_text():

            top_margin = left_margin = right_margin = bottom_margin = 10
            text_frame_width = self.window_width - combined_outer_frame_shape[0] - 3 * right_margin - left_margin
            text_frame_height = self.window_width - 4 * 90 - 40

            text_frame = tk.Frame(self.window, width = text_frame_width, height = text_frame_height, bg = 'white')
            text_frame.place(x = left_margin, y = 90)

            text_label = tk.Label(text_frame, text = 'Resolution: %dx%d\nContains Smoke: %s\nNumber of smoke px:\n%d\nStatus: %.2f%%'%(res[1], res[0], contains_smoke, n_smoke_pixels, completion_status), wraplength = text_frame_width, anchor = 'w', justify = 'left')
            text_label.pack()

        def upper_button():

            top_margin = left_margin = right_margin = bottom_margin = 10
            button_frame_width = 170
            button_frame_height = 100

            button_frame = tk.Frame(self.window, width = button_frame_width, height = button_frame_height)
            button_frame.place(x = left_margin, y = self.window_height - 2* (button_frame_height + bottom_margin))

            button = tk.Button(button_frame, text = 'NEXT [N]', width = button_frame_width // 10, height = button_frame_height // 10 - 5, bg = 'green', command = Next, font = DefaultFont, wraplength = button_frame_width - 20)
            button.place(x = -33, y = -8)

        def bottom_button():

            top_margin = left_margin = right_margin = bottom_margin = 10
            button_frame_width = 170
            button_frame_height = 100

            button_frame = tk.Frame(self.window, width = button_frame_width, height = button_frame_height)
            button_frame.place(x = left_margin, y = self.window_height - (button_frame_height + bottom_margin))

            button = tk.Button(button_frame, text = 'ADD TO BLACKLIST [B]', width = button_frame_width // 10, height = button_frame_height // 10 - 5, bg = 'maroon', command = Add2Blacklist, font = DefaultFont, wraplength = button_frame_width - 30)
            button.place(x = -33, y = -8)

        def Add2Blacklist():

            with open(self.BLACKLIST_FP, mode = 'a') as file:
                file.write(img_fp + ', ' + mask_fp + '\n')

            Next()

        def Next():
            self.window.quit()
            return False

        def event_handler(event):
            if event.char == 'q':
                exit()
            elif event.char == 'n':
                Next()
            elif event.char == 'b':
                Add2Blacklist()

        DefaultFont = font.Font(family = 'Lato', size = '15', weight = 'bold')

        ## ! Build right image frame: Begin

        ## In this method, all shapes are considered as (width, height)
        combined_shape = [combined.shape[1], combined.shape[0]]

        right_margin = 10
        bottom_margin = 10

        combined_outer_frame_shape = (800, 600)

        outer_combined_frame = tk.Frame(self.window, width = combined_outer_frame_shape[0], height = combined_outer_frame_shape[1], bg = 'black')
        outer_combined_frame.place(x = self.window_width - combined_outer_frame_shape[0] - right_margin, y = self.window_height - combined_outer_frame_shape[1] - bottom_margin)

        ## Resize image while keeping aspect ratio
        combined_aspect_ratio = combined_shape[1] / combined_shape[0]

        ## Adjust image and inner frame resolution, to fit exactly inside the outer frame
        combined_shape[0] = combined_outer_frame_shape[0]
        combined_shape[1] = int(combined_aspect_ratio * combined_shape[0])
        if combined_shape[1] > combined_outer_frame_shape[1]:
            combined_shape[1] = combined_outer_frame_shape[1]
            combined_shape[0] = int(combined_shape[1] / combined_aspect_ratio)
        combined = cv2.resize(combined, (combined_shape[0], combined_shape[1]))
        combined = Image.fromarray(combined)
        combined_tk = ImageTk.PhotoImage(combined)
        combined_inner_frame_shape = \
        (
            combined_shape[0],
            combined_shape[1]
        )

        ## Creating a child frame inside the outer image frame. Placing coordinate system places (0, 0) at the upper left edge of the outer frame.
        inner_combined_frame = tk.Frame(outer_combined_frame, width = combined_inner_frame_shape[0], height = combined_inner_frame_shape[1], bg="white")
        inner_combined_frame.place(x = combined_outer_frame_shape[0] // 2 - combined_inner_frame_shape[0] // 2, y = combined_outer_frame_shape[1] // 2 - combined_inner_frame_shape[1] // 2)

        combined_label = tk.Label(inner_combined_frame, image = combined_tk, bg = 'white')
        combined_label.grid(column = 0, row = 0)

        ## ! Build right image frame: End

        ## ! Build left top image frame: Begin

        ## In this method, all shapes are considered as (width, height)
        img_shape = [img.shape[1], img.shape[0]]

        right_margin = 10
        bottom_margin = 10

        img_outer_frame_shape = (400, 300)

        outer_img_frame = tk.Frame(self.window, width = img_outer_frame_shape[0], height = img_outer_frame_shape[1], bg = 'black')
        outer_img_frame.place(x = self.window_width - img_outer_frame_shape[0] - 2 * right_margin - combined_outer_frame_shape[0], y = self.window_height - 2* img_outer_frame_shape[1] - bottom_margin)

        ## Resize image while keeping aspect ratio
        img_aspect_ratio = img_shape[1] / img_shape[0]

        ## Adjust image and inner frame resolution, to fit exactly inside the outer frame
        img_shape[0] = img_outer_frame_shape[0]
        img_shape[1] = int(img_aspect_ratio * img_shape[0])
        if img_shape[1] > img_outer_frame_shape[1]:
            img_shape[1] = img_outer_frame_shape[1]
            img_shape[0] = int(img_shape[1] / img_aspect_ratio)
        img = cv2.resize(img, (img_shape[0], img_shape[1]))
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img)
        img_inner_frame_shape = \
        (
            img_shape[0],
            img_shape[1]
        )

        ## Creating a child frame inside the outer image frame. Placing coordinate system places (0, 0) at the upper left edge of the outer frame.
        inner_img_frame = tk.Frame(outer_img_frame, width = img_inner_frame_shape[0], height = img_inner_frame_shape[1], bg="white")
        inner_img_frame.place(x = img_outer_frame_shape[0] // 2 - img_inner_frame_shape[0] // 2, y = img_outer_frame_shape[1] // 2 - img_inner_frame_shape[1] // 2)

        img_label = tk.Label(inner_img_frame, image = img_tk, bg = 'white')
        img_label.grid(column = 0, row = 0)

        ## ! Build left top image frame: End

        ## ! Build left bottom image frame: Begin

        ## In this method, all shapes are considered as (width, height)
        mask_shape = [mask.shape[1], mask.shape[0]]

        right_margin = 10
        bottom_margin = 10

        mask_outer_frame_shape = (400, 300)

        outer_mask_frame = tk.Frame(self.window, width = mask_outer_frame_shape[0], height = mask_outer_frame_shape[1], bg = 'black')
        outer_mask_frame.place(x = self.window_width - mask_outer_frame_shape[0] - 2 * right_margin - combined_outer_frame_shape[0], y = self.window_height - mask_outer_frame_shape[1] - bottom_margin)

        ## Resize image while keeping aspect ratio
        mask_aspect_ratio = mask_shape[1] / mask_shape[0]

        ## Adjust image and inner frame resolution, to fit exactly inside the outer frame
        mask_shape[0] = mask_outer_frame_shape[0]
        mask_shape[1] = int(mask_aspect_ratio * mask_shape[0])
        if mask_shape[1] > mask_outer_frame_shape[1]:
            mask_shape[1] = mask_outer_frame_shape[1]
            mask_shape[0] = int(mask_shape[1] / mask_aspect_ratio)
        mask = cv2.resize(mask, (mask_shape[0], mask_shape[1]))
        mask = Image.fromarray(255 * mask)
        mask_tk = ImageTk.PhotoImage(mask)
        mask_inner_frame_shape = \
        (
            mask_shape[0],
            mask_shape[1]
        )

        ## Creating a child frame inside the outer image frame. Placing coordinate system places (0, 0) at the upper left edge of the outer frame.
        inner_mask_frame = tk.Frame(outer_mask_frame, width = mask_inner_frame_shape[0], height = mask_inner_frame_shape[1], bg="white")
        inner_mask_frame.place(x = mask_outer_frame_shape[0] // 2 - mask_inner_frame_shape[0] // 2, y = mask_outer_frame_shape[1] // 2 - mask_inner_frame_shape[1] // 2)

        mask_label = tk.Label(inner_mask_frame, image = mask_tk, bg = 'white')
        mask_label.grid(column = 0, row = 0)

        ## ! Build left bottom image frame: End

        ## Top text
        top_text()

        ## Left text
        left_text()

        ## Bottom button
        bottom_button()

        ## Bottom button
        upper_button()

        self.window.bind('<KeyPress>', event_handler)
        ## Must be on the same function otherwise garbage collector can ruin features
        self.window.mainloop()

        return True

def manual_evaluation_sequence(BLACKLIST_FP = '../blacklisted_instances.list', paths_fp = '../paths.json'):

    def session_save(mask_fp):

        with open(session_fp, 'w') as file:
            file.write(mask_fp)

    with open(file = paths_fp, mode = 'r') as json_file:
        paths_json = json.load(json_file)

    session_fp = '../session.path'

    if os.path.isfile(session_fp):
        print('W: Previous session found')
        start_over_trigger = input('Start over? Yes [Y] or No [N]\n> ')
        while start_over_trigger not in {'Y', 'N'}:
            start_over_trigger = input('Enter a proper answer\n> ')
        start_over_trigger = start_over_trigger == 'Y'
        if start_over_trigger:
            if os.path.isfile(BLACKLIST_FP):
                os.remove(BLACKLIST_FP)
            os.remove(session_fp)

    if os.path.isfile(session_fp):
        print('Loading previous session')
        with open(session_fp, 'r') as file:
            mask_session_fp = file.read()
        load_session_trigger = True
    else:
        print('Creating a new session')
        load_session_trigger = False

    dataset_dp = paths_json['ssmoke_data_dp']
    if dataset_dp[-1] != '/': dataset_dp += '/'

    data = data_tools.SegmData(dataset_dp = dataset_dp)

    ui = manual_evaluation_ui(BLACKLIST_FP = BLACKLIST_FP)

    while next(data):

        img = data.img
        mask = data.mask
        res = img.shape[:-1]
        img_fp = data.img_fp
        mask_fp = data.mask_fp
        n_smoke_pixels = data.n_smoke_pixels
        contains_smoke = 'Positive' if n_smoke_pixels > 0 else 'Negative'

        if load_session_trigger:
            if mask_fp == mask_session_fp:
                load_session_trigger = False
            else:
                continue

        combined = visuals.combine_img_mask(img = img, mask = mask)

        completion_status = 100*(data.INSTANCE_IDX+1)/data.n_instances

        ui.build(img = img, mask = mask, combined = combined, res = res, img_fp = img_fp, mask_fp = mask_fp, n_smoke_pixels = n_smoke_pixels, contains_smoke = contains_smoke, completion_status = completion_status)

        print('Completion status: %.2f%%'%(completion_status))

        session_save(mask_fp = mask_fp)


if __name__ == '__main__':

    manual_evaluation_sequence()

