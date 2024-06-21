from tkinter import Tk, Canvas, Spinbox, Button, Frame, ttk, DoubleVar
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import mgac
import os
import argparse


class ImageCanvas(Canvas):
    def __init__(self, master, image, image_path, resolution, **kwargs):
        super().__init__(master=master, **kwargs)
        self.bind("<Button-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        self.center = None
        self.ellipse = None
        self.image = image
        self.image_path = image_path  # Store the image path
        self.resolution = resolution  # Store the resolution of the image
        self.ellipse_axes = (
            None  # List to store the coordinates of the points on the ellipse
        )
        self.mask = None

    def on_click(self, event):
        if self.ellipse:
            self.delete(self.ellipse)
        self.center = (event.x, event.y)

    def on_drag(self, event):
        if self.ellipse:
            self.delete(self.ellipse)
        cx, cy = self.center
        self.ellipse = self.create_oval(
            cx - abs(cx - event.x),
            cy - abs(cy - event.y),
            cx + abs(cx - event.x),
            cy + abs(cy - event.y),
            outline="red",
        )
        self.ellipse_axes = (abs(cx - event.x), abs(cy - event.y))


def display_image(image_path, max_size=(1280, 720)):
    # Open an image file
    img = Image.open(image_path)
    resolution = img.size
    # Resize the image to fit within the specified maximum size
    img.thumbnail(max_size, Image.LANCZOS)
    # Create a Tkinter window
    window = Tk()
    window.title("MGAC Image Segmentation Tool")
    # Convert the Image object to a PhotoImage object
    img_tk = ImageTk.PhotoImage(img)
    # Create a canvas and add it to the window
    canvas = ImageCanvas(
        window, img_tk, image_path, resolution, width=img.size[0], height=img.size[1]
    )
    canvas.pack()

    # Add the spinboxes and button
    frame = Frame(window)
    frame.pack()

    balloon_label = ttk.Label(frame, text="Balloon:")
    balloon_label.pack(side="left")
    ballon_var = DoubleVar(value=1.0)
    balloon_spinbox = Spinbox(
        frame, from_=0.5, to=1.5, increment=0.1, format="%.1f", textvariable=ballon_var
    )
    balloon_spinbox.pack(side="left")

    threshold_label = ttk.Label(frame, text="Threshold:")
    threshold_label.pack(side="left")
    threshold_var = DoubleVar(value=0.95)
    threshold_spinbox = Spinbox(
        frame,
        from_=0.75,
        to=1.15,
        increment=0.1,
        format="%.2f",
        textvariable=threshold_var,
    )
    threshold_spinbox.pack(side="left")

    button = Button(
        frame,
        text="Execute MGAC",
        command=lambda: execute_mgac(
            canvas, balloon_spinbox, threshold_spinbox, mode="mgac"
        ),
    )
    button.pack(side="left")

    button = Button(
        frame,
        text="Execute Flood Fill",
        command=lambda: execute_mgac(
            canvas, balloon_spinbox, threshold_spinbox, mode="flood_fill"
        ),
    )
    button.pack(side="left")

    button = Button(
        frame,
        text="Save Mask",
        command=lambda: save_mask(canvas, window),
    )
    button.pack(side="left")

    # Add the image to the canvas
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    # Start the Tkinter event loop
    window.mainloop()

    return canvas


def execute_mgac(canvas, balloon_spinbox, threshold_spinbox, mode="mgac"):
    # Get the values from the spinboxes
    balloon = float(balloon_spinbox.get())
    threshold = float(threshold_spinbox.get())
    # Call the mgas function
    if canvas.center is None or canvas.ellipse_axes is None:
        print("Please select an ellipse.")
        return
    centre_x, centre_y = canvas.center
    centre_x = int(centre_x * 1500 / canvas.winfo_width())
    centre_y = int(centre_y * 1000 / canvas.winfo_height())
    a, b = canvas.ellipse_axes
    a = int(a * 1500 / canvas.winfo_width())
    b = int(b * 1000 / canvas.winfo_height())
    s = np.linspace(0, 2 * np.pi, 3200)  # 3200 is the number of points
    r = centre_y + b * np.sin(s)
    c = centre_x + a * np.cos(s)
    init = np.array([r, c]).T

    dilation_blur, result = mgac.edge_detection(canvas.image_path, centre_x, centre_y)
    if mode == "mgac":
        result, evolution = mgac.mgas(
            img=dilation_blur, initialisation=init, balloon=balloon, threshold=threshold
        )
        result = (result * 255).astype(np.uint8)

    canvas.mask = result
    # Convert the result to an image and display it
    result_img = Image.fromarray(result)
    result_img.show()


def save_mask(canvas, window):
    if canvas.mask is None:
        print("Please generate a mask first.")
        return
    mask_img = Image.fromarray(canvas.mask)
    output_name = os.path.splitext(canvas.image_path)[0] + "_mask.png"
    mask_img.save(output_name)
    print(f"Mask saved as {output_name}.")
    window.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MGAC Mask Generator")
    parser.add_argument("image", help="path to image")
    args = parser.parse_args()
    # Call the function with the path to your image file
    canvas = display_image(args.image)
