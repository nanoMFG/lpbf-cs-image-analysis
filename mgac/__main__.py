# from . import *
# import cv2 as cv
# import numpy as np
# import argparse
# import os


# if __name__ == "__main__":
# parser = argparse.ArgumentParser("MGAC mask generator")
# parser.add_argument("image", help="path to image")
# parser.add_argument("-ns", "--nosave", action="store_false", help="toggle saving the mask (default = True)")
# # parser.add_argument("-fh", "--fillholes", action="store_false", help="toggle filling holes in the mask (default = True)")
# args = parser.parse_args()

# img = cv.imread(args.image)
# if img is None or img.size == 0:
#     raise Exception(f"Unable to read image {args.image}. Please check the path.")

# window = SelectionWindow(img, "Magic Wand Selector")

# print("Left click to seed a selection.")
# print("Drag to continue floodfill.")
# print("Right click to start painting circles.")
# print("Drag to continue painting.")
# print(" * [SHIFT] subtracts from the selection.")
# print(" * [ALT] starts a new selection with a new seed.")
# print(" * [SHIFT] + [ALT] intersects the selections.")
# print("Adjust sliders to change tolerance of floodfill and radius of paiting.")
# print()

# window.show()

# output = window.mask
# if args.fillholes:
#     contour,hier = cv.findContours(output,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
#     for cnt in contour:
#         cv.drawContours(output,[cnt],0,255,-1)
# print(f"Save: {args.nosave}")
# if args.nosave:
#     output_name = os.path.splitext(args.image)[0]+'_mask.png'
#     cv.imwrite(output_name, output)
#     print(f"Successfully saved mask. Save location: {output_name}")

# cv.destroyAllWindows()

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import mgac


# Assuming the segmentation function is provided
def segmentation(img_path, centre_x, centre_y, axis_x, axis_y):
    s = np.linspace(0, 2 * np.pi, 3200)
    r = centre_y + axis_y * np.sin(s)
    c = centre_x + axis_x * np.cos(s)
    init = np.array([r, c]).T

    masks = []

    dilation_blur, mask = mgac.edge_detection(img_path, centre_x, centre_y)
    masks.append(mask)

    for balloon in [1.0, 1.2]:
        for thresh in [0.85, 0.95, 1.05]:
            output, evolution = mgac.mgas(
                img=dilation_blur,
                initialisation=init,
                balloon=balloon,
                threshold=thresh,
            )
            masks.append(output * 255)

    return masks


def run_segmentation():
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    if not img_path:
        return

    centre_x, centre_y, axis_x, axis_y = (
        100,
        100,
        50,
        50,
    )  # Example values, adjust as necessary
    masks = segmentation(img_path, centre_x, centre_y, axis_x, axis_y)

    # Create a window to display images
    window = tk.Toplevel(root)
    window.title("Segmentation Results")

    # Load and display the original image
    original_image = Image.open(img_path)
    original_image.thumbnail((200, 200))
    original_photo = ImageTk.PhotoImage(original_image)

    label_original = tk.Label(window, text="Original Image")
    label_original.pack()
    canvas_original = tk.Canvas(
        window, width=original_photo.width(), height=original_photo.height()
    )
    canvas_original.pack()
    canvas_original.create_image(0, 0, anchor=tk.NW, image=original_photo)

    # Display the segmentation results
    photos = []
    canvases = []

    for i, mask in enumerate(masks):
        result_image = Image.fromarray(mask.astype(np.uint8))
        result_image.thumbnail((200, 200))
        result_photo = ImageTk.PhotoImage(result_image)
        photos.append(result_photo)

        label = tk.Label(window, text=f"Segmentation {i+1}")
        label.pack()

        canvas = tk.Canvas(
            window, width=result_photo.width(), height=result_photo.height()
        )
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=result_photo)
        canvases.append(canvas)

    def save_selected_image(selected_index):
        selected_mask = masks[selected_index]
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG files", "*.png")]
        )
        if save_path:
            Image.fromarray(selected_mask.astype(np.uint8)).save(save_path)
            messagebox.showinfo("Image Saved", f"Image saved to {save_path}")

    # Create radio buttons for selecting an image to save
    selected_index = tk.IntVar()
    selected_index.set(0)

    for i in range(len(masks)):
        tk.Radiobutton(
            window, text=f"Segmentation {i+1}", variable=selected_index, value=i
        ).pack()

    save_button = tk.Button(
        window,
        text="Save Selected Image",
        command=lambda: save_selected_image(selected_index.get()),
    )
    save_button.pack()

    # Keep a reference to the images to prevent garbage collection
    window.original_photo = original_photo
    window.photos = photos
    window.mainloop()


root = tk.Tk()
root.title("Image Segmentation")

open_button = tk.Button(root, text="Open Image", command=run_segmentation)
open_button.pack()

root.mainloop()
