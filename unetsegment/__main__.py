import os, shutil
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import zipfile
import csv
import glob
import matplotlib.pyplot as plt
import argparse
from unetsegment import get_mask


class UNetSegmentApp:
    def __init__(self, master, args):
        self.master = master
        master.title("UNet Image Segmentation")

        # Model selection
        self.model_path_var = tk.StringVar(value=os.getenv("LPBFSEGMENT_MODEL", ""))
        tk.Label(self.master, text="Model Path:", anchor="e").grid(row=0, column=0)
        self.model_path_entry = tk.Entry(
            self.master, textvariable=self.model_path_var, width=70
        )
        self.model_path_entry.grid(row=0, column=1, columnspan=2)
        self.browse_model_button = tk.Button(
            self.master, text="Browse Model", command=self.browse_model
        )
        self.browse_model_button.grid(row=0, column=3)

        # Mode selection
        self.mode_var = tk.StringVar(value="Single Image")
        tk.Label(self.master, text="Mode:", anchor="e").grid(row=1, column=0)
        tk.OptionMenu(self.master, self.mode_var, "Single Image", "Batch Mode").grid(
            row=1, column=1, columnspan=2
        )

        # Image path selection
        self.image_path_var = tk.StringVar()
        if args.image:
            self.image_path_var.set(args.image)
            if self.model_path_var.get():
                self.load_unet_model()
                self.segment_button.config(state=tk.NORMAL)
        tk.Label(self.master, text="Image/Directory Path:", anchor="e").grid(
            row=2, column=0
        )
        self.image_path_entry = tk.Entry(
            self.master, textvariable=self.image_path_var, width=70
        )
        self.image_path_entry.grid(row=2, column=1, columnspan=2)
        self.browse_image_button = tk.Button(
            master, text="Browse Image", command=self.browse_image
        )
        self.browse_image_button.grid(row=2, column=3)

        # Status label
        self.status_label = tk.Label(self.master, text="Status: Waiting for input")
        self.status_label.grid(row=3, column=0, columnspan=2, padx=5)

        # Segment button
        self.segment_button = tk.Button(
            master, text="Segment", state=tk.DISABLED, command=self.segment
        )
        self.segment_button.grid(row=3, column=2, columnspan=1)

        # Save output button
        self.save_button = tk.Button(
            master, text="Save Output", state=tk.DISABLED, command=self.save_output
        )
        self.save_button.grid(row=3, column=3, columnspan=1)

        self.image_panel = tk.Canvas(self.master, width=400, height=500)
        self.image_panel.grid(row=5, column=0, columnspan=2)

        self.mask_panel = tk.Canvas(self.master, width=400, height=500)
        self.mask_panel.grid(row=5, column=2, columnspan=2)

        self.model = None
        self.image_path_list = None
        self.output_images = []
        self.metrics = []

    def load_unet_model(self):
        try:
            self.model = load_model(self.model_path_var.get())
            self.status_label.config(text="Status: Model loaded")
        except Exception as e:
            messagebox.showerror("Error loading model", str(e))

    def browse_model(self):
        model_path = filedialog.askopenfilename(
            initialdir=os.getcwd(), filetypes=[("HDF5 files", "*.h5 *.hdf5")]
        )
        if model_path:
            self.model_path_var.set(model_path)
            self.load_unet_model()

    def browse_image(self):
        if self.mode_var.get() == "Single Image":
            image_path = filedialog.askopenfilename(
                initialdir=os.getcwd(),
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")],
            )
        else:
            image_path = filedialog.askdirectory(initialdir=os.getcwd())
        if image_path:
            self.image_path_list = self.get_image_path_list(image_path)
            if len(self.image_path_list) > 0:
                self.image_path_var.set(image_path)
                self.load_unet_model()
                self.status_label.config(
                    text=f"Status: {len(self.image_path_list)} images found"
                )
                self.segment_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.DISABLED)
                try:
                    img = Image.open(self.image_path_list[0])
                    img.thumbnail((400, 500))
                    img = ImageTk.PhotoImage(img)
                    self.image_panel.create_image(0, 0, anchor="nw", image=img)
                    self.image_panel.image = img
                except Exception as e:
                    messagebox.showerror("Error displaying image", str(e))
            else:
                self.status_label.config(text="Status: No images found")

    def get_image_path_list(self, image_path):
        if self.mode_var.get() == "Single Image":
            return [image_path]
        else:
            return (
                glob.glob(image_path + "/**/*.png", recursive=True)
                + glob.glob(image_path + "/**/*.jpg", recursive=True)
                + glob.glob(image_path + "/**/*.jpeg", recursive=True)
                + glob.glob(image_path + "/**/*.tif", recursive=True)
                + glob.glob(image_path + "/**/*.tiff", recursive=True)
            )

    def segment(self):
        if not self.image_path_list or not self.model:
            messagebox.showerror("Error", "No images or model loaded.")
            return
        self.save_button.config(state=tk.DISABLED)
        self.output_images = []
        self.metrics = []
        for index, image_path in enumerate(self.image_path_list):
            self.status_label.config(
                text=f"Status: Segmenting image {index+1}/{len(self.image_path_list)}"
            )
            self.master.update()
            mask, metrics = get_mask(image_path, self.model)
            self.output_images.append((image_path, mask))
            self.metrics.append((image_path, metrics))
        self.save_button.config(state=tk.NORMAL)
        self.status_label.config(
            text=f"Status: Segmented {len(self.image_path_list)} images"
        )
        try:
            msk = self.output_images[0][1]
            msk.thumbnail((400, 500))
            msk = ImageTk.PhotoImage(msk)
            self.mask_panel.create_image(0, 0, anchor="nw", image=msk)
            self.mask_panel.image = msk
        except Exception as e:
            messagebox.showerror("Error displaying mask", str(e))
        # For demonstration, not displaying images in the GUI

    def save_output(self):
        if not self.image_path_list or not self.model:
            messagebox.showerror("Error", "No images or model loaded.")
            return

        output_dir = filedialog.askdirectory(initialdir=os.getcwd())
        if not output_dir:
            return
        if not os.path.exists(os.path.join(output_dir, "tmpdir")):
            os.mkdir(os.path.join(output_dir, "tmpdir"))
        temp_dir = os.path.join(output_dir, "tmpdir")
        with zipfile.ZipFile(os.path.join(output_dir, "output.zip"), "w") as zipf:
            with open(
                os.path.join(temp_dir, "metrics.csv"), "w", newline=""
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Image Name",
                        "width",
                        "height_above",
                        "height_below",
                        "alpha",
                        "beta",
                        "scale (um/pixel)",
                    ]
                )
                for image_path, metrics in self.metrics:
                    image_name = os.path.basename(image_path)
                    writer.writerow([image_name] + list(metrics))
            zipf.write(os.path.join(temp_dir, "metrics.csv"), "metrics.csv")
            for image_path, mask in self.output_images:
                image_name = os.path.basename(image_path)
                mask_name = image_name.rsplit(".", 1)[0] + "_mask.png"
                mask_path = os.path.join(temp_dir, mask_name)
                mask.save(mask_path)
                zipf.write(mask_path, arcname=mask_name)
        shutil.rmtree(temp_dir)  # Clean up
        self.status_label.config(
            text=f"Status: Output saved to .../{os.path.join(os.path.basename(output_dir), 'output.zip')[-50:]}"
        )
        messagebox.showinfo("Success", "Output saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LPBF CS Image segmentation using UNet")
    parser.add_argument("--image", help="path to image", required=False)
    args = parser.parse_args()
    root = tk.Tk()
    app = UNetSegmentApp(root, args)
    root.mainloop()
