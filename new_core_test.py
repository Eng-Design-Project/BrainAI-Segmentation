import segmentation
import data
import tkinter as tk
from tkinter import filedialog

class Core:
    def __init__(self, master):
        self.master = master

        # Button for selecting a folder
        self.select_folder_button = tk.Button(self.master, text="Select Folder", command=self.select_folder)
        self.select_folder_button.pack(pady=20)
        self.selected_folder = ""

        # Button for atlas segmenting a scan
        self.atlas_segment_button = tk.Button(self.master, text="Atlas Segment", command=self.atlas_segment)
        self.atlas_segment_button.pack(pady=20)

        # Additional button for demonstration
        self.another_button = tk.Button(self.master, text="Another Action", command=self.another_action)
        self.another_button.pack(pady=20)

        # Displaying a PNG (in progress)
        self.example_image1 = tk.PhotoImage(file='mytest.png')
        self.label = tk.Label(self.master, image=self.example_image1)
        self.label.place(x=0, y=0)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            print("Selected folder:", folder_path)
            self.selected_folder = folder_path
    
    def atlas_segment(self):
        print("Atlas Segmentation called")
        # Path to the directory that contains the DICOM files
        atlas_dir = data.get_atlas_path()
        if self.selected_folder == "":
            self.select_folder()
        if self.selected_folder != "":
            input_dir = self.selected_folder

            # Create 3d image with SITK
            atlas_image = data.get_3d_image(atlas_dir)
            input_image = data.get_3d_image(input_dir)

            registered_image = segmentation.atlas_segment(atlas_image, input_image)
            data.save_sitk_3d_img_to_dcm(registered_image, "registered")


    def another_action(self):
        print("Another action was performed!")

# Usage
root = tk.Tk()
root.geometry("600x400")
app = Core(root)
root.mainloop()
