import segmentation
import data
import tkinter as tk
from tkinter import filedialog
class AdvancedSegmentationPage:
    def __init__(self, master, core_instance):
        self.master = master
        self.core_instance = core_instance
        self.master.title("Advanced Segmentation")

        # Button for advanced segmentation
        self.advanced_segment_button = tk.Button(self.master, text="Advanced Segmentation", command=self.advanced_segment)
        self.advanced_segment_button.pack(pady=20)

        # Back button
        self.back_button = tk.Button(self.master, text="Back", command=self.go_back)
        self.back_button.pack(pady=20)

    def advanced_segment(self):
        print("Advanced Segmentation called")
        # Add your advanced segmentation logic here

    def go_back(self):
        self.master.destroy()  # Close the advanced segmentation window
        self.core_instance.show_main_window()  # Show the main window of the Core instance
    
class Core:
    def __init__(self, master):
        self.master = master

        # Button for selecting a folder
        self.select_folder_button = tk.Button(self.master, text="Select Folder", command=self.select_folder)
        self.select_folder_button.pack(pady=20)
        self.selected_folder = ""

        # Label for displaying the selected folder
        self.folder_label = tk.Label(self.master, text="Selected Folder: ")
        self.folder_label.pack()

        # Button for atlas segmenting a scan
        self.atlas_segment_button = tk.Button(self.master, text="Atlas Segment", command=self.atlas_segment)
        self.atlas_segment_button.pack(pady=20)

        # Additional button for demonstration
        #self.another_button = tk.Button(self.master, text="Another Action", command=self.another_action)
        #self.another_button.pack(pady=20)

<<<<<<< Updated upstream
=======
        self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation Page", command=self.open_advanced_segmentation)
        self.advanced_segmentation_button.pack(pady=20)

>>>>>>> Stashed changes
        # Button for displaying a PNG (in progress/subject to change)
        self.image_file_path='mytest.png' # this example file path can be changed by other functions
        self.image_button = tk.Button(self.master, text="Display Image", command= self.display_file_png)
        self.image_button.pack(pady=20)
<<<<<<< Updated upstream
    
=======

        # Button for showing segmentation results for an image
        self.show_image_results_button = tk.Button(self.master, text="Show Image Results", command=self.show_image_results)
        self.show_image_results_button.pack(pady=20)

        # Button for showing segmentation results for a folder
        self.show_folder_results_button = tk.Button(self.master, text="Show Folder Results", command=self.show_folder_results)
        self.show_folder_results_button.pack(pady=20)


>>>>>>> Stashed changes
    def display_file_png(self):
        # print("display image clicked")
        file_path = self.image_file_path
        self.image1=tk.PhotoImage(file=file_path)
        self.label = tk.Label(self.master, image = self.image1)
<<<<<<< Updated upstream
        self.label.place(x=20, y=20) # arbitrary position coordinates, they can be made into arguments for the function

=======
        self.label.place(x=20, y=20) # arbitrary position coordinates, they can be made into arguments for the function    

    def show_image_results(self):
        # This function will eventually display segmentation results for an image
        # You can add your image processing and display logic here
        print("Displaying segmentation results for an image")

    def show_folder_results(self):
        # This function will eventually display segmentation results for images in a folder
        # You can add your image processing and display logic here
        print("Displaying segmentation results for images in a folder")

    def open_advanced_segmentation(self):
        advanced_segmentation_window = tk.Toplevel(self.master)
        advanced_segmentation_page = AdvancedSegmentationPage(advanced_segmentation_window, self)

        # You can customize the appearance of the Toplevel window here

        advanced_segmentation_window.transient(self.master)
        advanced_segmentation_window.grab_set()
        advanced_segmentation_window.focus_set()
        advanced_segmentation_window.wait_window()

    def show_main_window(self):
        self.master.deiconify()  # Show the main window       
    
>>>>>>> Stashed changes
    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            print("Selected folder:", folder_path)
            self.selected_folder = folder_path
            self.update_folder_label()  # Update the label with the new folder path

    def update_folder_label(self):
        self.folder_label.config(text="Selected Folder: " + self.selected_folder)

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
