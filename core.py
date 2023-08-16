#import tkinter as tk
# core.py
import data
import segmentation
import gui_module

class Core:
    def __init__(self):
        self.gui = gui_module.GUIApp(self.handle_gui_click)
        #attempted to pass another arg: , self.perform_segmentation, but got error of too many args 
        self.directory1 = ""
        self.directory2 = ""

    def handle_gui_click(self, input_value):
        # Implement your logic here to handle the data received from the GUI
        print(f"Received input value from GUI: {input_value}")
        self.directory1 = input_value
        self.directory2 = input_value

    def run(self):
        self.gui.start()

    def get_3d_image(directory):
        image = data.get_3d_image(directory)
        return image
    
    def view_sitk_3d_image(image, numslices, displayText):
        data.view_sitk_3d_image(image, numslices)

    def view_slice_metadata_from_directory(directory):
        data.view_slice_metadata_from_directory(directory)

    def resize_and_convert_to_3d_image(directory):
        new_images = data.resize_and_convert_to_3d_image(directory)
        return new_images

    def save_sitk_3d_img_png(directory, new_dir):
        data.save_sitk_3d_img_png(directory, new_dir)

    def basic_segment(atlas, image):
        registered_image = segmentation.basic_segment(atlas, image)
        return registered_image

    def perform_segmentation(self):
        if self.directory1 and self.directory2:
            self.image1 = data.get_3d_image(self.directory1)
            self.image2 = data.get_3d_image(self.directory2)
            if self.image1 and self.image2:
                registered_image = segmentation.basic_segment(self.image1, self.image2)
                data.view_sitk_3d_image(registered_image, 5, "Segmented Image")

    """ def perform_segmentation(self):
        if self.directory1 and self.directory2:
            atlas_image = data.get_3d_image(self.directory1)  # Load atlas image
            input_image = data.get_3d_image(self.directory2)  # Load input image
            if atlas_image and input_image:
                registered_image = self.basic_segment(atlas_image, input_image)  # Call basic_segment
                data.view_sitk_3d_image(registered_image, 5, "Segmented Image")"""

    def select_folder(self): 
        self.gui.select_folder()
                                    

if __name__ == "__main__":
    core = Core()
    core.run()
    # Path to the directory that contains the DICOM files
    directory1 = "scan1"
    directory2 = "scan2"

    #core.select_folder()
    

    # Create 3d image with SITK
    #image1 = core.get_3d_image(directory1)
    #image2 = core.get_3d_image(directory2)

    #view slices of 3d image
    #core.view_sitk_3d_image(image1, 10)
    #core.view_sitk_3d_image(image2, 10)

    #view metadata of slices in directory
    #core.view_slice_metadata_from_directory(directory1)
    #core.view_slice_metadata_from_directory(directory2)

    


    


#for using tkinter, this was inside the core class
"""
    def call_gui_method(self):
        # Dummy method to simulate the core calling a method in the GUI
        # In the actual implementation, you'll perform more meaningful actions
        if self.gui_instance is None:
            self.gui_instance = gui.GUI(root, self.segment_flag, self)  # Pass the Core instance to the GUI
        print("Core is calling a GUI method.")
        self.segment_flag = self.gui_instance.get_segment_flag()

    def run_gui(self):
        if self.gui_instance is not None:
            self.gui_instance.run()
"""
#for tkinter, this was in the if __name__ == "__main__":
"""root = tk.Tk()  # Create the root Tkinter window
    root.title("Core with GUI")

    # Run the core.call_gui_method after initializing the GUI
    core.call_gui_method()

    # Run the GUI initialization in the main thread using 'after()'
    root.after(0, core.run_gui)

    root.mainloop()"""