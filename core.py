import data
#import tkinter as tk
# core.py
import gui_module

class Core:
    def __init__(self):
        self.gui = gui_module.GUIApp(self.handle_gui_click)

    def handle_gui_click(self, input_value):
        # Implement your logic here to handle the data received from the GUI
        print(f"Received input value from GUI: {input_value}")
        if self.gui.segment_flag == True:
            print("seg flag set true")
            #this currently does not work, 
            #the segment flag is retrieved here before it is set in the GUI module 

    def run(self):
        self.gui.start()

    def get_3d_image(directory):
        image = data.get_3d_image(directory)
        return image
    
    def view_sitk_3d_image(image, numslices):
        data.view_sitk_3d_image(image, numslices)

    def view_slice_metadata_from_directory(directory):
        data.view_slice_metadata_from_directory(directory)

if __name__ == "__main__":
    core = Core()
    core.run()
    # Path to the directory that contains the DICOM files
    directory1 = "scan1"
    directory2 = "scan2"

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