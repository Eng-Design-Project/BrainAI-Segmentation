#import data
#import segmentation
import tkinter as tk
import gui  # Import your GUI module
import threading

class Core:
    def __init__(self):
        self.segment_flag = False
        self.gui_instance = None  # Store a reference to the GUI instance

    def call_gui_method(self):
        # Dummy method to simulate the core calling a method in the GUI
        # In the actual implementation, you'll perform more meaningful actions
        if self.gui_instance is None:
            self.gui_instance = gui.GUI(self.segment_flag)  # Create the GUI instance if not created already
            gui_thread = threading.Thread(target=self.gui_instance.run)
            gui_thread.start()
        print("Core is calling a GUI method.")
        #self.segment_flag = self.gui_instance.get_segment_flag()
        if self.segment_flag:
            print("oh hey")
            
            #self.gui_instance.set_segment_flag(False)  # Modify the flag in the GUI

if __name__ == "__main__":
    core = Core()
    #core.call_gui_method()
