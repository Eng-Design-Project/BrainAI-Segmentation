#import data
#import tkinter as tk
import gui_module

class Core:
    def __init__(self):
        self.gui = gui_module.GUIApp("Hello, World!")

    def run(self):
        self.gui.start()

if __name__ == "__main__":
    core = Core()
    core.run()

    


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