import tkinter as tk

class GUI:
    def __init__(self, segment_flag):
        self.running = False
        self.segment_flag = segment_flag

        self.window = tk.Tk()
        self.window.title("Automated Segmentation of Brain Images with AI")

        self.segment_button = tk.Button(self.window, text="Segment", justify="center", command=lambda: self.segment_toggle())
        self.segment_button.pack()

    def segment_toggle(self):
        self.segment_flag = not self.segment_flag
        print("Segment flag:", self.segment_flag)

    def get_segment_flag(self):
        return self.segment_flag

    def set_segment_flag(self, value):
        self.segment_flag = value
    

    def run(self):
        self.running = True
        self.window.mainloop()

