import dearpygui.dearpygui as dpg

class GUIApp:
    def __init__(self, input_value):
        self.input_value = input_value

    def handle_click(self, sender, app_data):
        print(f"Button clicked, input value is: {self.input_value}")

    def start(self):
        with dpg.handler_registry():
            dpg.add_handler(dpg.mvEvent_MouseRelease, self.handle_click)

        with dpg.window(label="Example Window"):
            dpg.add_button(label="Submit")

        dpg.create_context()
        dpg.create_viewport(title='Dear PyGui', width=600, height=300)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()















"""import tkinter as tk

class GUI:
    def __init__(self, root, segment_flag, core_instance):
        self.running = False
        self.segment_flag = segment_flag
        self.core_instance = core_instance

        self.window = root
        self.window.title("Automated Segmentation of Brain Images with AI")

        self.create_segment_button()

    def create_segment_button(self):
        self.segment_button = tk.Button(self.window, text="Segment", justify="center", command=lambda: self.segment_toggle())
        self.segment_button.pack()

    def segment_toggle(self):
        self.segment_flag = not self.segment_flag
        print("Segment flag:", self.segment_flag)
        self.core_instance.segment_flag = self.segment_flag

    def get_segment_flag(self):
        return self.segment_flag

    def set_segment_flag(self, value):
        self.segment_flag = value

    def run(self):
        self.running = True
        self.window.mainloop()"""
