# gui_module.py
import dearpygui.dearpygui as dpg

dpg.create_context()

class GUIApp:
    def __init__(self, callback):
        self.segment_flag = False
        self.callback = callback
        self.input_value = ""

    def handle_click(self, sender, app_data):
        # Get the actual input value from the GUI input text field
        self.input_value = dpg.get_value(self.input_value)
        self.callback(self.input_value)
        self.handle_segment_flag_toggle()

    def handle_segment_flag_toggle(self):
        self.segment_flag = not self.segment_flag
        print(f"Segment flag is now: {self.segment_flag}")

    def create_submit_button(self):
        with dpg.window(label="Example Window"):
            self.input_value = dpg.add_input_text(label="Input Value", width=200)
            dpg.add_button(label="Submit", width=500, height=200, callback=self.handle_click)

    def callback(sender, app_data):
        print('OK was clicked.')
        print("Sender: ", sender)
        print("App Data: ", app_data)

    def cancel_callback(sender, app_data):
        print('Cancel was clicked.')
        print("Sender: ", sender)
        print("App Data: ", app_data)

    def select_folder(callback, cancel_callback):
        dpg.add_file_dialog(
            directory_selector=True, show=False, callback=callback, tag="file_dialog_id",
            cancel_callback=cancel_callback, width=700 ,height=400)

        with dpg.window(label="Tutorial", width=800, height=300):
            dpg.add_button(label="Directory Selector", callback=lambda: dpg.show_item("file_dialog_id"))

    def start(self):
        self.create_submit_button()

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

