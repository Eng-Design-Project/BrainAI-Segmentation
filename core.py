# to create a new virtual environment, type : python -m venv env_name
# to activate (do this in powershell terminal, it doesn't work in git bash)
# my_venv/Scripts/activate, or type my_venv (press tab) Scripts (press tab) activate
# pip install tk

import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk
from tkinter import Canvas, Scrollbar, Frame
from PIL import Image, ImageTk  # Import PIL for image manipulation
from tkinter import Toplevel, Radiobutton, Button, StringVar

import data
import segmentation

class Core:
    def __init__(self, master):
        self.master = master
        self.current_page = None  # Track the current page being displayed
        self.segmentation_results = {}  # Initialize the segmentation_results variable as an empty dictionary
        self.popup_window = None  # Add this line to define popup_window
        self.results_label = tk.Label(self.master, text="")
        self.results_label.pack_forget()  # Hide the label by default
        self.atlas_segmentation_completed = False  # Initialize the attribute as False

        self.master.title("QuickSeg")
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12))
        self.advanced_algo =tk.StringVar() #this will be set to either 'Deep Learning' or 'Clustering' depending on the button click
        
        # Select folder button
        self.select_folder_button = tk.Button(self.master, text="Select Folder", command=self.select_folder)
        self.select_folder_button.pack(pady=20)
        self.selected_folder = ""

        # Label for selected folder button
        self.folder_label = tk.Label(self.master, text="Selected Folder: ")
        self.folder_label.pack()

        # Atlas segment button
        self.atlas_segment_button = tk.Button(self.master, text="Atlas Segment", command=self.atlas_segment)
        self.atlas_segment_button.pack(pady=20)

        # Initialize the Internal Atlas Segmentation button (hidden initially)
        self.internal_atlas_segment_button = tk.Button(self.master, text="Internal Atlas Segmentation")
        self.internal_atlas_segment_button.pack_forget()
        
        # Image scoring button
        self.image_scoring_button = tk.Button(self.master, text="Score Images")
        self.image_scoring_button.pack(pady=20)

        # Advanced segmentation button
        self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation", command=lambda: self.change_buttons([self.select_folder_button, self.folder_label, self.image_scoring_button, self.deep_learning_button, self.clustering_button, self.save_message_label, self.advanced_back_button], self.master))
        self.advanced_segmentation_button.pack(pady=20)

        # Clustering button
        self.clustering_button = tk.Button(self.master, text="Clustering")

        # Deep learning button
        self.deep_learning_button = tk.Button(self.master, text="Deep Learning")

        # Advanced segmentation back button
        self.advanced_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.select_folder_button, self.folder_label, self.atlas_segment_button, self.image_scoring_button, self.advanced_segmentation_button, self.show_image_results_button, self.view_DCMS_btn, self.save_message_label], self.master))

        """self.image_file_path = 'mytest.png'
        self.image_button = tk.Button(self.master, text="Display Image", command=self.display_file_png)
        self.image_button.pack(pady=20)"""

        # Button for showing segmentation results for an image
        self.show_image_results_button = tk.Button(self.master, text="Show Image Results", command=self.show_image_results)
        self.show_image_results_button.pack(pady=20)

        # Button for showing segmentation results for a folder
        self.view_DCMS_btn = tk.Button(self.master, text="View DCM Images from Folder", command=self.view_DCMs_from_file)
        self.view_DCMS_btn.pack(pady=20)

        #self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation", command=lambda: self.change_buttons([], [self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button]))
        #self.advanced_segmentation_button.pack(pady=20)

        self.image_paths = []  # List to store image file paths
        self.current_image_index = 0  # Index of the currently displayed image

        # Create "Previous" and "Next" buttons for image navigation
        self.previous_button = tk.Button(self.master, text="Previous")
        self.next_button = tk.Button(self.master, text="Next")

        self.image_label = tk.Label(self.master)

        self.save_message_label = tk.Label(self.master, text="")
        self.save_message_label.pack_forget()
    
    #656
    def atlas_segment(self):
        print("Atlas Segmentation called")
        #gets selected folder global from core class
        #if empty, have to get it
        if(self.selected_folder == ""):
            print("no folder selected") 
            #prompt the user to select a folder
            self.select_folder()

        # Check if the selected folder is a valid segment results directory
        if not data.contains_only_dcms(self.selected_folder):
            error_message = "Incorrect input folder, contains non-DCMs"
            self.show_popup_message(error_message, close_callback=self.select_folder)
            # Clear the selected folder
            self.selected_folder = ""
            # Prompt the user to select a folder again
            self.select_folder()
            return  # Exit the function or handle the invalid folder as needed   
               
        # use functions in data to read the atlas and 
        #   the image in question into memory here as 3d np array images
        image = data.get_3d_image(self.selected_folder)
        atlas_path = data.get_atlas_path()
        atlas = data.get_3d_image(atlas_path)
        # get atlas colors as 3d np array
        color_atlas_path = data.get_color_atlas_path()
        color_atlas = data.get_2d_png_array_list(color_atlas_path)
        # call execute atlas seg, passing image, atlas and atlas colors as args
        seg_results = segmentation.execute_atlas_seg(atlas, color_atlas, image)
        # returns dict of simple itk images
        # save them as dcms to the nested folder
        # Check if the selected folder is a valid segment results directory
        
        #save seg results to file, and to data.segmentation_results
        self.save_seg_results(seg_results)
        
        #display seg results
        self.show_image_results(seg_results)

        #here to test execute internal_atlas_seg
        #self.execute_internal_atlas_seg()

    #707
    def save_seg_results(self, seg_results):
        # Ask the user to select a folder for saving the results
        save_folder = filedialog.askdirectory(title="Select Save Folder")

        if save_folder:
            # Prompt the user to enter a file name within the GUI
            file_name = simpledialog.askstring("Input", "Enter file name:")
            
            if file_name:
                # Save the segmentation results with the user-specified file name
                data.store_seg_img_on_file(seg_results, self.selected_folder, f"{save_folder}/{file_name}.DCMs")
                data.store_seg_png_on_file(seg_results, f"{save_folder}/{file_name}.PNGs")
            # save dict of 3d np array images to data global seg results
            # Show a message to inform the user that the folder was selected for saving
            self.save_message = "Selected folder for saving: " + save_folder
            self.save_message_label = tk.Label(self.master, text=self.save_message)
            self.save_message_label.pack()
            data.segmentation_results = seg_results       
        # Set a flag to indicate that atlas segmentation has been performed
        self.atlas_segmentation_completed = True
        print("Atlas segmentation completed")  # Add this line for debugging
        self.change_buttons([self.select_folder_button, self.folder_label, self.atlas_segment_button, self.image_scoring_button, self.advanced_segmentation_button, self.show_image_results_button, self.view_DCMS_btn, self.save_message_label], self.master)


    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder")
        if folder_path:
            print("Selected folder:", folder_path)
            self.selected_folder = folder_path
            self.update_folder_label()

    def update_folder_label(self):
        # Update the label to display the selected folder path
        self.folder_label.config(text="Selected Folder: " + self.selected_folder)

    def show_image_results(self, image_dict=None):
        # This function will display segmentation results for an image

        if image_dict is None:
            folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")
            while not data.is_segment_results_dir(folder):
                tk.messagebox.showwarning(
                    title="Invalid Selection",
                    message="The folder you selected does not match the expected structure. Select a folder with sub-folders containing DCM files.")
                folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")
            image_dict = data.subfolders_to_dictionary(folder)
        
        pngs_dict = data.array_dict_to_png_dict(image_dict)

        popup_window = tk.Toplevel(self.master)
        popup_window.title("Results of Segmentation")

        # New dictionary to keep track of indexes for each segmentation type
        segmentation_indexes = {region: 0 for region in pngs_dict.keys()}
        current_segmentation = next(iter(pngs_dict))  # Initialize with the first key

        def update_image():
            index = segmentation_indexes[current_segmentation]
            image_list = pngs_dict[current_segmentation]
            image = image_list[index]
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo

        def handle_segment_selection(segmentation_type):
            nonlocal current_segmentation
            current_segmentation = segmentation_type
            update_image()

        def handle_previous():
            segmentation_indexes[current_segmentation] = (segmentation_indexes[current_segmentation] - 1) % len(pngs_dict[current_segmentation])
            update_image()

        def handle_next():
            segmentation_indexes[current_segmentation] = (segmentation_indexes[current_segmentation] + 1) % len(pngs_dict[current_segmentation])
            update_image()

        image_label = tk.Label(popup_window)
        image_label.pack()
        button_frame = tk.Frame(popup_window)
        button_frame.pack()

        # Create Previous/Next buttons
        previous_button = tk.Button(button_frame, text="Previous", command=handle_previous)
        next_button = tk.Button(button_frame, text="Next", command=handle_next)
        previous_button.pack(side="left", padx=10)
        next_button.pack(side="right", padx=10)

        # Create buttons for each region in the image dictionary
        for region in pngs_dict.keys():
            btn = tk.Button(popup_window, text=region, command=lambda r=region: handle_segment_selection(r))
            btn.pack(pady=2)  # Adjust padding as needed

        update_image()
        popup_window.geometry("300x300")  # Adjust width and height as needed

    def view_DCMs_from_file(self):
        # This function will eventually display DCMs from file
        # note, currently only works for un-segmented DCMs
        folder = filedialog.askdirectory(title="Select a folder containing only DCM files")
        if (data.contains_only_dcms(folder)):
            # convert each file to a PNG and save it to list
            png_list = []
            np_3d = data.get_3d_image(folder)
            png_list = data.convert_3d_numpy_to_png_list(np_3d)

            #create the popup
            popup_wind = tk.Toplevel(self.master)
            popup_wind.title("DCM images in Folder")
            curr_index = 0

            def handle_nex():
                nonlocal curr_index
                curr_index +=1
                if curr_index >= len(png_list):
                    curr_index = 0  # Cycle back to the first image
                update() 
                
            def handle_prev():
                nonlocal curr_index
                curr_index -=1
                if curr_index < 0:
                    curr_index = len(png_list) - 1  # Cycle back to the last image                
                update() 
                
            def update():
                nonlocal png_list, curr_index
                image = png_list[curr_index]
                photo = ImageTk.PhotoImage(image)
                image_label.configure(image=photo)
                image_label.image = photo
                # Update the label to show the current index
                index_label.config(text=f"Image {curr_index + 1} out of {len(png_list)}")

            image_label = tk.Label(popup_wind)
            image_label.pack()
            
            # Add a label to display the current index
            index_label = tk.Label(popup_wind, text="")
            index_label.pack()

            button_frame = tk.Frame(popup_wind)
            button_frame.pack()
            prev_btn = tk.Button(button_frame, text="Previous", command=handle_prev)
            nex_btn = tk.Button(button_frame, text="Next", command=handle_nex)


            prev_btn.pack(side="left", padx=10)
            nex_btn.pack(side="right", padx=10)
            update()

            popup_wind.geometry("400x300")  # Adjust width and height as needed

        #else if the folder does not have DCMs...  
        else:
            tk.messagebox.showwarning(title="Invalid Selection", message="Select a folder containg only DCM files.") 

    def change_buttons(self, show_list, parent):
        self.forget_all_packed_widgets(parent)
        for button in show_list:
            button.pack(pady=20)
            if button == self.atlas_segment_button and self.atlas_segmentation_completed == True:
                self.internal_atlas_segment_button.pack(pady=20)    
        # Check if the "Advanced Segmentation Back" button is clicked and the "Atlas Segmentation" is not complete
        #if self.advanced_segmentation_button in show_list and not self.atlas_segmentation_completed:
         #   self.internal_atlas_segment_button.pack_forget()  
               

    def forget_all_packed_widgets(self, parent):
        for widget in parent.winfo_children():
            if widget.winfo_manager() == 'pack':
                widget.pack_forget()    

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x700")
    app = Core(root)
    root.mainloop()