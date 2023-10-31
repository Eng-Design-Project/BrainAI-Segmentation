import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk
from tkinter import Canvas, Scrollbar, Frame
from PIL import Image, ImageTk  # Import PIL for image manipulation
from tkinter import Toplevel, Radiobutton, Button, StringVar



#import deep_learning
#import clustering
import segmentation
import data
import os
import deep_learning_copy


"""class AdvancedSegmentationPage:
    def __init__(self, master, core_instance):
        self.master = master
        self.core_instance = core_instance
        self.master.title("Advanced Segmentation")

        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12))

        self.image_scoring_button = tk.Button(self.master, text="Score Images", command=self.core_instance.open_image_scoring_popup)
        self.image_scoring_button.pack(pady=20)
        self.clustering_button = tk.Button(self.master, text="Clustering", command=self.core_instance.show_clustering_buttons)
        self.clustering_button.pack(pady=20)
        self.deep_learning_button = tk.Button(self.master, text="Deep Learning", command=self.core_instance.show_deep_learning_buttons)
        self.deep_learning_button.pack(pady=20)
        self.back_button = tk.Button(self.master, text="Back", command=self.hide_buttons)
        self.back_button.pack(pady=20)
        self.hide_buttons()

    def hide_buttons(self):
        for button in [self.image_scoring_button, self.clustering_button, self.deep_learning_button, self.back_button]:
            button.pack_forget()

    def show_buttons(self):
        for button in [self.image_scoring_button, self.clustering_button, self.deep_learning_button, self.back_button]:
            button.pack(pady=20)"""


class ImageScoringPopup:
    # Constructor for the ImageScoringPopup class
    # Initialize the class with a parent/master window, image paths, and a callback function
    def __init__(self, master,image1, image2, callback):
        self.master = master
        self.callback = callback
        #image1 and image2 are pillow images
        self.image1 = image1
        self.image1 = image2
        self.current_image_index = 0

        # Create the main frame for the popup
        self.popup_frame = Frame(master)
        self.popup_frame.pack(fill='both', expand=True)

        # Create a canvas to display images and allow scrolling
        self.canvas = Canvas(self.popup_frame, width=800, height=600)
        self.canvas.pack(side="left", fill="both", expand=True)

        # Create a vertical scrollbar for the canvas
        self.scrollbar = Scrollbar(self.popup_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        # Configure the canvas to scroll with the scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create an inner frame within the canvas for placing widgets
        self.inner_frame = Frame(self.canvas)
        self.inner_frame_canvas = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        # Load and display images from the provided paths
        self.images = [image1, image2]
        self.photo_images = [ImageTk.PhotoImage(image) for image in self.images]

        # Display the current image in a label
        self.image_label = tk.Label(self.inner_frame, image=self.photo_images[self.current_image_index])
        self.image_label.pack(pady=(20, 10), anchor="center")

        # Create buttons for navigating to the previous and next images
        self.prev_button = tk.Button(self.inner_frame, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side="left", padx=10)
        
        self.next_button = tk.Button(self.inner_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(side="right", padx=10)

        # Create sliders for scoring the displayed images
        self.score_label1 = tk.Label(self.inner_frame, text="Score Image 1:")
        self.score_label1.pack(pady=10, anchor="center")

        self.score_entry1 = tk.Scale(self.inner_frame, from_=1, to=10, orient="horizontal", sliderrelief='flat')
        self.score_entry1.pack(pady=10, anchor="center")

        self.score_label2 = tk.Label(self.inner_frame, text="Score Image 2:")
        self.score_label2.pack(pady=10, anchor="center")

        self.score_entry2 = tk.Scale(self.inner_frame, from_=1, to=10, orient="horizontal", sliderrelief='flat')
        self.score_entry2.pack(pady=10, anchor="center")

        # Create a button to submit the scores
        self.submit_button = tk.Button(self.inner_frame, text="Submit", command=self.submit_scores)
        self.submit_button.pack(pady=20, anchor="center")

        # Bind events to handle canvas and inner frame configuration
        self.inner_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def on_frame_configure(self, event):
        # Configure the canvas scroll region based on the inner frame size
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        # Configure the canvas item based on its width
        self.canvas.itemconfig(self.inner_frame_canvas, width=event.width)

    def show_previous_image(self):
        # Display the previous image if available
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_label.config(image=self.photo_images[self.current_image_index])

    def show_next_image(self):
        # Display the next image if available
        if self.current_image_index < len(self.photo_images) - 1:
            self.current_image_index += 1
            self.image_label.config(image=self.photo_images[self.current_image_index])

    def submit_scores(self):
        try:
            # Get the scores entered by the user from the GUI
            score1 = float(self.score_entry1.get())
            score2 = float(self.score_entry2.get())
        
            # Ensure the scores are within the desired range (1-10)
            #min_score = min(score1, score2)
            #max_score = max(score1, score2)
            #min and max are hardcoded, not set by user (or risk divide by zero error)
            min_score = 1 # Minimum possible score
            max_score = 10 # Maximum possible score

            # Normalize the scores between 0 and 1
            normalized_score1 = (score1 - min_score) / (max_score - min_score)
            normalized_score2 = (score2 - min_score) / (max_score - min_score)

            # Save the normalized scores or pass them to your CNN
            self.callback(normalized_score1, normalized_score2)

            # Close the popup window
            self.master.destroy()
        except ValueError:
            # Handle invalid input (e.g., non-numeric input)
            print("Invalid input. Please enter numeric scores.")

#how UX for exec clustering or exec dl should go:
    #user selects options: algorithm, full scan or pre atlas seg
    #if (full scan selected):
    #   while (self.selected_folder != dcm folder):
    #       popup says "select a folder containing dcms"
    #       select folder
    #   exec_full_scan_clustering(self.selected_folder, algo)
    #   
    #else if (pre_atlas_seg selected):
    #   while (self.selected_folder != seg folder):
    #          popup says "select a segmented dcm folder"
    #          select folder
    #          if (self.selected_folder == dcm folder):
    #               run atlas_seg(self.selected_folder)
    #   exec_seg_clustering(self.selected_folder, algo)
    #

class Core:
    def __init__(self, master):
        self.master = master
        self.current_page = None  # Track the current page being displayed
        self.segmentation_results = {}  # Initialize the segmentation_results variable as an empty dictionary
        self.popup_window = None  # Add this line to define popup_window
        self.results_label = tk.Label(self.master, text="")
        self.results_label.pack_forget()  # Hide the label by default

        self.master.title("Image Analysis Tool")
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

        # Image scoring button
        self.image_scoring_button = tk.Button(self.master, text="Score Images", command=self.open_image_scoring_popup)
        self.image_scoring_button.pack(pady=20)

        # Advanced segmentation button
        self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation", command=lambda: self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button], [self.advanced_segmentation_button, self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button, self.execute_clustering_button]))
        self.advanced_segmentation_button.pack(pady=20)

        # Clustering button
        self.clustering_button = tk.Button(self.master, text="Clustering", command=lambda:(self.advanced_algo.set("Clustering"), self.change_buttons([self.execute_clustering_button, self.algo_btn, self.full_scan_btn, self.pre_atlas_seg_btn, self.clustering_back_button],[self.advanced_segmentation_button, self.deep_learning_button, self.clustering_button, self.advanced_back_button])))

        # Deep learning button
        self.deep_learning_button = tk.Button(self.master, text="Deep Learning", command=lambda:(self.advanced_algo.set("Deep Learning"), self.change_buttons([self.execute_deep_learning, self.algo_btn, self.full_scan_btn, self.pre_atlas_seg_btn, self.deeplearning_back_button],[self.deep_learning_button, self.clustering_button, self.execute_clustering_button, self.advanced_back_button])))

        # Execute deep learning button
        self.execute_deep_learning = tk.Button(self.master, text="Execute Deep Learning", command=lambda:self.change_buttons([],[])) #self.execute_deep_learning_click)

        # Advanced segmentation back button
        self.advanced_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.atlas_segment_button, self.image_scoring_button, self.advanced_segmentation_button, self.show_image_results_button, self.show_folder_results_button],[self.image_scoring_button, self.deep_learning_button, self.clustering_button, self.advanced_back_button]))

        # Clustering back button
        self.clustering_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.results_label, self.execute_clustering_button, self.image_label, self.previous_button, self.next_button, self.algo_btn, self.full_scan_btn, self.pre_atlas_seg_btn, self.clustering_back_button]))

        # Deep learning back button
        self.deeplearning_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.execute_deep_learning, self.image_label ,self.previous_button, self.next_button, self.algo_btn, self.full_scan_btn, self.pre_atlas_seg_btn, self.deeplearning_back_button]))
        

        """self.image_file_path = 'mytest.png'
        self.image_button = tk.Button(self.master, text="Display Image", command=self.display_file_png)
        self.image_button.pack(pady=20)"""

        # Button for showing segmentation results for an image
        self.show_image_results_button = tk.Button(self.master, text="Show Image Results", command=self.show_image_results)
        self.show_image_results_button.pack(pady=20)

        # Button for showing segmentation results for a folder
        self.show_folder_results_button = tk.Button(self.master, text="Show Folder Results", command=self.show_folder_results)
        self.show_folder_results_button.pack(pady=20)

        self.algo_btn = tk.Button(self.master, text = "Algorithm")
        self.full_scan_btn = tk.Button(self.master, text = "Full Scan", command = self.full_scan)
        self.pre_atlas_seg_btn = tk.Button(self.master, text = "Pre Atlas Segmentation", command = self.pre_atlas_seg)

        #self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation", command=lambda: self.change_buttons([], [self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button]))
        #self.advanced_segmentation_button.pack(pady=20)

        

        # Add a button to execute clustering
        self.execute_clustering_button = tk.Button(self.master, text="Execute Clustering", command=self.execute_clustering)
        #self.execute_clustering_button.pack(pady=20)

        self.image_paths = []  # List to store image file paths
        self.current_image_index = 0  # Index of the currently displayed image

        # Create "Previous" and "Next" buttons for image navigation
        self.previous_button = tk.Button(self.master, text="Previous", command=self.show_previous_image)
        self.next_button = tk.Button(self.master, text="Next", command=self.show_next_image)

        self.image_label = tk.Label(self.master)
    
    def execute_clustering(self):
        # Get the selected segmentation method
        selected_segmentation_method = self.get_selected_segmentation_method()

        if selected_segmentation_method == "atlas_segmentation":
            # If the selected method is "atlas_segmentation," call the atlas_segment function
            self.atlas_segment()
        else:
            if not data.segmentation_results:
                # If segmentation results are not available, call atlas_segment
                self.atlas_segment()
            # Open a clustering options popup
            self.open_clustering_options_popup()

    def open_clustering_options_popup(self):
        if self.get_selected_segmentation_method() == "atlas_segmentation" and not data.segmentation_results:
            # If the selected method is "atlas_segmentation" and segmentation results are not available, select a folder
            self.select_folder()
        else:
            # Create a popup window for clustering options
            popup_window = tk.Toplevel(self.master)
            popup_window.title("Select Clustering Parameters")

            # Create a label to instruct the user
            label = tk.Label(popup_window, text="Select clustering parameters:")
            label.pack(pady=10)

            segment_var = tk.StringVar()
            segment_var.set(None)
            whole_brain = tk.Radiobutton(popup_window, text="Whole Brain", variable=segment_var, value="Whole Brain")
            whole_brain.pack()
            segment = tk.Radiobutton(popup_window, text="Segment", variable=segment_var, value="Segment")
            segment.pack()

            # Create radio buttons for clustering algorithm options
            algorithm_var = tk.StringVar()
            algorithm_var.set(None)
            kmeans_option = tk.Radiobutton(popup_window, text="K-Means", variable=algorithm_var, value="K-Means")
            kmeans_option.pack()
            dbscan_option = tk.Radiobutton(popup_window, text="DBSCAN", variable=algorithm_var, value="DBSCAN")
            dbscan_option.pack()
            hierarchical_option = tk.Radiobutton(popup_window, text="Hierarchical", variable=algorithm_var, value="Hierarchical")
            hierarchical_option.pack()

            # Create radio buttons for data source options
            source_var = tk.StringVar()
            source_var.set(None)  # Set an initial value that does not correspond to any option
            file_option = tk.Radiobutton(popup_window, text="From File", variable=source_var, value="file")
            file_option.pack()
            memory_option = tk.Radiobutton(popup_window, text="From Memory", variable=source_var, value="memory")
            memory_option.pack()

            # Create a button to confirm the selection and execute clustering
            confirm_button = tk.Button(popup_window, text="Execute Clustering", command=lambda: self.handle_clustering_selection(popup_window, algorithm_var.get(), source_var.get()))
            confirm_button.pack(pady=20)
            """
            if((self.clustering_algorithm_combobox.get()!="") and (self.selected_folder!="")):
            data.set_seg_results(self.selected_folder) # this function sets data.segmentation_results
            self.segmentation_results = data.segmentation_results
            print("Selected file or folder:", self.selected_folder)
            # Logic to perform clustering from a file and set the clustering_results variable
            clustering_results = {}  # Implement file-based clustering logic here
            algorithm = self.clustering_algorithm_combobox.get()
            self.display_clustering_results(algorithm, clustering_results)
            self.results_label = tk.Label(self.master, text=f"Clustering Results for {algorithm}:")
            #self.results_label.pack()
            # Set image paths and current image index (replace with your own data)
            folder_path = "scan1_pngs"
            self.image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".png")]
            self.current_image_index = 0
            if self.image_paths:
                self.show_current_image()
                self.previous_button.pack(pady=10, anchor="center")
                self.next_button.pack(pady=10, anchor="center")
            else:
                self.previous_button.pack_forget()
                self.next_button.pack_forget()

        else:
            # Create a popup window for selecting clustering parameters
            popup_window = tk.Toplevel(self.master)
            popup_window.title("Select Clustering Parameters")

            # Create a label to instruct the user
            label = tk.Label(popup_window, text="Select clustering parameters:")
            label.pack(pady=10)

            # Create radio buttons for clustering algorithm options
            algorithm_var = tk.StringVar()
            algorithm_var.set(None)
            kmeans_option = tk.Radiobutton(popup_window, text="K-Means", variable=algorithm_var, value="K-Means")
            kmeans_option.pack()
            dbscan_option = tk.Radiobutton(popup_window, text="DBSCAN", variable=algorithm_var, value="DBSCAN")
            dbscan_option.pack()
            hierarchical_option = tk.Radiobutton(popup_window, text="Hierarchical", variable=algorithm_var, value="Hierarchical")
            hierarchical_option.pack()
            """

    def handle_clustering_selection(self, popup_window, algorithm, source):
        # Close the popup window
        popup_window.destroy()

        if source == "file":
            # Add code to get the selected file or folder here and store it
            selected_folder = self.get_selected_folder()
            data.set_seg_results(selected_folder)
            self.segmentation_results = data.segmentation_results
            print("Selected file or folder:", selected_folder)

            # Logic to perform clustering from a file and set the clustering_results variable
            clustering_results = {}  # Implement file-based clustering logic here
        elif source == "memory":
            # Logic to perform clustering from memory and set the clustering_results variable
            data.set_seg_results()
            self.segmentation_results = data.segmentation_results
            clustering_results = {}  # Implement memory-based clustering logic here

        # Display clustering results within the GUI
        self.display_clustering_results(algorithm, clustering_results)

        # You can use labels or other widgets to display the clustering results.

        # Set image paths and current image index (replace with your own data)
        folder_path = "scan1_pngs"
        self.image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".png")]
        self.current_image_index = 0

        # Show or hide "Previous" and "Next" buttons based on whether images are available
        if self.image_paths:
            self.show_current_image()
            self.previous_button.pack(pady=10, anchor="center")
            self.next_button.pack(pady=10, anchor="center")
        else:
            self.previous_button.pack_forget()
            self.next_button.pack_forget()

    def display_clustering_results(self, algorithm, clustering_results):
        # Create a label or canvas to display the clustering results
        self.results_label = tk.Label(self.master, text=f"Clustering Results for {algorithm}:")
        self.results_label.pack()
        # You can use labels or other widgets to display the clustering results here.

    def show_current_image(self):
        # Display the current image in the popup
        if self.image_paths:
            current_image_path = self.image_paths[self.current_image_index]
            image = Image.open(current_image_path)
            photo = ImageTk.PhotoImage(image)

            # Update the image label with the current image
            self.image_label.config(image=photo)
            self.image_label.photo = photo
            self.image_label.pack()

    def show_previous_image(self):
        # Display the previous image in the popup
        if self.image_paths:
            # Calculate the index of the previous image, considering the loop behavior
            self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.show_current_image()

    def show_next_image(self):
        # Display the next image in the popup
        if self.image_paths:
            # Calculate the index of the next image, considering the loop behavior
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.show_current_image()

    def execute_deep_learning_click(self):
        # Get the selected segmentation method
        selected_segmentation_method = self.get_selected_segmentation_method()

        if selected_segmentation_method == "atlas_segmentation":
            # If the selected method is "atlas_segmentation," call the atlas_segment function
            self.atlas_segment()
        else:
            if not data.segmentation_results:
                # If segmentation results are not available, call atlas_segment
                self.atlas_segment()
                # Open a segmentation selection popup
            self.open_segmentation_selection_popup()


    def open_segmentation_selection_popup(self):
        #wrap all of this in an if statement that checks if data.segmentation results is empty ( and run the logic)
        #then we call the deep learning function with the segmentation results passed as a parameter
    #if (data.segmentation_results=={}):
        # Create a popup window for selecting segmentation results
        if self.get_selected_segmentation_method() == "atlas_segmentation" and not data.segmentation_results:
            # If the selected method is "atlas_segmentation" and segmentation results are not available, select a folder
            self.select_folder()
        else:
            popup_window = tk.Toplevel(self.master)
            popup_window.title("Select Segmentation Results")

            label = tk.Label(popup_window, text="Select segmentation results source:")
            label.pack(pady=10)

            algorithm_var = tk.StringVar()
            algorithm_var.set(None)
            U_Net = tk.Radiobutton(popup_window, text="U-Net", variable=algorithm_var, value="U-Net")
            U_Net.pack()
            selection_var = tk.StringVar()
            selection_var.set(None)
            file_option = tk.Radiobutton(popup_window, text="From File", variable=selection_var, value="file")
            file_option.pack()
            memory_option = tk.Radiobutton(popup_window, text="From Memory", variable=selection_var, value="memory")
            memory_option.pack()

            confirm_button = tk.Button(popup_window, text="Confirm", command=lambda: self.handle_segmentation_selection(popup_window, selection_var.get(), algorithm_var.get()))
            confirm_button.pack(pady=20)

        #if seg results loaded from file, have to convert nested folder directory to dict of sitk images
        #alternative is running atlas seg

        # else:
        # Call the deep learning function with data.segmentation_results as a parameter
        #self.deep_learning_function(data.segmentation_results)


    def handle_segmentation_selection(self, popup_window, selection, algorithm):
        # Close the popup window
        popup_window.destroy()

        if selection == "file":
            # Get the selected folder and set segmentation results
            selected_folder = self.get_selected_folder()
            data.set_seg_results(selected_folder)
            print("Selected folder:", selected_folder)
        elif selection == "memory":
            # Set segmentation results from memory
            data.set_seg_results()
            self.segmentation_results = {}

        if algorithm == "U-Net":
            # Call the U_Net function
            self.U_Net()    

        folder_path = "scan1_pngs"
        self.image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".png")]
        self.current_image_index = 0

        if self.image_paths:
            self.show_current_image()
            self.previous_button.pack(pady=10, anchor="center")
            self.next_button.pack(pady=10, anchor="center")
        else:
            self.previous_button.pack_forget()
            self.next_button.pack_forget()

    def get_selected_segmentation_method(self):
        # Determine the selected segmentation method based on button states
        if self.atlas_segment_button.cget('state') == 'active':
            return "atlas_segmentation"
        elif self.clustering_button.cget('state') == 'active':
            return "clustering"
        elif self.deep_learning_button.cget('state') == 'active':
            return "deep_learning"
        else:
            return "default_method"        
        
    def show_main_window(self):
        self.master.deiconify()  # Show the main window

    def select_folder(self):
        # Display a message to inform the user
        
        # Open a dialog to select a folder path
        folder_path = filedialog.askdirectory(title="Select Folder",
            message="Please select a folder for your task."
        )
        # If a folder is selected, store it and update the folder label
        if folder_path:
            print("Selected folder:", folder_path)
            self.selected_folder = folder_path
            self.update_folder_label()

    def get_selected_folder(self):
        # Open a dialog to select a folder path and return it
        folder_path = filedialog.askdirectory()
        return folder_path

    def update_folder_label(self):
        # Update the label to display the selected folder path
        self.folder_label.config(text="Selected Folder: " + self.selected_folder)

    def display_file_png(self):
        # Display a PNG image file specified by the image_file_path attribute
        file_path = self.image_file_path
        self.image1 = tk.PhotoImage(file=file_path)
        self.label = tk.Label(self.master, image=self.image1)
        self.label.place(x=20, y=20)

    def U_Net(self):
            # Define a dictionary of SimpleITK images (adjust as needed)
        sitk_images_dict = {
            "image1": data.get_3d_image("scan1"),
            "image2": data.get_3d_image("scan2"),
        }

        # Call the dlAlgorithm function from deep_learning_copy module
        deep_learning_copy.dlAlgorithm(sitk_images_dict)

    def custom_askdirectory(title):
        # Create a custom dialog with a label
        dialog = tk.Toplevel()
        dialog.title(title)

        label = tk.Label(dialog, text=title)
        label.pack(padx=20, pady=10)

        selected_directory = filedialog.askdirectory(parent=dialog, title=title)
        dialog.destroy()  # Close the custom dialog

        return selected_directory        

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
        #   the image in question into memory here as sitk images
        image = data.get_3d_image(self.selected_folder)
        atlas_path = data.get_atlas_path()
        atlas = data.get_3d_image(atlas_path)
        # get atlas colors as 3d np array
        color_atlas = data.get_2d_png_array_list("color atlas")
        # call execute atlas seg, passing image, atlas and atlas colors as args
        seg_results = segmentation.execute_atlas_seg(atlas, color_atlas, image)
        # returns dict of simple itk images
        # save them as dcms to the nested folder
        # Check if the selected folder is a valid segment results directory
        

        #I want a function that converts the sitk image dicts to dicts with pngs
        png_dict = data.sitk_dict_to_png_dict(seg_results)

        # Ask the user to select a folder for saving the results
        save_folder = filedialog.askdirectory(
            title="Select Save Folder",
            message="Please select a folder to save the results."
        )

        if save_folder:
            # Prompt the user to enter a file name within the GUI
            file_name = simpledialog.askstring("Input", "Enter file name:")
            
            if file_name:
                # Save the segmentation results with the user-specified file name
                data.store_seg_img_on_file(seg_results, f"{save_folder}/{file_name}.DCMs")
                data.store_seg_png_on_file(seg_results, f"{save_folder}/{file_name}.PNGs")
            # save dict of sitk images to data global seg results
            # Show a message to inform the user that the folder was selected for saving
            save_message = "Selected folder for saving: " + save_folder
            save_message_label = tk.Label(self.master, text=save_message)
            save_message_label.pack()
            data.segmentation_results = seg_results
        # Set a flag to indicate that atlas segmentation has been performed
        
        # Create a popup window for selecting segmentation type
        popup_window = tk.Toplevel(self.master)
        popup_window.title("Select Segmentation Type")

        # Initialize variables to keep track of the current index for "Brain" and "Skull"
        brain_index = 0
        skull_index = 0
        current_segmentation = "Brain"  # Initialize with "Brain" as the default

        def update_image():
            nonlocal brain_index, skull_index, current_segmentation
            if current_segmentation == "Brain":
                image_list = png_dict['Brain']
                index = brain_index
            else:
                image_list = png_dict['Skull']
                index = skull_index

            image = image_list[index]
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo

        def handle_brain_skull_selection(segmentation_type):
            nonlocal current_segmentation
            if segmentation_type == "Brain":
                if current_segmentation == "Brain":
                    return  # If already on "Brain," do nothing
                current_segmentation = "Brain"
                update_image()
            else:
                if current_segmentation == "Skull":
                    return  # If already on "Skull," do nothing
                current_segmentation = "Skull"
                update_image()

        def handle_previous():
            nonlocal brain_index, skull_index
            if current_segmentation == "Brain":
                brain_index = (brain_index - 1) % len(png_dict['Brain'])
            else:
                skull_index = (skull_index - 1) % len(png_dict['Skull'])
            update_image()

        def handle_next():
            nonlocal brain_index, skull_index
            if current_segmentation == "Brain":
                brain_index = (brain_index + 1) % len(png_dict['Brain'])
            else:
                skull_index = (skull_index + 1) % len(png_dict['Skull'])
            update_image()

        # Create a label to display the image
        image_label = tk.Label(popup_window)
        image_label.pack()

        # Create a frame for the "Previous" and "Next" buttons
        button_frame = tk.Frame(popup_window)
        button_frame.pack()

        # Create buttons for "Previous" and "Next" in the popup window
        previous_button = tk.Button(button_frame, text="Previous", command=handle_previous)
        next_button = tk.Button(button_frame, text="Next", command=handle_next)
        previous_button.pack(side="left", padx=10)
        next_button.pack(side="right", padx=10)

        # Create buttons for "Brain" and "Skull" in the popup window
        brain_button = tk.Button(popup_window, text="Brain", command=lambda: handle_brain_skull_selection("Brain"))
        skull_button = tk.Button(popup_window, text="Skull", command=lambda: handle_brain_skull_selection("Skull"))
        brain_button.pack(pady=10)
        skull_button.pack(pady=10)

        # Initialize the initial segmentation type to "Brain"
        update_image()

    def show_popup_message(self, message, close_callback=None):
        # Create a new popup window
        popup_window = tk.Toplevel()
        popup_window.title("Popup")

        # Create a label to display the message
        label = tk.Label(popup_window, text=message)
        label.pack(padx=20, pady=10)

        def on_close():
            if close_callback:
                close_callback()
            popup_window.destroy()

        # Create a button to close the popup
        close_button = tk.Button(popup_window, text="Close", command=on_close)
        close_button.pack(pady=10)

        # Center the popup window on the screen
        popup_window.geometry(f"+{popup_window.winfo_screenwidth() // 2 - popup_window.winfo_reqwidth() // 2}+{popup_window.winfo_screenheight() // 2 - popup_window.winfo_reqheight() // 2}")

        # Start the main loop for the popup window
        popup_window.mainloop()    
                


        


    """def perform_segmentation_and_display(self, coords_dict):
        results = {}

        for key, coords in coords_dict.items():
            # Load the image from coordinates using data.py functions
            image = data.load_image_from_coords(coords)

            # Perform segmentation using your segmentation.py functions
            segmentation_result = segmentation.perform_segmentation(image)

            # Display the segmentation results (you can customize this part)
            self.display_segmentation_result(segmentation_result, key)

            results[key] = segmentation_result

        data.segmentation_results = results

    def convert_and_save_segmentation_results(self, segmentation_results, output_dir):
        if not segmentation_results:
            print("Segmentation results are empty. Perform segmentation first.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for key, seg_result in segmentation_results.items():
            dcm_file_path = os.path.join(output_dir, f"{key}.dcm")
            png_file_path = os.path.join(output_dir, f"{key}.png")
            data.save_sitk_3d_img_to_dcm(seg_result, dcm_file_path)
            data.save_sitk_3d_img_to_png(seg_result, png_file_path)"""
    
    

    def open_image_scoring_popup(self):
        #should take PIL images as input instead of directories

        image_paths = [
        "scan1_pngs/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011156_1_S32031_I54071.png",  # Replace with actual image paths
        "scan1_pngs/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011437_2_S32031_I54071.png"]

        #pillow images
        image1 = Image.open(image_paths[0])
        image2 = Image.open(image_paths[1])

        popup_window = tk.Toplevel(self.master)
        image_scoring_popup = ImageScoringPopup(popup_window, image1, image2, self.save_scores)
        

    def save_scores(self, score1, score2):
        # Implement your logic to save the scores here
        print("Score for Image 1:", score1)
        print("Score for Image 2:", score2)
    

    """def show_clustering_buttons(self):
        self.clustering_page.show_buttons()

    def show_deep_learning_buttons(self):
        self.deep_learning_page.show_buttons()"""

    def show_image_results(self):
        # This function will eventually display segmentation results for an image
        # You can add your image processing and display logic here
        # can take a directory (a folder containing sub-folders, each subfolder containing dcms) as input and then 
        # use PIL to turn them into images to display in a popup, similar to how ImageScoringPopup is now
        folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")
        while data.is_segment_results_dir(folder) != True:
            tk.messagebox.showwarning(title="Invalid Selection", message=
            "The folder you selected does not match the expected structure. Select a folder with sub-folders containg DCM files.")
            folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")

        image_dict = data.subfolders_to_dictionary(folder)
        pngs_dict = data.sitk_dict_to_png_dict(image_dict)

        #below is the same code that was used for the atlas_segment popup
        popup_window = tk.Toplevel(self.master)
        popup_window.title("Select Segmentation Type")
        brain_index = 0
        skull_index = 0
        current_segmentation = "Brain"  # Initialize with "Brain" as the default

        def update_image():
            nonlocal brain_index, skull_index, current_segmentation
            if current_segmentation == "Brain":
                image_list = pngs_dict['Brain']
                index = brain_index
            else:
                image_list = pngs_dict['Skull']
                index = skull_index
            image = image_list[index]
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo

        def handle_brain_skull_selection(segmentation_type):
            nonlocal current_segmentation
            if segmentation_type == "Brain":
                if current_segmentation == "Brain":
                    return  # If already on "Brain," do nothing
                current_segmentation = "Brain"
                update_image()
            else:
                if current_segmentation == "Skull":
                    return  # If already on "Skull," do nothing
                current_segmentation = "Skull"
                update_image()

        def handle_previous():
            nonlocal brain_index, skull_index
            if current_segmentation == "Brain":
                brain_index = (brain_index - 1) % len(pngs_dict['Brain'])
            else:
                skull_index = (skull_index - 1) % len(pngs_dict['Skull'])
            update_image()

        def handle_next():
            nonlocal brain_index, skull_index
            if current_segmentation == "Brain":
                brain_index = (brain_index + 1) % len(pngs_dict['Brain'])
            else:
                skull_index = (skull_index + 1) % len(pngs_dict['Skull'])
            update_image()

        image_label = tk.Label(popup_window)
        image_label.pack()
        button_frame = tk.Frame(popup_window)
        button_frame.pack()
        previous_button = tk.Button(button_frame, text="Previous", command=handle_previous)
        next_button = tk.Button(button_frame, text="Next", command=handle_next)
        previous_button.pack(side="left", padx=10)
        next_button.pack(side="right", padx=10)
        brain_button = tk.Button(popup_window, text="Brain", command=lambda: handle_brain_skull_selection("Brain"))
        skull_button = tk.Button(popup_window, text="Skull", command=lambda: handle_brain_skull_selection("Skull"))
        brain_button.pack(pady=10)
        skull_button.pack(pady=10)
        update_image()

    def show_folder_results(self):
        # This function will eventually display segmentation results for images in a folder
        # You can add your image processing and display logic here
        print("Displaying segmentation results for images in a folder") 

    def change_buttons(self, show_list, hide_list):
        for button in hide_list:
            button.pack_forget()
        for button in show_list:
            button.pack(pady=20) 

    def full_scan(self):
        #   while (self.selected_folder != dcm folder):
        #       popup says "select a folder containing dcms"
        #       select folder
        #   exec_full_scan_clustering(self.selected_folder, algo)
        #self.open_clustering_options_popup()
        
        while(data.contains_only_dcms(self.selected_folder)!=True or self.selected_folder == ""):
            self.selected_folder = filedialog.askdirectory(title = "Select a folder containing DCMs")
            if (self.selected_folder == ""): break # this triggers if the user clicks "cancel" or "X"

        if(self.advanced_algo.get() == "Deep Learning"):
        # there's a bit of redundance here, because the user will get asked to select a folder again    
            self.open_segmentation_selection_popup()
        elif(self.advanced_algo.get() == "Clustering"):
            self.open_clustering_options_popup()

        print("full scan button clicked")

    def pre_atlas_seg(self):
    #else if (pre_atlas_seg selected):
    #   while (self.selected_folder != seg folder):
    #          popup says "select a segmented dcm folder"
    #          select folder
    #          if (self.selected_folder == dcm folder):
    #               run atlas_seg(self.selected_folder)
    #   exec_seg_clustering(self.selected_folder, algo)
    
        while(self.selected_folder == "" or data.is_segment_results_dir(self.selected_folder)!=True):
            self.selected_folder = filedialog.askdirectory(title = "Select a segmented DCM folder")
            if(data.contains_only_dcms(self.selected_folder)):
                self.atlas_segment()
                break # while loop breaks if user selects a dcm folder
            if (self.selected_folder == ""): break # this triggers if the user clicks "cancel" or "X"

        if(self.advanced_algo.get() == "Deep Learning"):
            # there's a bit of redundance here, because the user will get asked to select a folder again    
            self.open_segmentation_selection_popup()

        elif(self.advanced_algo.get() == "Clustering"):
            self.open_clustering_options_popup()

        print("pre atlas seg clicked")

"""class ClusteringPage:
    def __init__(self, master, core_instance):
        self.master = master
        self.core_instance = core_instance

        self.back_button = tk.Button(self.master, text="Back", command=self.go_back)
        self.back_button.pack(pady=20)

        self.hide_buttons()

    def hide_buttons(self):
        self.back_button.pack_forget()

    def show_buttons(self):
        self.back_button.pack(pady=20)

    def go_back(self):
        self.hide_buttons()
        self.core_instance.advanced_segmentation_page.show_buttons()

class DeepLearningPage:
    def __init__(self, master, core_instance):
        self.master = master
        self.core_instance = core_instance

        self.back_button = tk.Button(self.master, text="Back", command=self.go_back)
        self.back_button.pack(pady=20)

        self.hide_buttons()

    def hide_buttons(self):
        self.back_button.pack_forget()

    def show_buttons(self):
        self.back_button.pack(pady=20)

    def go_back(self):
        self.hide_buttons()
        self.core_instance.advanced_segmentation_page.show_buttons()"""

# Usage
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x700")
    app = Core(root)
    root.mainloop()