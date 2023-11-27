import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk
from tkinter import Canvas, Scrollbar, Frame
from tkinter import messagebox
import numpy as np


from PIL import Image, ImageTk, ImageOps  # Import PIL for image manipulation
from tkinter import Toplevel, Radiobutton, Button, StringVar



#import deep_learning
import clustering
import segmentation
import data
import os
import deep_learning_copy
import deep_learning


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
        self.atlas_segmentation_completed = False  # Initialize the attribute as False
        # Add a variable to track the state of clustering description visibility
        self.show_clustering_description = False


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
        self.internal_atlas_segment_button = tk.Button(self.master, text="Internal Atlas Segmentation", command=self.execute_internal_atlas_seg)
        self.internal_atlas_segment_button.pack_forget()
        
        # Image scoring button
        self.image_scoring_button = tk.Button(self.master, text="Score Images", command=self.test_open_image_popup)
        self.image_scoring_button.pack(pady=20)

        # Advanced segmentation button
        self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation", command=lambda: self.change_buttons([self.select_folder_button, self.folder_label, self.image_scoring_button, self.deep_learning_button, self.clustering_button, self.save_message_label, self.advanced_back_button], self.master))
        self.advanced_segmentation_button.pack(pady=20)

        # Clustering button
        self.clustering_button = tk.Button(self.master, text="Clustering", command=self.clustering_click)

        # Deep learning button
        self.deep_learning_button = tk.Button(self.master, text="Deep Learning", command=lambda:self.deep_learning_click())

        # Advanced segmentation back button
        self.advanced_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.select_folder_button, self.folder_label, self.atlas_segment_button, self.image_scoring_button, self.advanced_segmentation_button, self.show_image_results_button, self.view_DCMS_btn, self.toggle_desc_button, self.save_message_label], self.master))

        """self.image_file_path = 'mytest.png'
        self.image_button = tk.Button(self.master, text="Display Image", command=self.display_file_png)
        self.image_button.pack(pady=20)"""

        # Button for showing segmentation results for an image
        self.show_image_results_button = tk.Button(self.master, text="Show Segmentation Results", command=self.show_seg_results)
        self.show_image_results_button.pack(pady=20)

        # Button for showing segmentation results for a folder
        self.view_DCMS_btn = tk.Button(self.master, text="View DCM Images from Folder", command=self.view_DCMs_from_file)
        self.view_DCMS_btn.pack(pady=20)

        # Add a toggle button for showing/hiding clustering descriptions
        self.toggle_desc_button = tk.Button(self.master, text="Show Descriptions", command=self.toggle_descriptions)
        self.toggle_desc_button.pack(pady=10)

        #self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation", command=lambda: self.change_buttons([], [self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button]))
        #self.advanced_segmentation_button.pack(pady=20)

        self.image_paths = []  # List to store image file paths
        self.current_image_index = 0  # Index of the currently displayed image

        # Create "Previous" and "Next" buttons for image navigation
        self.previous_button = tk.Button(self.master, text="Previous", command=self.show_previous_image)
        self.next_button = tk.Button(self.master, text="Next", command=self.show_next_image)

        self.image_label = tk.Label(self.master)

        self.save_message_label = tk.Label(self.master, text="")
        self.save_message_label.pack_forget()
        
    def clustering_click(self):
    
        # Create a popup window for clustering options
        popup_window = tk.Toplevel(self.master)
        popup_window.title("Select Clustering Parameters")

        # Create a label to instruct the user
        label = tk.Label(popup_window, text="Select clustering parameters:")
        label.pack(pady=10)

        # Add a separator for the first category
        separator1 = ttk.Separator(popup_window, orient="horizontal")
        separator1.pack(fill="x", padx=20, pady=5)

        segment_description = tk.Label(popup_window, text="Choose 'Whole Scan' for clustering the entire scan or 'Segment' to select the results of segmentation.")
        segment_description.pack()
        segment_var = tk.StringVar()
        segment_var.set(None)
        whole_brain = tk.Radiobutton(popup_window, text="Whole Scan", variable=segment_var, value="Whole Scan")
        whole_brain.pack()
        segment = tk.Radiobutton(popup_window, text="Segment", variable=segment_var, value="Segment")
        segment.pack()
        #Note: We need to grey out or otherwise deselect options for any algos but DBSCAN if whole scan selected 
        #temporarily, just have DBSCAN be the default value

        # Add a separator for the second category
        separator2 = ttk.Separator(popup_window, orient="horizontal")
        separator2.pack(fill="x", padx=20, pady=5)

        # Create radio buttons for clustering algorithm options
        algorithm_description = tk.Label(popup_window, text="Select a clustering algorithm to also view its description.")
        algorithm_description.pack()
        algorithm_var = tk.StringVar()
        algorithm_var.set(None)
        kmeans_option = tk.Radiobutton(popup_window, text="K-Means", variable=algorithm_var, value="K-Means", command=lambda: self.show_algorithm_description("K-Means", "Best for more uniform scans featuring roughly spherical clusters of similar size and intensity."))
        kmeans_option.pack()
        dbscan_option = tk.Radiobutton(popup_window, text="DBSCAN", variable=algorithm_var, value="DBSCAN", command=lambda: self.show_algorithm_description("DBSCAN", "Optimal for scans with significant noise and arbitrarily shaped clusters; particularly effective for areas with visible, distinct density variations."))
        dbscan_option.pack()
        hierarchical_option = tk.Radiobutton(popup_window, text="Hierarchical", variable=algorithm_var, value="Hierarchical", command=lambda: self.show_algorithm_description("Hierarchical", "Best suited for scans indicating a nested structure as it could help reveal the relationship between smaller clusters that are nested within larger ones."))
        hierarchical_option.pack()

        # Add a separator for the third category
        separator3 = ttk.Separator(popup_window, orient="horizontal")
        separator3.pack(fill="x", padx=20, pady=5)

        source_description = tk.Label(popup_window, text="Choose the data source for clustering.")
        source_description.pack()
        # Create radio buttons for data source options
        source_var = tk.StringVar()
        source_var.set(None)  # Set an initial value that does not correspond to any option
        file_option = tk.Radiobutton(popup_window, text="From File", variable=source_var, value="file")
        file_option.pack()
        memory_option = tk.Radiobutton(popup_window, text="From Memory (recent Seg Results)", variable=source_var, value="memory")
        memory_option.pack()

        # Create a button to confirm the selection and execute clustering
        confirm_button = tk.Button(popup_window, text="Execute Clustering", command=lambda: self.handle_clustering_selection(popup_window, segment_var.get(), algorithm_var.get(), source_var.get()))
        confirm_button.pack(pady=20)

    def show_algorithm_description(self, algorithm_name, algorithm_description):
        # Only show the description if the toggle is on
        if self.show_clustering_description:
            messagebox.showinfo(algorithm_name, algorithm_description)

    def toggle_descriptions(self):
        # Toggle the state
        self.show_clustering_description = not self.show_clustering_description

        # Update the button text based on the state
        self.toggle_desc_button.config(text="Hide Descriptions" if self.show_clustering_description else "Show Descriptions")
        
        
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

    def handle_clustering_selection(self, popup_window, seg_var, algorithm, source):
        folder = "" #this variable is local to the function
        print(source,", ", seg_var, ", ",algorithm )
        if source == "memory":
            if seg_var == "Segment":
                if not data.segmentation_results:
                    #note, data.segmentation results is set after atlas_segment() function is called
                    tk.messagebox.showwarning(title="Invalid Selection", message="No segmentation in memory, you need to select from file.")
                    source = "file"
            if seg_var == "Whole Scan":
                if not self.selected_folder:
                    tk.messagebox.showwarning(title="Invalid Selection", message="No Scan in memory, you need to select from file.")
                    source = "file"                
                elif not data.contains_only_dcms(self.selected_folder):
                    tk.messagebox.showwarning(title="Invalid Selection", message="You need to select from file containing only DCMs.")
                    source = "file"
                else:
                    volume = data.get_3d_image(self.selected_folder)



        if source == "file":
            if seg_var == "Segment":
                data.segmentation_results = None
                while not data.segmentation_results: #note, this may result in infinite loop, need some flag if window closed
                    folder = filedialog.askdirectory(title="Select folder with segmentation results")
                    if folder == '':
                        popup_window.destroy()
                        return
                    #check if save folder matches expected structure
                    if data.is_segment_results_dir(folder):
                        #the function below sets data.segmentation_results to an 3d np array image dict
                        data.set_seg_results_with_dir(folder)
                    else:
                        tk.messagebox.showwarning(title="Invalid Selection", message="The folder you selected does not match the expected structure. Select a folder with sub-folders containg DCM files.")
                        # in the future, add logic to query to user if they want to do atlas seg first,
                        # if contains_only_dcms(selection) == true
            if seg_var == "Whole Scan":
                folder = filedialog.askdirectory(title="Select folder with dcms")
                if folder == '':
                        popup_window.destroy()
                        return
                while not data.contains_only_dcms(folder):#note, this may result in infinite loop, need some flag if window closed
                    tk.messagebox.showwarning(title="Invalid Selection", message="Select a folder containing only DCM files.")
                    folder = filedialog.askdirectory(title="Select folder with dcms")
                    if folder == '':
                        popup_window.destroy()
                        return
                volume = data.get_3d_image(folder)
        
            
            
                
        #implement in popup selection later
        #if (seg scan && !seg_results) || (full scan && !selected_file)
            #user shouldn't been able to select 'from memory'
        
        if seg_var == "Segment":
            #the first argument should be a pre-atlas segmented scan, the 2nd argument should be a string of the chosen algo
            dict_of_coords_dicts = clustering.execute_seg_clustering(data.segmentation_results, algorithm, 5)
            for region in data.segmentation_results.keys():
                clustered_dict = segmentation.create_seg_images_from_image(data.segmentation_results[region], dict_of_coords_dicts[region])
                self.show_seg_results(clustered_dict)
                # for cluster in clustered_dict.keys():
                #     self.show_image_results(clustered_dict[cluster])
                    
        if seg_var == "Whole Scan":
            # note, when it comes to whole scan, only the DBSCAN algorithm works at the moment
        
            #cluster coordinates returned, not noise, actual clusters for now
            #could user select number of clusters?
            coords_dict = clustering.execute_whole_clustering(volume, algorithm, 5)
            
            #seg brain with cluster coords
            clustered_dict = segmentation.create_seg_images_from_image(volume, coords_dict)

            #saving clusters breaks: cluster names are integers
            #self.save_seg_results(clustered_dict)
            
            self.show_seg_results(clustered_dict)
    
        # Close the popup window
        popup_window.destroy()

    #this isn't currently in use, any type of seg results should be displayed by one function
    def display_clustering_results_deprecated(self, algorithm, clustering_results):
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

    def deep_learning_click(self):
        #wrap all of this in an if statement that checks if data.segmentation results is empty ( and run the logic)
        #then we call the deep learning function with the segmentation results passed as a parameter
    #if (data.segmentation_results=={}):
        # Create a popup window for selecting segmentation results
    
        # Create a popup window for clustering options
        popup_window = tk.Toplevel(self.master)
        popup_window.title("Select DL Parameters")

        # Create a label to instruct the user
        label = tk.Label(popup_window, text="Select DL parameters:")
        label.pack(pady=10)

        # Add a separator for the first category
        separator1 = ttk.Separator(popup_window, orient="horizontal")
        separator1.pack(fill="x", padx=20, pady=5)

        segment_var = tk.StringVar()
        segment_var.set(None)
        whole_brain = tk.Radiobutton(popup_window, text="Whole Scan", variable=segment_var, value="Whole Scan")
        whole_brain.pack()
        segment = tk.Radiobutton(popup_window, text="Segment", variable=segment_var, value="Segment")
        segment.pack()
        #Note: We need to grey out or otherwise deselect options for any algos but DBSCAN if whole scan selected 
        #temporarily, just have DBSCAN be the default value

        # Add a separator for the second category
        separator2 = ttk.Separator(popup_window, orient="horizontal")
        separator2.pack(fill="x", padx=20, pady=5)

        # Create radio buttons for clustering algorithm options
        algorithm_var = tk.StringVar()
        algorithm_var.set(None)
        kmeans_option = tk.Radiobutton(popup_window, text="U-net", variable=algorithm_var, value="U-net")
        kmeans_option.pack()
        dbscan_option = tk.Radiobutton(popup_window, text="Custom", variable=algorithm_var, value="Custom")
        dbscan_option.pack()

        # Add a separator for the third category
        separator3 = ttk.Separator(popup_window, orient="horizontal")
        separator3.pack(fill="x", padx=20, pady=5)

        # Create radio buttons for data source options
        source_var = tk.StringVar()
        source_var.set(None)  # Set an initial value that does not correspond to any option
        file_option = tk.Radiobutton(popup_window, text="From File", variable=source_var, value="file")
        file_option.pack()
        memory_option = tk.Radiobutton(popup_window, text="From Memory (recent Seg Results or selected file)", variable=source_var, value="memory")
        memory_option.pack()

        # Create a button to confirm the selection and execute clustering
        confirm_button = tk.Button(popup_window, text="Execute Deeplearning", command=lambda: self.handle_deeplearning_selection(popup_window, segment_var.get(), algorithm_var.get(), source_var.get()))
        confirm_button.pack(pady=20)

        #if seg results loaded from file, have to convert nested folder directory to dict of 3d np arrays
        #alternative is running atlas seg

        # else:
        # Call the deep learning function with data.segmentation_results as a parameter
        #self.deep_learning_function(data.segmentation_results)


    def handle_deeplearning_selection(self, popup_window, seg_var, algorithm, source):
        folder = "" #this variable is local to the function
        print(source,", ", seg_var, ", ",algorithm )
        if source == "memory":
            if seg_var == "Segment":
                if not data.segmentation_results:
                    #note, data.segmentation results is set after atlas_segment() function is called
                    tk.messagebox.showwarning(title="Invalid Selection", message="No segmentation in memory, you need to select from file.")
                    source = "file"
            if seg_var == "Whole Scan":
                if not self.selected_folder:
                    tk.messagebox.showwarning(title="Invalid Selection", message="No Scan in memory, you need to select from file.")
                    source = "file"                
                elif not data.contains_only_dcms(self.selected_folder):
                    tk.messagebox.showwarning(title="Invalid Selection", message="You need to select from file containing only DCMs.")
                    source = "file"
                else:
                    volume = data.get_3d_image(self.selected_folder)



        if source == "file":
            if seg_var == "Segment":
                data.segmentation_results = None
                while not data.segmentation_results: #note, this may result in infinite loop, need some flag if window closed
                    folder = filedialog.askdirectory(title="Select folder with segmentation results")
                    if folder == '':
                        popup_window.destroy()
                        return
                    #check if save folder matches expected structure
                    if data.is_segment_results_dir(folder):
                        #the function below sets data.segmentation_results to an 3d np array image dict
                        data.set_seg_results_with_dir(folder)
                    else:
                        tk.messagebox.showwarning(title="Invalid Selection", message="The folder you selected does not match the expected structure. Select a folder with sub-folders containg DCM files.")
                        # in the future, add logic to query to user if they want to do atlas seg first,
                        # if contains_only_dcms(selection) == true
            if seg_var == "Whole Scan":
                folder = filedialog.askdirectory(title="Select folder with dcms")
                if folder == '':
                    popup_window.destroy()
                    return
                while not data.contains_only_dcms(folder):#note, this may result in infinite loop, need some flag if window closed
                    tk.messagebox.showwarning(title="Invalid Selection", message="Select a folder containing only DCM files.")
                    folder = filedialog.askdirectory(title="Select folder with dcms")
                    if folder == '':
                        popup_window.destroy()
                        return
                volume = data.get_3d_image(folder)
        
            
            
                
        #implement in popup selection later
        #if (seg scan && !seg_results) || (full scan && !selected_file)
            #user shouldn't been able to select 'from memory'
        
        if seg_var == "Segment":
            #the first argument should be a pre-atlas segmented scan, the 2nd argument should be a string of the chosen algo
            if algorithm == "U-net":
                print("segment u-net")
                
            else:
                print("segment custom")
                self.train_custom_dl_model(data.segmentation_results)
            
                    
        if seg_var == "Whole Scan":
            if algorithm == "U-net":
                print("whole scan u-net")
            else:
                print("whole scan custom")
                self.train_custom_dl_model(volume)

            
            #self.show_seg_results(dl_dict)
    
        # Close the popup window
        popup_window.destroy()

    def train_custom_dl_model(self, input):
        print("training custom dl model")
        dict_of_3d_arrays = {}
        if isinstance(input, dict):
            print("input is a dictionary.")
            dict_of_3d_arrays = input
        else:
            print("input is an array.")
            dict_of_3d_arrays["FullScan"] = input
        classifier = deep_learning.CustomClassifierMultiModel(dict_of_3d_arrays)
        classif_dict = classifier.trainDL()
        results = segmentation.filter_noise_from_images(dict_of_3d_arrays, classif_dict)
        self.show_seg_results(results)
            


        
    def get_selected_segmentation_method(self):
        #later, this will be removed. when a button calls a function,
        #  we should know which algo is being called by passing an argument
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
        folder_path = filedialog.askdirectory(title="Select Folder")
        # If a folder is selected, store it and update the folder label
        if folder_path:
            print("Selected folder:", folder_path)
            self.selected_folder = folder_path
            self.update_folder_label()


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
        images_dict = {
            "image1": data.get_3d_image("scan1"),
            "image2": data.get_3d_image("scan2"),
        }

        # Call the dlAlgorithm function from deep_learning_copy module
        deep_learning_copy.dlAlgorithm(images_dict)

    def custom_askdirectory(title):
        #might replace other usses of askdirectory, to display a message

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
        # Check if the selected folder is a valid segment results directory
        if (not data.contains_only_dcms(self.selected_folder) or self.selected_folder == ""):
            self.show_popup_message("Incorrect input folder, select a folder only containing DCMs")
            # Clear the selected folder
            self.selected_folder = ""
            return  # Exit the function or handle the invalid folder as needed   
               
        # use functions in data to read the atlas and 
        #   the image in question into memory here as 3d np array images
        image = data.get_3d_image(self.selected_folder)

        image_depth = image.shape[0]   
        atlas_paths = data.get_atlas_path(image_depth)
        #if atlas_paths = 0  popup "Image too large" break out of function
        atlas = data.get_3d_image(atlas_paths[0])
        # get atlas colors as 3d np array
        color_atlas = data.get_2d_png_array_list(atlas_paths[1])

             
        # call execute atlas seg, passing image, atlas and atlas colors as args
        seg_results = segmentation.execute_atlas_seg(atlas, color_atlas, image)
        del seg_results["Skull"]
        data.segmentation_results = seg_results
        # returns dict of simple itk images
        # save them as dcms to the nested folder
        # Check if the selected folder is a valid segment results directory
        
        #save seg results to file, and to data.segmentation_results
        save_success = self.save_seg_results(seg_results)
        if (save_success):
            # Set a flag to indicate that atlas segmentation has been performed
                self.atlas_segmentation_completed = True
                print("Atlas segmentation completed")  # Add this line for debugging
                self.change_buttons([self.select_folder_button, self.folder_label, self.atlas_segment_button, self.image_scoring_button, self.advanced_segmentation_button, self.show_image_results_button, self.view_DCMS_btn, self.save_message_label], self.master)
        else:
            print("Failed to save Results")
            self.show_popup_message("Failed to save results")
        #display seg results
        self.show_seg_results(seg_results)

        #here to test execute internal_atlas_seg
        #self.execute_internal_atlas_seg()

    def execute_internal_atlas_seg(self):
        print("Internal Atlas Segmentation")
        if (data.segmentation_results != None):
            internal_color_atlas = data.get_2d_png_array_list("Color Atlas internal")
            internal_seg_results = segmentation.execute_internal_atlas_seg(data.segmentation_results, internal_color_atlas)
            #save results, to file and data.seg results        
            save_success = self.save_seg_results(internal_seg_results)
            if (save_success):
                print("Internal Atlas segmentation completed")  # Add this line for debugging
            else:
                self.show_popup_message("Failed to save results")
                #display results
            self.show_seg_results(internal_seg_results)
        else:
            self.show_popup_message("There are no atlas segmentation results saved to internally segment.")

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
                return True       
                
        #Note: returns True to indicate success, False otherwise
        return False

    

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

    def test_open_image_popup(self):
        folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")
        while not data.is_segment_results_dir(folder):
            tk.messagebox.showwarning(
                title="Invalid Selection",
                message="The folder you selected does not match the expected structure. Select a folder with sub-folders containing DCM files.")
            folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")
        image_dict = data.subfolders_to_dictionary(folder)
        scores_dict = self.open_image_scoring_popup(image_dict)
        

    def open_image_scoring_popup(self, image_dict):
        score_dictionary = {}
        
        # Assuming data.array_dict_to_png_dict is a function you have defined
        pngs_dict = data.array_dict_to_png_dict(image_dict)

        popup_window = tk.Toplevel(self.master)
        popup_window.title("Results of Segmentation")

        # New dictionary to keep track of indexes for each segmentation type
        segmentation_indexes = {region: 0 for region in pngs_dict.keys()}
        current_segmentation = next(iter(pngs_dict))  # Initialize with the first key

        image_label = tk.Label(popup_window)
        image_label.pack()

        # Score scale
        score_scale = tk.Scale(popup_window, from_=0, to=10, orient="horizontal")
        score_scale.pack()

        def update_image():
            index = segmentation_indexes[current_segmentation]
            image_list = pngs_dict[current_segmentation]
            image = image_list[index]
            photo = ImageTk.PhotoImage(image)
            image_label.configure(image=photo)
            image_label.image = photo
            # Update scale to current score or default
            score_scale.set(score_dictionary.get(current_segmentation, 0))

        def score_image():
            score = score_scale.get()
            score_dictionary[current_segmentation] = score
            print(f"Scored {current_segmentation}: {score}")

        score_button = tk.Button(popup_window, text="Score Image", command=score_image)
        score_button.pack()

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
        popup_window.geometry("400x400")  # Adjust width and height as needed 

        def finalize_scoring():
            popup_window.destroy()  # This closes the popup window
            for key in score_dictionary.keys():
                print(score_dictionary[key])
            return score_dictionary

        # Add a finalize button
        finalize_button = tk.Button(popup_window, text="Finalize Scoring", command=finalize_scoring)
        finalize_button.pack() 
  

        #min_score = 1 # Minimum possible score
        #max_score = 10 # Maximum possible score

            # Normalize the scores between 0 and 1
        #normalized_score1 = (score1 - min_score) / (max_score - min_score)
        #normalized_score2 = (score2 - min_score) / (max_score - min_score)

    """def show_clustering_buttons(self):
        self.clustering_page.show_buttons()

    def show_deep_learning_buttons(self):
        self.deep_learning_page.show_buttons()"""
   
        
    def show_seg_results(self, image_dict=None):
        # This function will display segmentation results for an image

        if image_dict is None:
            folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")
            if folder == '':
                        return
            while not data.is_segment_results_dir(folder):
                tk.messagebox.showwarning(
                    title="Invalid Selection",
                    message="The folder you selected does not match the expected structure. Select a folder with sub-folders containing DCM files.")
                folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")
                if folder == '':
                        return
            image_dict = data.subfolders_to_dictionary(folder)
        print("Showing seg results")
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

            # Convert the image to grayscale
            image = ImageOps.grayscale(image)

            # Set all white pixels to black (assuming white pixels have value 255)
            image = image.point(lambda p: p if p != 255 else 0)

            # Resize the image to 224x224 using LANCZOS (formerly ANTIALIAS)
            image = image.resize((224, 224), Image.Resampling.LANCZOS)

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

    # def show_image_results(self, image_dict=None):
    #     # This function will eventually display segmentation results for an image
    #     # You can add your image processing and display logic here
    #     # can take a directory (a folder containing sub-folders, each subfolder containing dcms) as input and then 
    #     # use PIL to turn them into images to display in a popup, similar to how ImageScoringPopup is now
    #     if image_dict == None:
    #         folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")
    #         while data.is_segment_results_dir(folder) != True:
    #             tk.messagebox.showwarning(title="Invalid Selection", message=
    #             "The folder you selected does not match the expected structure. Select a folder with sub-folders containg DCM files.")
    #             folder = filedialog.askdirectory(title="Select folder with subfolders containing DCM files")
    #         image_dict = data.subfolders_to_dictionary(folder)
    #     pngs_dict = data.array_dict_to_png_dict(image_dict)

    #     #below is the same code that was used for the atlas_segment popup
    #     popup_window = tk.Toplevel(self.master)
    #     popup_window.title("Results of Segmentation")
    #     brain_index = 0
    #     skull_index = 0
    #     current_segmentation = "Brain"  # Initialize with "Brain" as the default

    #     def update_image():
    #         nonlocal brain_index, skull_index, current_segmentation
    #         if current_segmentation == "Brain":
    #             image_list = pngs_dict['Brain']
    #             index = brain_index
    #         else:
    #             image_list = pngs_dict['Skull']
    #             index = skull_index
    #         image = image_list[index]
    #         photo = ImageTk.PhotoImage(image)
    #         image_label.configure(image=photo)
    #         image_label.image = photo

    #     def handle_brain_skull_selection(segmentation_type):
    #         nonlocal current_segmentation
    #         if segmentation_type == "Brain":
    #             if current_segmentation == "Brain":
    #                 return  # If already on "Brain," do nothing
    #             current_segmentation = "Brain"
    #             update_image()
    #         else:
    #             if current_segmentation == "Skull":
    #                 return  # If already on "Skull," do nothing
    #             current_segmentation = "Skull"
    #             update_image()

    #     def handle_previous():
    #         nonlocal brain_index, skull_index
    #         if current_segmentation == "Brain":
    #             brain_index = (brain_index - 1) % len(pngs_dict['Brain'])
    #         else:
    #             skull_index = (skull_index - 1) % len(pngs_dict['Skull'])
    #         update_image()

    #     def handle_next():
    #         nonlocal brain_index, skull_index
    #         if current_segmentation == "Brain":
    #             brain_index = (brain_index + 1) % len(pngs_dict['Brain'])
    #         else:
    #             skull_index = (skull_index + 1) % len(pngs_dict['Skull'])
    #         update_image()

    #     image_label = tk.Label(popup_window)
    #     image_label.pack()
    #     button_frame = tk.Frame(popup_window)
    #     button_frame.pack()
    #     previous_button = tk.Button(button_frame, text="Previous", command=handle_previous)
    #     next_button = tk.Button(button_frame, text="Next", command=handle_next)
    #     previous_button.pack(side="left", padx=10)
    #     next_button.pack(side="right", padx=10)
    #     brain_button = tk.Button(popup_window, text="Brain", command=lambda: handle_brain_skull_selection("Brain"))
    #     skull_button = tk.Button(popup_window, text="Skull", command=lambda: handle_brain_skull_selection("Skull"))
    #     brain_button.pack(pady=10)
    #     skull_button.pack(pady=10)
    #     update_image()
    #     popup_window.geometry("300x300")  # Adjust width and height as needed

    def view_DCMs_from_file(self):
        # This function will eventually display DCMs from file
        # note, currently only works for un-segmented DCMs
        folder = filedialog.askdirectory(title="Select a folder containing only DCM files")
        if folder == '':
            return
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
            self.open_deeplearning_options_popup()
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
            self.open_deeplearning_options_popup()

        elif(self.advanced_algo.get() == "Clustering"):
            self.open_clustering_options_popup()

        print("pre atlas seg clicked")



# Usage
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x700")
    app = Core(root)
    root.mainloop()