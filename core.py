import tkinter as tk
from tkinter import filedialog
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
    def __init__(self, master,image_paths, callback):
        self.master = master
        self.callback = callback
        self.image_paths = image_paths
        self.current_image_index = 0

        self.popup_frame = Frame(master)
        self.popup_frame.pack(fill='both', expand=True)

        self.canvas = Canvas(self.popup_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = Scrollbar(self.popup_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.inner_frame = Frame(self.canvas)
        self.inner_frame_canvas = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.images = [Image.open(path) for path in image_paths]
        self.photo_images = [ImageTk.PhotoImage(image) for image in self.images]

        self.image_label = tk.Label(self.inner_frame, image=self.photo_images[self.current_image_index])
        self.image_label.pack(pady=(20, 10), anchor="center")

        self.prev_button = tk.Button(self.inner_frame, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side="left", padx=10)
        
        self.next_button = tk.Button(self.inner_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(side="right", padx=10)

        self.score_label1 = tk.Label(self.inner_frame, text="Score Image 1:")
        self.score_label1.pack(pady=10, anchor="center")

        self.score_entry1 = tk.Scale(self.inner_frame, from_=1, to=10, orient="horizontal", sliderrelief='flat')
        self.score_entry1.pack(pady=10, anchor="center")

        self.score_label2 = tk.Label(self.inner_frame, text="Score Image 2:")
        self.score_label2.pack(pady=10, anchor="center")

        self.score_entry2 = tk.Scale(self.inner_frame, from_=1, to=10, orient="horizontal", sliderrelief='flat')
        self.score_entry2.pack(pady=10, anchor="center")

        self.submit_button = tk.Button(self.inner_frame, text="Submit", command=self.submit_scores)
        self.submit_button.pack(pady=20, anchor="center")

        self.inner_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.inner_frame_canvas, width=event.width)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_label.config(image=self.photo_images[self.current_image_index])

    def show_next_image(self):
        if self.current_image_index < len(self.photo_images) - 1:
            self.current_image_index += 1
            self.image_label.config(image=self.photo_images[self.current_image_index])


    def submit_scores(self):
        try:
            score1 = float(self.score_entry1.get())
            score2 = float(self.score_entry2.get())
        
            # Ensure the scores are within the desired range (0-10 or 1-10 or 1-5)
            #min_score = min(score1, score2)
            #max_score = max(score1, score2)
            #min and max are hardcoded, not set by user (or risk divide by zero error)
            min_score = 1
            max_score = 10

            # Normalize the scores between 0 and 1
            normalized_score1 = (score1 - min_score) / (max_score - min_score)
            normalized_score2 = (score2 - min_score) / (max_score - min_score)

            # Save the normalized scores or pass them to your CNN
            self.callback(normalized_score1, normalized_score2)

            self.master.destroy()
        except ValueError:
            # Handle invalid input (e.g., non-numeric input)
            print("Invalid input. Please enter numeric scores.")

class Core:
    def __init__(self, master):
        self.master = master
        self.current_page = None  # Track the current page being displayed
        self.segmentation_results = {}  # Initialize the segmentation_results variable as an empty dictionary
        self.popup_window = None  # Add this line to define popup_window



        self.master.title("Image Analysis Tool")
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12))

        self.select_folder_button = tk.Button(self.master, text="Select Folder", command=self.select_folder)
        self.select_folder_button.pack(pady=20)
        self.selected_folder = ""

        self.folder_label = tk.Label(self.master, text="Selected Folder: ")
        self.folder_label.pack()

        self.atlas_segment_button = tk.Button(self.master, text="Atlas Segment", command=self.atlas_segment)
        self.atlas_segment_button.pack(pady=20)

        self.image_scoring_button = tk.Button(self.master, text="Score Images", command=self.open_image_scoring_popup)
        self.image_scoring_button.pack(pady=20)

        self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation", command=lambda: self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button], [self.advanced_segmentation_button, self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button, self.execute_clustering_button]))
        self.advanced_segmentation_button.pack(pady=20)

        self.clustering_button = tk.Button(self.master, text="Clustering", command=lambda:self.change_buttons([self.execute_clustering_button, self.clustering_back_button],[self.advanced_segmentation_button, self.deep_learning_button, self.clustering_button, self.advanced_back_button]))

        self.deep_learning_button = tk.Button(self.master, text="Deep Learning", command=lambda:self.change_buttons([self.U_Net_button, self.execute_deep_learning, self.deeplearning_back_button],[self.deep_learning_button, self.clustering_button, self.execute_clustering_button, self.advanced_back_button]))

        self.execute_deep_learning = tk.Button(self.master, text="Execute Deep Learning", command=self.execute_deep_learning_click)

        self.advanced_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.advanced_segmentation_button, self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button],[self.deep_learning_button, self.clustering_button, self.advanced_back_button]))

        self.clustering_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.results_label, self.execute_clustering_button, self.image_label, self.previous_button, self.next_button, self.clustering_back_button]))

        self.U_Net_button = tk.Button(self.master, text="U-Net", command=self.U_Net)

        self.deeplearning_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.execute_deep_learning, self.image_label ,self.previous_button, self.next_button, self.U_Net_button, self.deeplearning_back_button]))

        self.submit_brain_skull_button = tk.Button(self.popup_window, text="Submit", command=self.submit_segmentation_type)
        

        """self.image_file_path = 'mytest.png'
        self.image_button = tk.Button(self.master, text="Display Image", command=self.display_file_png)
        self.image_button.pack(pady=20)"""

        # Button for showing segmentation results for an image
        self.show_image_results_button = tk.Button(self.master, text="Show Image Results", command=self.show_image_results)
        self.show_image_results_button.pack(pady=20)

        # Button for showing segmentation results for a folder
        self.show_folder_results_button = tk.Button(self.master, text="Show Folder Results", command=self.show_folder_results)
        self.show_folder_results_button.pack(pady=20)

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
        selected_segmentation_method = self.get_selected_segmentation_method()

        if selected_segmentation_method == "atlas_segmentation":
            self.atlas_segment()
        else:
            if not data.segmentation_results:
                self.atlas_segment()
            self.open_clustering_options_popup()

    def open_clustering_options_popup(self):
        if self.get_selected_segmentation_method() == "atlas_segmentation" and not data.segmentation_results:
            self.select_folder()
        else:
            # Create a popup window for clustering options
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
        if self.image_paths:
            current_image_path = self.image_paths[self.current_image_index]
            image = Image.open(current_image_path)
            photo = ImageTk.PhotoImage(image)

            # Update the image label with the current image
            self.image_label.config(image=photo)
            self.image_label.photo = photo
            self.image_label.pack()

    def show_previous_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.show_current_image()

    def show_next_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.show_current_image()

    def execute_deep_learning_click(self):
        selected_segmentation_method = self.get_selected_segmentation_method()

        if selected_segmentation_method == "atlas_segmentation":
            self.atlas_segment()
        else:
            if not data.segmentation_results:
                self.atlas_segment()
            self.open_segmentation_selection_popup()


    def open_segmentation_selection_popup(self):
        #wrap all of this in an if statement that checks if data.segmentation results is empty ( and run the logic)
        #then we call the deep learning function with the segmentation results passed as a parameter
    #if (data.segmentation_results=={}):
        # Create a popup window for selecting segmentation results
        if self.get_selected_segmentation_method() == "atlas_segmentation" and not data.segmentation_results:
            self.select_folder()
        else:
            popup_window = tk.Toplevel(self.master)
            popup_window.title("Select Segmentation Results")

            label = tk.Label(popup_window, text="Select segmentation results source:")
            label.pack(pady=10)

            selection_var = tk.StringVar()
            selection_var.set(None)
            file_option = tk.Radiobutton(popup_window, text="From File", variable=selection_var, value="file")
            file_option.pack()
            memory_option = tk.Radiobutton(popup_window, text="From Memory", variable=selection_var, value="memory")
            memory_option.pack()

            confirm_button = tk.Button(popup_window, text="Confirm", command=lambda: self.handle_segmentation_selection(popup_window, selection_var.get()))
            confirm_button.pack(pady=20)

        #if seg results loaded from file, have to convert nested folder directory to dict of sitk images
        #alternative is running atlas seg

        # else:
        # Call the deep learning function with data.segmentation_results as a parameter
        #self.deep_learning_function(data.segmentation_results)


    def handle_segmentation_selection(self, popup_window, selection):
        # Close the popup window
        popup_window.destroy()

        if selection == "file":
            selected_folder = self.get_selected_folder()
            data.set_seg_results(selected_folder)
            print("Selected folder:", selected_folder)
        elif selection == "memory":
            data.set_seg_results()
            self.segmentation_results = {}

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
        folder_path = filedialog.askdirectory()
        if folder_path:
            print("Selected folder:", folder_path)
            self.selected_folder = folder_path
            self.update_folder_label()

    def get_selected_folder(self):
        folder_path = filedialog.askdirectory()
        return folder_path

    def update_folder_label(self):
        self.folder_label.config(text="Selected Folder: " + self.selected_folder)

    def display_file_png(self):
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

    def atlas_segment(self):
        print("Atlas Segmentation called")
        #gets selected folder global from core class
        #if empty, have to get it
        if(self.selected_folder == ""):
            print("no folder selected") 
            #prompt the user to select a folder
            self.select_folder()
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
        data.store_seg_img_on_file(seg_results, "atl_segmentation_DCMs")
        # save as pngs in nested folder by region structure
        data.store_seg_png_on_file(seg_results, "atl_segmentation_PNGs")
        # display pngs in gui
        # save dict of sitk images to data global seg results
        data.segmentation_results = seg_results
        # Set a flag to indicate that atlas segmentation has been performed
        # Create a popup window for segmentation type selection
        popup_window = tk.Toplevel(self.master)
        popup_window.title("Select Segmentation Type")

        # Create a label to instruct the user
        label = tk.Label(popup_window, text="Select segmentation type:")
        label.pack(pady=10)

        # Create a variable to store the selected segmentation type
        self.segmentation_type_var = tk.StringVar()
        self.segmentation_type_var.set("Brain")  # Default selection

        # Create radio buttons for "Brain" and "Skull" options
        brain_option = tk.Radiobutton(popup_window, text="Brain", variable=self.segmentation_type_var, value="Brain")
        brain_option.pack()
        skull_option = tk.Radiobutton(popup_window, text="Skull", variable=self.segmentation_type_var, value="Skull")
        skull_option.pack()

    # Create the "Submit" button in the popup window
        submit_brain_skull_button = tk.Button(popup_window, text="Submit", command=self.submit_segmentation_type)
        submit_brain_skull_button.pack(pady=10)

        self.popup_window = popup_window  # Store the popup window as an instance variable
    # Create a callback function for the submit button
    def submit_segmentation_type(self):
        selected_segmentation_type = self.segmentation_type_var.get()
        print("Selected Segmentation Type:", selected_segmentation_type)

        # Based on the selected type, display either brain or skull images
        if selected_segmentation_type == "Brain":
            # Display brain images
            self.display_segmentation_results("Brain")
        elif selected_segmentation_type == "Skull":
            # Display skull images
            self.display_segmentation_results("Skull")

        self.popup_window.destroy()

    def display_segmentation_results(self, segmentation_type):
        # This function will display segmentation results based on the selected type
        # You can implement your image display logic here
        print(f"Displaying {segmentation_type} images")
        # Add code to display the relevant images based on segmentation_type
        


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
        image_paths = [
        "scan1_pngs/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011156_1_S32031_I54071.png",  # Replace with actual image paths
        "scan1_pngs/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011437_2_S32031_I54071.png"]

        popup_window = tk.Toplevel(self.master)
        image_scoring_popup = ImageScoringPopup(popup_window, image_paths, self.save_scores)
        

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
        print("Displaying segmentation results for an image")

    def show_folder_results(self):
        # This function will eventually display segmentation results for images in a folder
        # You can add your image processing and display logic here
        print("Displaying segmentation results for images in a folder") 

    def change_buttons(self, show_list, hide_list):
        for button in hide_list:
            button.pack_forget()
        for button in show_list:
            button.pack(pady=20) 

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