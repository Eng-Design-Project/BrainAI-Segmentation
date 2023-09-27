import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import Canvas, Scrollbar, Frame
from PIL import Image, ImageTk  # Import PIL for image manipulation

import deep_learning
import clustering
import data

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

#global variables
segmentation_results= None
iteration = 0 # score iterations, meaning how many times the user has scored a pair of images

class ImageScoringPopup:
    def __init__(self, master, image_path, callback):
        self.master = master
        self.callback = callback
        self.image1_path = image_path[0]
        self.image2_path = image_path[1]
        print("master: ",self.master)

        self.image1 = tk.PhotoImage(file=image_path[0])
        self.image2 = tk.PhotoImage(file=image_path[1])

        self.image_label1 = tk.Label(self.master, image=self.image1)
        self.image_label1.pack(side="left", padx=20, pady=20)

        self.image_label2 = tk.Label(self.master, image=self.image2)
        self.image_label2.pack(side="right", padx=20, pady=20)

        self.score_label1 = tk.Label(self.master, text="Score Image 1:")
        self.score_label1.pack(pady=10)
        self.score_entry1 = tk.Scale(self.master, from_=1, to=10, orient="horizontal", sliderrelief='flat')
        self.score_entry1.pack()

        self.score_label2 = tk.Label(self.master, text="Score Image 2:")
        self.score_label2.pack(pady=10)
        self.score_entry2 = tk.Scale(self.master, from_=1, to=10, orient="horizontal", sliderrelief='flat')
        self.score_entry2.pack()

        self.submit_button = tk.Button(self.master, text="Submit", command=self.submit_scores)
        self.submit_button.pack(pady=20)                                                                    

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
            global iteration
            print(iteration)
            iteration += 1
            score1 = float(self.score_entry1.get())
            score2 = float(self.score_entry2.get())
        
            # Ensure the scores are within the desired range (0-10 or 1-10 or 1-5)
            min_score = min(score1, score2)
            max_score = max(score1, score2)

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

        self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation", command=lambda: self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button], [self.advanced_segmentation_button, self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button,self.clustering_algorithm_label, self.clustering_algorithm_combobox, self.execute_clustering_button]))
        self.advanced_segmentation_button.pack(pady=20)

        self.clustering_button = tk.Button(self.master, text="Clustering", command=lambda:self.change_buttons([self.clustering_algorithm_label, self.clustering_algorithm_combobox, self.execute_clustering_button, self.clustering_back_button],[self.advanced_segmentation_button, self.deep_learning_button, self.clustering_button, self.advanced_back_button]))

        self.deep_learning_button = tk.Button(self.master, text="Deep Learning", command=lambda:self.change_buttons([self.execute_deep_learning, self.deeplearning_back_button],[self.deep_learning_button, self.clustering_button,self.clustering_algorithm_label, self.clustering_algorithm_combobox, self.execute_clustering_button, self.advanced_back_button]))

        self.execute_deep_learning = tk.Button(self.master, text="Execute Deep Learning", command=self.execute_deep_learning_click)

        self.advanced_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.advanced_segmentation_button, self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button],[self.deep_learning_button, self.clustering_button, self.advanced_back_button]))

        self.clustering_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.clustering_algorithm_label, self.clustering_algorithm_combobox, self.execute_clustering_button, self.clustering_algorithm_label, self.previous_button,self.next_button, self.clustering_back_button]))

        self.deeplearning_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.execute_deep_learning, self.deeplearning_back_button]))

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

        self.clustering_algorithm_label = tk.Label(self.master, text="Select Clustering Algorithm:")
        #self.clustering_algorithm_label.pack(pady=10)
        self.clustering_algorithm_combobox = ttk.Combobox(self.master, values=["K-Means", "DBSCAN", "Hierarchical", "Other"])
        #self.clustering_algorithm_combobox.pack()

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

    def handle_clustering_selection(self, popup_window, algorithm, source):
        # Close the popup window
        popup_window.destroy()

        if source == "file":
            # Add code to get the selected file or folder here and store it
            selected_folder = self.get_selected_folder()
            data.set_seg_results(selected_folder)
            print("Selected file or folder:", selected_folder)

            # Logic to perform clustering from a file and set the clustering_results variable
            clustering_results = {}  # Implement file-based clustering logic here
        elif source == "memory":
            # Logic to perform clustering from memory and set the clustering_results variable
            data.set_seg_results()
            clustering_results = {}  # Implement memory-based clustering logic here

        # Display clustering results within the GUI
        self.display_clustering_results(algorithm, clustering_results)

        # You can use labels or other widgets to display the clustering results.

        # Set image paths and current image index (replace with your own data)
        self.image_paths = ["scan1_pngs/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011156_1_S32031_I54071.png", "scan1_pngs/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011437_2_S32031_I54071.png", "scan1_pngs/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011546_3_S32031_I54071.png"]
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
        results_label = tk.Label(self.master, text=f"Clustering Results for {algorithm}:")
        results_label.pack()
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
        #wrap all of this in an if statement that checks if data.segmentation results is empty ( and run the logic)
        #then we call the deep learning function with the segmentation results passed as a parameter
    #if (data.segmentation_results=={}):
        # Create a popup window for selecting segmentation results
        popup_window = tk.Toplevel(self.master)
        popup_window.title("Select Segmentation Results")

        # Create a label to instruct the user
        label = tk.Label(popup_window, text="Select segmentation results source:")
        label.pack(pady=10)

        # Create radio buttons for file and memory options
        selection_var = tk.StringVar()
        selection_var.set(None)  # Set an initial value that does not correspond to any option
        file_option = tk.Radiobutton(popup_window, text="From File", variable=selection_var, value="file")
        file_option.pack()
        memory_option = tk.Radiobutton(popup_window, text="From Memory", variable=selection_var, value="memory")
        memory_option.pack()

        # Create a button to confirm the selection
        confirm_button = tk.Button(popup_window, text="Confirm", command=lambda: self.handle_segmentation_selection(popup_window, selection_var.get()))
        confirm_button.pack(pady=20)
        # else:
        # Call the deep learning function with data.segmentation_results as a parameter
        #self.deep_learning_function(data.segmentation_results)


    def handle_segmentation_selection(self, popup_window, selection):
        # Close the popup window
        popup_window.destroy()

        if selection == "file":
            # Add code to get the selected folder here and store it
            selected_folder = self.get_selected_folder()
            data.set_seg_results(selected_folder)
            print("Selected folder:", selected_folder)
            # Logic to select segmentation results from a file and set the variable
            # You can use file dialogs to allow the user to choose a file
            segmentation_results = {}  # Implement file selection logic here
        elif selection == "memory":
            data.set_seg_results()
            # Logic to select segmentation results from memory and set the variable
            # You can populate segmentation_results with data from memory
            segmentation_results = {}  # Implement memory selection logic here

        # Now you have the segmentation_results variable with the selected data
        # You can use it for deep learning or any other processing
        print("Selected segmentation results:", segmentation_results)
        
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

    def atlas_segment(self):
        print("Atlas Segmentation called")
        # Implement your atlas segmentation logic here

    """def show_advanced_segmentation_buttons(self):
        if self.current_page:
            self.current_page.hide_buttons()

        # Show the buttons of the new page
        self.advanced_segmentation_page.show_buttons()

        # Update the current page
        self.current_page = self.advanced_segmentation_page"""

    def open_image_scoring_popup(self):
        global iteration
        image_paths = [
        "scan1_pngs/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011156_"+str(iteration*2+1)+"_S32031_I54071.png",  # Replace with actual image paths
        "scan1_pngs/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011156_"+str(iteration*2+2)+"_S32031_I54071.png"]

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
