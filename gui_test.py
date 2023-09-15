import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import Canvas, Scrollbar, Frame


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
    def __init__(self, master, image1_path, image2_path, callback):
        self.master = master
        self.callback = callback
        self.image1_path = image1_path
        self.image2_path = image2_path

        self.popup_frame = Frame(master)
        self.popup_frame.pack(fill='both', expand=True)

        self.canvas = Canvas(self.popup_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = Scrollbar(self.popup_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.inner_frame = Frame(self.canvas)
        self.inner_frame_canvas = self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.image1 = tk.PhotoImage(file=image1_path)
        self.image2 = tk.PhotoImage(file=image2_path)

        self.image_label1 = tk.Label(self.inner_frame, image=self.image1)
        self.image_label1.pack(pady=(20, 10), anchor="center")

        self.image_label2 = tk.Label(self.inner_frame, image=self.image2)
        self.image_label2.pack(pady=(20, 10), anchor="center")

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

    def submit_scores(self):
        try:
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

        self.deep_learning_button = tk.Button(self.master, text="Deep Learning", command=lambda:self.change_buttons([self.deeplearning_back_button],[self.deep_learning_button, self.clustering_button,self.clustering_algorithm_label, self.clustering_algorithm_combobox, self.execute_clustering_button, self.advanced_back_button]))

        self.advanced_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.advanced_segmentation_button, self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button],[self.deep_learning_button, self.clustering_button, self.advanced_back_button]))

        self.clustering_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.clustering_algorithm_label, self.clustering_algorithm_combobox, self.execute_clustering_button,self.clustering_text, self.results_label, self.clustering_back_button]))

        self.deeplearning_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.deeplearning_back_button]))

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
        
    def execute_clustering(self):
        # Get the selected clustering algorithm
        selected_algorithm = self.clustering_algorithm_combobox.get()

        clustering_results = ""
        # Implement clustering logic based on the selected algorithm
        if selected_algorithm == "K-Means":
        # Implement K-Means clustering logic here
            clustering_results = "K-Means clustering results..."
        elif selected_algorithm == "DBSCAN":
        # Implement DBSCAN clustering logic here
            clustering_results = "DBSCAN clustering results..."
        elif selected_algorithm == "Hierarchical":
        # Implement Hierarchical clustering logic here
            clustering_results = "Hierarchical clustering results..."
        elif selected_algorithm == "Other":
        # Implement your custom clustering algorithm logic here
            clustering_results = "Other clustering results..."

    # Display clustering results within the GUI or perform any desired actions
    # Display clustering results within the GUI
        self.display_clustering_results(clustering_results)
    # You can use labels or other widgets to display the clustering results.
    def display_clustering_results(self, clustering_results):
        # Create a label or canvas to display the clustering results
        self.results_label = tk.Label(self.master, text="Clustering Results:")
        self.results_label.pack()

        # Create a text widget to show the clustering results
        self.clustering_text = tk.Text(self.master, height=10, width=50)
        self.clustering_text.insert(tk.END, clustering_results)
        self.clustering_text.pack()
        
    def show_main_window(self):
        self.master.deiconify()  # Show the main window

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            print("Selected folder:", folder_path)
            self.selected_folder = folder_path
            self.update_folder_label()

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
        # image1_path = "C:\\Users\\kevin\\Documents\\classes\\ED1\\BrainAI-Segmentation\\scan 1\\ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011156_1_S32031_I54071.png"
        # image2_path = "C:\\Users\\kevin\\Documents\\classes\\ED1\\BrainAI-Segmentation\\scan 1\\ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011437_2_S32031_I54071.png"
        image1_path = "/Users/kylepalmer/Documents/GitHub/BrainAI-Segmentation/scan 1/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011156_1_S32031_I54071.png"  # Replace with actual image paths
        image2_path = "/Users/kylepalmer/Documents/GitHub/BrainAI-Segmentation/scan 1/ADNI_003_S_1257_PT_ADNI_br_raw_20070510122011437_2_S32031_I54071.png"

        popup_window = tk.Toplevel(self.master)
        image_scoring_popup = ImageScoringPopup(popup_window, image1_path, image2_path, self.save_scores)
        

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