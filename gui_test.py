import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import deep_learning
import segmentation
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


class ImageScoringPopup:
    def __init__(self, master, image1_path, image2_path, callback):
        self.master = master
        self.callback = callback
        self.image1_path = image1_path
        self.image2_path = image2_path

        self.image1 = tk.PhotoImage(file=image1_path)
        self.image2 = tk.PhotoImage(file=image2_path)

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

        self.advanced_segmentation_button = tk.Button(self.master, text="Advanced Segmentation", command=lambda: self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button], [self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button, self.advanced_segmentation_button]))
        self.advanced_segmentation_button.pack(pady=20)

        self.clustering_button = tk.Button(self.master, text="Clustering", command=lambda:self.change_buttons([self.clustering_back_button],[self.deep_learning_button, self.advanced_back_button, self.clustering_button]))

        # Define the Deep Learning button here with the correct command
        self.deep_learning_button = tk.Button(self.master, text="Deep Learning", command=self.start_deep_learning)
        self.deep_learning_button.pack(pady=20)

        self.advanced_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.atlas_segment_button, self.show_image_results_button, self.show_folder_results_button],[self.deep_learning_button, self.clustering_button, self.advanced_back_button]))

        self.clustering_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.advanced_segmentation_button, self.clustering_back_button]))

        self.deeplearning_back_button = tk.Button(self.master, text="Back", command=lambda:self.change_buttons([self.deep_learning_button, self.clustering_button, self.advanced_back_button],[self.advanced_segmentation_button,self.deeplearning_back_button]))


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
        # update the global vars in the deep learning module
        deep_learning.get_user_score(score1, score2)

    def start_deep_learning(self):
        # Implement your deep learning logic here
        print("Initiating Deep Learning tasks")    
    

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
        for button in show_list:
            button.pack(pady=20)
        for button in hide_list:
            button.pack_forget()   
    
    # a function that will take a dictionary with regions (keys) and coordinates (values) as input,
    # it will call create_seg_img() from the segmenation module; it's gonna spit out images 
    # then we're calling the function in data that's saving the segmented image on file
    # this is not yet complete, needs testing
    def save_segmentation(image, region_dict, new_dir):
        region_images = segmentation.create_seg_images(image, region_dict)
        data.store_seg_img_on_file(region_images, new_dir)
        print("save_segmentation() has been called")


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