import os

def reverse_file_order(folder_path, file_extension):
    # Find all files with the specified extension
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]

    # Extract the numbers from the filenames and sort them
    filenumbers = [int(f.split('_')[-1].split('.')[0]) for f in files]
    filenumbers.sort()

    # Find the maximum number to set the new naming scheme
    max_number = max(filenumbers)

    # First, rename all files to a temporary naming scheme
    temp_files = []
    for f in files:
        temp_filename = f.replace(file_extension, '.temp' + file_extension)
        os.rename(os.path.join(folder_path, f), os.path.join(folder_path, temp_filename))
        temp_files.append(temp_filename)

    # Rename the temporary files to the new names
    for temp_f in temp_files:
        old_number = int(temp_f.split('_')[-1].split('.')[0])
        new_number = max_number - old_number
        new_filename = temp_f.replace('.temp', '').replace(str(old_number), str(new_number))
        os.rename(os.path.join(folder_path, temp_f), os.path.join(folder_path, new_filename))


folder_path = "Sorted_Brain_Scans_PNG_Colored - Copy"
file_extension = ".png"  # or ".dcm" for DCM files
reverse_file_order(folder_path, file_extension)