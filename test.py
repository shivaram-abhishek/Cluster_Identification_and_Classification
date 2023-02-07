
import os
import brainExtraction
import shutil

# dir = ".\\PatientData\\Data"
dir = ".\\testPatient"

files_list = os.listdir(dir)

slices_dir = os.path.join(os.getcwd(), "Slices")
boundary_dir = os.path.join(os.getcwd(), "Boundaries")

if os.path.exists(slices_dir):
    shutil.rmtree(slices_dir)

if os.path.exists(boundary_dir):
    shutil.rmtree(boundary_dir)

os.mkdir(slices_dir)
os.mkdir(boundary_dir)

for file_name in files_list:
    if "thresh" in file_name:
        img_name = os.path.join(dir, file_name)
        slices_folder = brainExtraction.slice_images(img_name)
        brainExtraction.countour_detection(slices_folder)
