# %%

import os
import shutil
import clustering

# dir = ".\\PatientData\\Data"
dir = ".\\testPatient"

files_list = os.listdir(dir)

slices_dir = os.path.join(os.getcwd(), "Slices")
clusters_dir = os.path.join(os.getcwd(), "Clusters")
# %%
if os.path.exists(slices_dir):
    shutil.rmtree(slices_dir)

if os.path.exists(clusters_dir):
    shutil.rmtree(clusters_dir)

os.mkdir(slices_dir)
os.mkdir(clusters_dir)

# %%

for file_name in files_list:
    if "thresh" in file_name:
        img_name = os.path.join(dir, file_name)
        slices_folder = clustering.slice_images(img_name)
        clustering.count_clusters(slices_folder)
        

# %%
