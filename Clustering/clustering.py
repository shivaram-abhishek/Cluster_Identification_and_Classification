from PIL import Image
import numpy as np
import os
import shutil
import pandas as pd
from sklearn.cluster import DBSCAN

def slice_images(image_path):
    image = Image.open(image_path)

    if ".png" in image_path:
        file_name = image_path.split("\\")[-1].strip('.png')
    elif ".jpg" in image_path:
        file_name = image_path.split("\\")[-1].strip('.jpg')

    im_array = np.asarray(image)

    im_rows = int(im_array.shape[0]) # number of rows
    im_columns = int(im_array.shape[1]) # number of columns

    # Get the coordinates of the first "R" pixel
    for col in range(im_columns):
        for row in range(im_rows):
            if list(im_array[row][col]) != [0, 0, 0]: # first non-zero pixel
                start_row = row
                start_col = col
                start_pix = im_array[row][col]
                break
        if list(im_array[row][col]) != [0, 0, 0]:
            break

    # Get a list of all starting "R" columns
    r_list_cols = []
    r_prev_col = im_array[0][0]

    for col in range(start_col, im_columns):
        if list(im_array[start_row][col]) == list(start_pix): # if pixel is [255, 255, 255]
            if list(im_array[start_row][col]) != list(r_prev_col): # if previous pixel is [0, 0, 0]
                r_width = 1
                r_list_cols.append(col)
            elif list(im_array[start_row][col]) == list(r_prev_col): # if previous pixel is [255, 255, 255]
                r_width += 1 # width of "R"
        r_prev_col = im_array[start_row][col]

    # Get a list of all starting "R" rows
    r_list_rows = []
    r_prev_row = im_array[0][0]

    for row in range(start_row, im_rows):
        if list(im_array[row][start_col]) == list(start_pix): # if pixel is [255, 255, 255]
            if list(im_array[row][start_col]) != list(r_prev_row):# if previous pixel is [0, 0, 0]
                r_height = 1
                r_list_rows.append(row)
            elif list(im_array[row][start_col]) == list(r_prev_row):# if previous pixel is [255, 255, 255]
                r_height += 1 # height of "R"
        r_prev_row = im_array[row][start_col]

    # Get coordinates of "R"
    r_coor_list = []

    for r in r_list_rows:
        row = []
        for c in r_list_cols:
            if list(im_array[r][c]) == list(start_pix): # if pixel is [255, 255, 255]
                row.append([r, c])
        if len(row) >= 2:
            r_coor_list.append(row)

    width = r_coor_list[0][1][1] - r_coor_list[0][0][1] # width of slice
    height = r_coor_list[1][0][0] - r_coor_list[0][0][0] # height of slice

    image_slices = []

    for row in r_coor_list:
        for coor in row:
            image_slices.append(image.crop((coor[1] + r_width, coor[0] + r_height, coor[1] + width, coor[0] + height))) # image.crop(left, top, right, bottom)

    file_count = 0

    output_folder = os.path.join(os.getcwd(), "Slices", file_name)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.mkdir(output_folder)

    for i in range(len(image_slices)):
        if np.count_nonzero(np.asarray(image_slices[i]) != np.asarray([0, 0, 0])) > 500:
            file_path = os.path.join(output_folder, "img_slice_" + str(file_count) +".png")
            image_slices[i].save(file_path)
            file_count += 1

    return output_folder

# image_path = "C:\\Users\\AbishekShivaram\\OneDrive - Scout Clean Energy\\Desktop\\Personal\\CSE 572\\Assignment 2\\testPatient\\IC_3_thresh.png"

# slice_images(image_path)



def count_clusters(slices_folder): # input is slices folder location created by slice_images()

    # Read the folder and get image slices names

    output_folder = slices_folder.replace("Slices", "Clusters")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.mkdir(output_folder)

    slices_list = os.listdir(slices_folder)

    clusters_list = []

    # For each slice in the folder

    for img_slice in slices_list:
        slice_loc = os.path.join(slices_folder, img_slice)

        if ".png" in slice_loc:
            file_name = img_slice.strip('.png')
        elif ".jpg" in slice_loc:
            file_name = img_slice.strip('.jpg')

        image = Image.open(slice_loc)

        new_image = []
        image_array = []

        # Remove white area of the image slice and make other area yellow

        for item in image.getdata():
            if item[2] >= 135 and item[0] <= 135 and item[1] <= 135:
                new_image.append((255, 255, 0))
                image_array.append(np.array(255))
            elif item[1] >= 135 and item[0] <= 135 and item[2] <= 135:
                new_image.append((255, 255, 0))
                image_array.append(np.array(255))
            elif item[0] >= 135 and item[2] <= 135 and item[1] <= 135:
                new_image.append((255, 255, 0))
                image_array.append(np.array(255))
            else:
                new_image.append((0, 0, 0))
                image_array.append(np.array(0))

        # Put the data in image

        image_array = np.array(image_array)

        image_array = image_array.reshape(113, 114)

        image.putdata(new_image)

        # Find coordinates of the colored points

        coor = []

        for i in range(len(image_array)):
            for j in range(len(image_array[i])):
                if image_array[i][j] == 0:
                    pass
                else:
                    coor.append(np.array([i, j]))

        # Only images with more than 35 colored points in the image will be considered

        if len(coor) >= 35:

            X = np.array(coor)

            # Find clusters using DBSCAN

            db = DBSCAN(eps=5, min_samples=50).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # Number of clusters
            unique_labels = set(labels)

            cluster_image = []
            count = 0

            omit_list = []

            for lbl in list(unique_labels):
                if lbl == -1:
                    pass
                else:
                    if np.count_nonzero(labels == lbl) <= 135:
                        omit_list.append(lbl)
                        # print(lbl, omit_list, unique_labels)
                        n_clusters_ -= 1
            # Put the cluster data in image to remove noise points and color all other points yellow

            for i in range(len(image_array)):
                for j in range(len(image_array[i])):
                    if image_array[i][j] == 0:
                        cluster_image.append((0, 0, 0))
                    else:
                        if labels[count] == -1 or labels[count] in omit_list:
                            cluster_image.append((0, 0, 0))
                        else:
                            cluster_image.append((255, 255, 0))
                        count += 1

            image.putdata(cluster_image)

            # Only consider images with 1 or more clusters and save them

            if n_clusters_ > 0:

                file_path = os.path.join(output_folder, "img_cluster_" + str(file_name) +".png")
                image.save(file_path)

                clusters_list.append(["img_cluster_"+str(file_name), n_clusters_])
    
    # Create CSV file with number of clusters in each image slice

    output_df = pd.DataFrame(clusters_list, columns=['Image name', 'Number of clusters'])

    output_df.loc[len(output_df)] = ['Total', output_df['Number of clusters'].sum()]

    df_path = os.path.join(output_folder, 'Clusters.csv')

    output_df.to_csv(df_path)
