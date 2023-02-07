
from PIL import Image
import cv2
import numpy as np
import os
import shutil



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
        if np.count_nonzero(np.asarray(image_slices[i]) != np.asarray([0, 0, 0])) > 100:
            file_path = os.path.join(output_folder, "img_slice_" + str(file_count) +".png")
            image_slices[i].save(file_path)
            file_count += 1

    return output_folder

def countour_detection(slices_folder):

    slices_list = os.listdir(slices_folder)
    output_folder = slices_folder.replace("Slices", "Boundaries")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.mkdir(output_folder)

    for img_slice in slices_list:
        slice_loc = os.path.join(slices_folder, img_slice)

        if ".png" in slice_loc:
            file_name = slice_loc.split("\\")[-1].strip('.png')
        elif ".jpg" in slice_loc:
            file_name = slice_loc.split("\\")[-1].strip('.jpg')

        image = cv2.imread(slice_loc)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        file_path = os.path.join(output_folder, "img_cluster_" + str(file_name) +".png")
        cv2.imwrite(file_path, image)

    return output_folder
