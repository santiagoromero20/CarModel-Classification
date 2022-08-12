"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
import os
from utils.utils import walkdir
from utils.detection import get_vehicle_coordinates
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    if os.path.exists(output_data_folder) == False:
        os.mkdir(output_data_folder)
    
    path_tr = os.path.join(output_data_folder, "train")
    if os.path.exists(path_tr) == False:
        os.mkdir(path_tr)

    path_te  = os.path.join(output_data_folder, "test")
    if os.path.exists(path_te) == False:
        os.mkdir(path_te)
    
    #1- . Iterate over each image in `data_folder``
    for dirpath, filename in walkdir(data_folder):
        #Load Image
        image_path = os.path.join(dirpath, filename)
        img = cv2.imread(image_path)
        #Crop Image
        x1, y1, x2, y2 = get_vehicle_coordinates(img)
        cropped_img = img[y1:y2, x1:x2]
        #Storing it on new folders
        splitted_path = dirpath.split("/")
        class_name = splitted_path[-1]
        if splitted_path[-2] == "train":
            path_tr_class = os.path.join(path_tr, class_name)
            path_tr_img_output = os.path.join(path_tr_class, filename)
            if os.path.exists(path_tr_class) == False:
                os.mkdir(path_tr_class)
            cv2.imwrite(path_tr_img_output , cropped_img)
        elif splitted_path[-2] == "test":
            path_te_class = os.path.join(path_te, class_name)
            path_te_img_output = os.path.join(path_te_class, filename)
            if os.path.exists(path_te_class) == False:
                os.mkdir(path_te_class)
            cv2.imwrite(path_te_img_output, cropped_img)
        else:
            pass
        


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
