"""
This script will be used to separate and copy images coming from
`car_ims.tgz` (extract the .tgz content first) between `train` and `test`
folders according to the column `subset` from `car_dataset_labels.csv`.
It will also create all the needed subfolders inside `train`/`test` in order
to copy each image to the folder corresponding to its class.
"""

import argparse
import os
import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. E.g. "
            "`/home/app/src/data/car_ims/`."
        ),
    )
    parser.add_argument(
        "labels",
        type=str,
        help=(
            "Full path to the CSV file with data labels. E.g. "
            "`/home/app/src/data/car_dataset_labels.csv`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits. E.g. `/home/app/src/data/car_ims_v1/`."
        ),
    )

    args = parser.parse_args()

    return args



def main(data_folder, labels, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to raw images folder.

    labels : str
        Full path to CSV file with data annotations.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        train/test splits.
    """

    df_labels = pd.read_csv(labels)

    #Creating "Output_data_folder" folder and inside of it, train and test folders
    if os.path.exists(output_data_folder) == False:
        os.mkdir(output_data_folder)
    
    path_tr = os.path.join(output_data_folder, "train")
    if os.path.exists(path_tr) == False:
        os.mkdir(path_tr)

    path_te  = os.path.join(output_data_folder, "test")
    if os.path.exists(path_te) == False:
        os.mkdir(path_te)


    #Inside each folder, create an unique folder for each class with its respectives images
    for index, columns in df_labels.iterrows():
        if df_labels.loc[index, "subset"] == "train":
            path_tr_class      = os.path.join(path_tr, columns["class"])
            path_tr_img_source = os.path.join(data_folder, columns["img_name"])
            path_tr_img_output = os.path.join(path_tr_class, columns["img_name"])

            if os.path.exists(path_tr_class) == False:
                os.mkdir(path_tr_class)

            if os.path.exists(path_tr_img_output) == False:
                os.link(path_tr_img_source ,path_tr_img_output)

        elif df_labels.loc[index, "subset"] == "test":
            path_te_class      = os.path.join(path_te, columns["class"])
            path_te_img_source = os.path.join(data_folder, columns["img_name"])
            path_te_img_output = os.path.join(path_te_class, columns["img_name"])

            if os.path.exists(path_te_class) == False:
                os.mkdir(path_te_class)

            if os.path.exists(path_te_img_output) == False:
                os.link(path_te_img_source ,path_te_img_output)            
        else:
            pass
        


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.labels, args.output_data_folder)
