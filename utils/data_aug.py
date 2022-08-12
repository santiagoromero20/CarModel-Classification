from imp import load_compiled
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def create_data_aug_layer(data_aug_layer):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """

    data_aug_layers = []
    if data_aug_layer == {}:
        pass
    else:
        for key, value in data_aug_layer.items():
            if key == "random_flip":
                l1 = layers.RandomFlip(**data_aug_layer["random_flip"])
                data_aug_layers.append(l1)
            elif key == "random_rotation":
                l2 = layers.RandomRotation(**data_aug_layer["random_rotation"])
                data_aug_layers.append(l2)
            elif key == "random_zoom": 
                l3 = layers.RandomZoom(**data_aug_layer["random_zoom"])
                data_aug_layers.append(l3)
            elif key == "random_translation":
                l4 = layers.RandomTranslation(**data_aug_layer["random_translation"])
                data_aug_layers.append(l4)
            elif key == "random_crop":
                l5 = layers.RandomCrop(**data_aug_layer["random_crop"])
                data_aug_layers.append(l5)
            elif key == "random_contrast":
                l6 = layers.RandomContrast(**data_aug_layer["random_contrast"])
                data_aug_layers.append(l6)
            elif key == "random_width":
                l7 = layers.RandomWidth(**data_aug_layer["random_width"])
                data_aug_layers.append(l7)               


    # Return a keras.Sequential model having the the new layers created
    data_augmentation = tf.keras.Sequential(data_aug_layers)

    return data_augmentation

