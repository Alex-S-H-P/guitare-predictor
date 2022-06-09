"""
A set of useful functions to functions and constants that pertain to this project's files.

Any major modification to the file system should be reflected here.

Author : Alexandre SCHŒPP https://github.com/Alex-S-H-P/
"""

import os

GNRL_PATH_TO_DATA_SET: str = "dataset/"
GNRL_PATH_TO_DATA_SET_SET: bool = False


def get_general_path_to_dataset() -> str:
    """
    Returns the path to the database

    :return: the path to dataset/
    """
    global GNRL_PATH_TO_DATA_SET
    global GNRL_PATH_TO_DATA_SET_SET

    if GNRL_PATH_TO_DATA_SET_SET:
        return GNRL_PATH_TO_DATA_SET

    if (cwd := os.getcwd()).endswith("codebase"):
        GNRL_PATH_TO_DATA_SET = "../dataset/"
    elif not cwd.endswith("predictor"):
        cur_path = __file__  # we get the file
        cur_path = os.path.dirname(cur_path)  # it's parent directory
        cur_path = str(cur_path) + "/../dataset/"
        GNRL_PATH_TO_DATA_SET = cur_path
    GNRL_PATH_TO_DATA_SET_SET = True
    return GNRL_PATH_TO_DATA_SET


def path_to_specific_dataset(dataset_name: str) -> str:
    """
    Returns the path to a specific dataset.
    Does not worry about said dataset existing.
    Any and all dataset should be placed in the dataset/ folder

    :param dataset_name: what you want
    :return: where it is

    """
    return get_general_path_to_dataset() + dataset_name
