import os

GNRL_PATH_TO_DATA_SET: str = "dataset/"
GNRL_PATH_TO_DATA_SET_SET: bool = False


def get_general_path_to_dataset():
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


def path_to_specific_dataset(dataset_name: str):
    return get_general_path_to_dataset() + dataset_name
