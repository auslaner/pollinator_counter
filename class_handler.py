import errno

import os

CLASSES = ["Pollinator", "Not_Pollinator", "Unknown"]


def create_classification_folders(class_names, write_path):
    for classification in class_names:
        try:
            os.mkdir(os.path.join(write_path, classification))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
