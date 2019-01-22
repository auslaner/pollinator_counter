"""Manages variables based on the OS the script is running on."""
import os
import sys


def get_system_paths():
    system_paths = {}
    if sys.platform == "darwin":
        # OS X
        system_paths['home'] = os.environ["HOME"]
    elif sys.platform == "win32":
        # Windows
        system_paths['home'] = os.environ["HOMEDRIVE"] + os.environ["HOMEPATH"]

    return system_paths
