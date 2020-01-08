# -*- coding: utf-8 -*-
#
#
# Define the tool that will be used for other program.
#
import os
import shutil
import time
import json
import pickle


def load_pickle(path):
    """load data by pickle."""
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def write_pickle(data, path):
    """dump file to dir."""
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def write_txt(data, path):
    """write string to dir"""
    with open(path, 'w') as handle:
            handle.write(data)
