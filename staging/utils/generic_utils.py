import os
from pathlib import Path


def resolve_data_path(path):
    path1 = "../data/" + path
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path1[1:]):
        return path1[1:]

    path2 = "staging/data/" + path
    if os.path.exists(path2):
        return path2
    else:
        raise ValueError("File not found ! Seached %s and %s" % (path1, path2))


def construct_data_path(filename):
    if 'staging' in os.getcwd():
        path = '../data/' + filename
    else:
        path = '/staging/data/' + filename

    directory = os.path.split(path)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    return path
