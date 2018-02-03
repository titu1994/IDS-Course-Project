import os


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

