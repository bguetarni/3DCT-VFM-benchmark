import os, pathlib
import pydicom

def get_directory_level(path1, path2):
    """
    Return the number of levels that separate each path with the common parent path
    """
    commonpath = os.path.commonpath([path1, path2])
    relpath1 = os.path.relpath(path1, start=commonpath)
    relpath2 = os.path.relpath(path2, start=commonpath)
    return len(pathlib.Path(relpath1).parents), len(pathlib.Path(relpath2).parents)
