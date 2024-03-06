import os

def get_fileparts(filepath):
    drive, path = os.path.splitdrive(filepath)
    path, filename = os.path.split(path)
    (filename, ext) = filename.split('.')
    path = drive+path
    return (path, filename, ext)