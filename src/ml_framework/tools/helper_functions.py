import os


def get_fileparts(filepath):
    drive, path = os.path.splitdrive(filepath)
    path, filename = os.path.split(path)
    (filename, ext) = filename.split('.')
    path = drive+path
    return (path, filename, ext)


def get_workspace_path():
    path = os.path.dirname(os.path.abspath(__file__))

    this_pyfile_path = os.path.split(os.path.abspath(__file__))[0]
    step_back_path = os.path.split(this_pyfile_path)[0]
    step_back_path = os.path.split(step_back_path)[0]
    step_back_path = os.path.split(step_back_path)[0]

    workspacePath = step_back_path + os.path.sep

    return workspacePath
