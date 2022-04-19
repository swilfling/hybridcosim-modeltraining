import os
import shutil
from datetime import datetime
from pathlib import Path


def copy_tree_to_dir(src_tree, dst_dir):
    shutil.copytree(src_tree, os.path.join(dst_dir, get_filename(src_tree)), dirs_exist_ok=True)

def get_filename(src_path):
    return list(os.path.split(src_path))[-1]


def copy_files_to_dir(src_paths, dst_dir):
    for src_path in src_paths:
        shutil.copy(src_path, os.path.join(dst_dir, get_filename(src_path)))


def copy_files(src_paths, dst_paths):
    for src_path, dst_path in zip(src_paths,dst_paths):
        shutil.copy(src_path, dst_path)


def create_gitignore(dst_dir):
    create_file(os.path.join(dst_dir, ".gitignore"), "**")


def create_file(file_path_full, lines):
    if lines:
        with open(file_path_full, "w") as f:
            f.writelines(lines)


def copy_tree_contents_to_dir(src_path, dst_path):
    directories = os.scandir(src_path)
    for entry in directories:
        if entry.is_dir():
            print(entry.name)
            copy_tree_to_dir(entry, dst_path)
        if entry.is_file():
            print(entry.name)
            copy_files_to_dir([entry], dst_path)


def get_python_filenames(dir):
    for root, dirs, files in os.walk(dir):
        return [file.split(".")[0] for file in files if file.split('.')[-1] == 'py']


def move_model(file_paths, dst_dir):
    # Remove old files
    shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    # Copy additional sources to resources path
    additional_sources = [os.path.join(file_paths.src_root, source) for source in file_paths.src_files.values()]
    dst_paths = file_paths.src_files.keys()
    for dir_ in [os.path.split(path)[:-1] for path in dst_paths]:
        if os.path.join(*dir_):
            os.makedirs(os.path.join(dst_dir, *dir_), exist_ok=True)
    additional_dsts = [os.path.join(dst_dir, dst) for dst in dst_paths]
    copy_files(additional_sources, additional_dsts)

    # Copy source directories to resources path
    [copy_tree_to_dir(os.path.join(file_paths.src_root, path), dst_dir) for path in file_paths.src_dirs]
    # Copy Pickle model and parameters for each feature
    res_dst = os.path.join(dst_dir, file_paths.resource_dst)
    [ shutil.copytree(os.path.join(file_paths.resource_root, src), os.path.join(res_dst, dst),dirs_exist_ok=True) for src, dst in file_paths.resource_dirs.items()]
    # Create gitignore in dst dir
    create_gitignore(dst_dir)


def copy_file_to_dir(src_path, dst_dir, dst_filename):
    shutil.copy(src_path, os.path.join(dst_dir, dst_filename))


def create_file_name_timestamp():
    return "Experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def create_dir(path):
    os.makedirs(path, exist_ok=True)
    return Path(path)

def create_dir_byname(path, dirname):
    return create_dir(os.path.join(path, dirname))
