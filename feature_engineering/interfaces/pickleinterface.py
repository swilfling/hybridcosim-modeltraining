import os
import pickle
from . import BasicInterface


class PickleInterface(BasicInterface):
    """
    Interface for storing objects as pickle file.
    """
    @classmethod
    def load_pkl(cls, path: str, filename: str):
        """
            Load from pickle file.
            @param path: directory containing file
            @param filename: filename
            @return: object from file
        """
        with open(os.path.join(path, filename), "rb") as f:
            return pickle.load(f)

    def save_pkl(self, path, filename):
        """
            Save object to pickle file.
            @param path: directory containing file
            @param filename: filename
        """
        with open(os.path.join(path, filename), "wb") as f:
            pickle.dump(self, f)

