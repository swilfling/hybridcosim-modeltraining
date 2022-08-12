from .trainingdata import TrainingData

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt



def plot_training_results(result: TrainingData, path="./", filename="Result", fig_title: str= "Result", fmt="png"):
    """
    Plot training results
    @param result: TrainingResults structure
    @param path: Output dir
    @param title: filename
    """
    day = 4 * 24
    int = (0 + day * 10, 0 + day * 12)
    print("starting", int[0])
    print(int[1])
    print('Plotting results in Interval:', result.test_index[int[0]], result.test_index[int[1]])
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.style.use('classic')
    ax.plot(result.test_index[int[0]:int[1]], result.test_target[int[0]:int[1]], color='blue', label='true')
    ax.plot(result.test_index[int[0]:int[1]], result.test_prediction[int[0]:int[1]], color='orange', label='prediction')
    ax.grid(linestyle="--", alpha=0.6, color='gray')
    ax.set_xlim([result.test_index[int[0]], result.test_index[int[1]]])
    ax.legend(loc='upper left', frameon=True)
    ax.set_ylabel('Temperature (°C)')
    fig.suptitle(fig_title, y=0.95)
    with open(os.path.join(path, f'{filename}.{fmt}'), "wb") as fp:
        plt.savefig(fp, dpi=300)
    plt.close()

