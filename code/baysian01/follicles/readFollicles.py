"""
reads a csv file from path with columnes 'time', 'counts'.
returns a follicleCounts object with the data from the csv file as a field named 'follicles' (2D numpy array).
"""

import numpy as np
import matplotlib.pyplot as plt


class follicleCounts:
    def __init__(self, follicles, data_times_discretisation = 2):
        self.follicles = follicles
        self.external_hazard = np.inf
        self.data_times_discretisation = data_times_discretisation

    def plotData(self, ax=None, **kwargs):
        """
        Plots the follicle counts data (counts vs time) as a scatter plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.follicles[:,0], self.follicles[:,1], **kwargs)
        ax.set_xlabel('time')
        ax.set_ylabel('counts')
        return ax
    


def folliclesFromFile(path, data_times_discretisation = 2):
    import pandas as pd
    df = pd.read_csv(path)
    return follicleCounts(folliclesFromDataFrame(df), data_times_discretisation)


def folliclesFromDataFrame(df):
    follicles = df[['t', 'counts']].values
    follicles = np.array(follicles)
    follicles = follicles[follicles[:,0]>0,:]
    return follicles


def readSeedFollicles(seed_file):
    import pandas as pd
    df = pd.read_csv(seed_file, index_col=0)
    theta = df.loc['Estimate'][['Lambda0', 'Lambda1', 'Lambda2']].values
    return theta
