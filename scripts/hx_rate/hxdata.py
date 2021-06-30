import pandas as pd
import numpy as np


def load_data_from_hdx_ms_dist_(fpath):
    """
    get hdx mass distribution data
    :param fpath: input .csv path for distribution
    :return: timepoints, mass_distribution list
    """
    df = pd.read_csv(fpath)
    time_points = np.asarray(df.columns.values[1:], dtype=float)
    mass_distribution = df.iloc[:, 1:].values.T
    return time_points, mass_distribution


if __name__ == '__main__':

    pass
