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


def load_sample_data():
    """
    gets the sample data from the workfoler/input_hx_dist/ folder
    :return: protein_name, protein_sequence, timepoints_list, mass_distribution array (2D)
    """
    hx_ms_dist_fpath = '../../workfolder/input_hx_dist/HEEH_rd4_0097_hx_mass_dist.csv'
    timepoints, mass_distribution = load_data_from_hdx_ms_dist_(hx_ms_dist_fpath)
    sample_fpath = '../../workfolder/sample.csv'
    sample_df = pd.read_csv(sample_fpath)
    prot_name = sample_df['name'].values[0]
    prot_seq = sample_df['sequence'].values[0]
    return prot_name, prot_seq, timepoints, mass_distribution


if __name__ == '__main__':

    pass
