import pandas as pd
import numpy as np
import pickle


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
    sample_fpath = '../../workfolder/sample.csv'
    sample_df = pd.read_csv(sample_fpath)
    prot_name = sample_df['name'].values[0]
    prot_seq = sample_df['sequence'].values[0]
    hx_ms_dist_fpath = sample_df['hx_dist_fpath'].values[0]
    timepoints, mass_distribution = load_data_from_hdx_ms_dist_(hx_ms_dist_fpath)
    return prot_name, prot_seq, timepoints, mass_distribution


def write_pickle_object(obj, filepath):
    """
    write an object to a pickle file
    :param obj: object
    :param filepath: pickle file path
    :return: None
    """
    with open(filepath, 'wb') as outfile:
        pickle.dump(obj, outfile)


def write_hx_rate_output(hx_rates, output_path):
    """

    :param hx_rates:
    :param output_path:
    :return: None
    """

    header = 'ind,hx_rate\n'
    data_string = ''

    for ind, hx_rate in enumerate(hx_rates):
        data_string += '{},{}\n'.format(ind, hx_rate)

    with open(output_path, 'w') as outfile:
        outfile.write(header + data_string)
        outfile.close()


def write_isotope_dist_timepoints(timepoints, isotope_dist_array, output_path):

    timepoint_str = ','.join([str(x) for x in timepoints])
    header = 'ind,' + timepoint_str + '\n'
    data_string = ''
    for ind, arr in enumerate(isotope_dist_array.T):
        arr_str = ','.join([str(x) for x in arr])
        data_string += '{},{}\n'.format(ind, arr_str)

    with open(output_path, 'w') as outfile:
        outfile.write(header + data_string)
        outfile.close()


if __name__ == '__main__':

    pass
