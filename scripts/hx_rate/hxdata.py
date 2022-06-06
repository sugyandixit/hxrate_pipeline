import pandas as pd
import numpy as np
import pickle
import os


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


def load_pickle_object(pickle_fpath):
    """
    load pickle object from pickle file
    :param pickle_fpath: pickle filepath
    :return:
    """
    with open(pickle_fpath, 'rb') as pkfile:
        pkobj = pickle.load(pkfile)
    return pkobj


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


def write_hx_rate_output_bayes(hxrate_mean_array,
                               hxrate_median_array,
                               hxrate_std_array,
                               hxrate_5percent_array,
                               hxrate_95percent_array,
                               neff_array,
                               r_hat_array,
                               output_path):

    header = 'ind,rate_mean,rate_median,rate_std,rate_5%,rate_95%,n_eff,r_hat\n'
    data_string = ''

    for num in range(len(hxrate_mean_array)):

        data_string += '{},{},{},{},{},{},{},{}\n'.format(num,
                                                          hxrate_mean_array[num],
                                                          hxrate_median_array[num],
                                                          hxrate_std_array[num],
                                                          hxrate_95percent_array[num],
                                                          hxrate_5percent_array[num],
                                                          neff_array[num],
                                                          r_hat_array[num])

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


def write_backexchange_array(timepoints, backexchange_array, output_path):
    """
    write merged backexchange array
    :param timepoints: timepoints array
    :param backexchange_array: backexchange array
    :param output_path: output path
    :return: Nothing
    """
    header = 'timepoints,backexchange\n'
    data_string = ''
    for tp, backexchange in zip(timepoints, backexchange_array):
        data_string += '{},{}\n'.format(tp, backexchange)

    with open(output_path, 'w') as outfile:
        outfile.write(header + data_string)
        outfile.close()


def write_backexchange_correction_array(timepoints, backexchange_correction_array, output_path):
    data_string = ''
    header = 'time,avg_dm_rate\n'
    for ind, (time, avg_mass_rate) in enumerate(zip(timepoints, backexchange_correction_array)):
        data_string += '{},{}\n'.format(time, avg_mass_rate)

    with open(output_path, 'w') as outfile:
        outfile.write(header + data_string)
        outfile.close()


def load_tp_dependent_dict(filepath):
    """
    load the file that has timepoint in first column and a tp dependent variable on second column and outputs a
    dictionary
    :param filepath: filepath
    :return: dictionary
    """
    df = pd.read_csv(filepath)
    timepoints = df.iloc[:, 0].values
    variable_arr = df.iloc[:, 1].values

    out_dict = dict()

    for ind, (tp, var_value) in enumerate(zip(timepoints, variable_arr)):

        out_dict[tp] = var_value

    return out_dict


def write_merge_factor(merge_factor, opt_mse, opt_nfev, opt_nit, opt_success, opt_message, output_path):

    header = 'factor,mse,opt_nfev,opt_nit,opt_success,opt_message\n'
    data_string = '{},{},{},{},{},{}\n'.format(merge_factor, opt_mse, opt_nfev, opt_nit, opt_success, opt_message)

    with open(output_path, 'w') as outfile:
        outfile.write(header + data_string)
        outfile.close()


def make_new_dir(dirpath):

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    return dirpath


def write_merge_dist_summary(list_of_csv_files, output_fpath, list_of_protein_names=None,
                             file_delim_string='_merge_factor.csv'):

    if list_of_protein_names is None:
        list_of_protein_names = ['PROTEIN' for _ in range(len(list_of_csv_files))]

    with open(output_fpath, 'w') as outfile:

        header = 'fname,prot_name,factor,mse,opt_nfev,opt_nit,opt_success,opt_message\n'
        outfile.write(header)

        for ind, (csv_fpath, prot_name) in enumerate(zip(list_of_csv_files, list_of_protein_names)):

            merge_name = os.path.split(csv_fpath)[-1].strip(file_delim_string)
            df = pd.read_csv(csv_fpath)

            line = '{},{},{},{},{},{},{},{}\n'.format(merge_name,
                                                   prot_name,
                                                   df['factor'].values[0],
                                                   df['mse'].values[0],
                                                   df['opt_nfev'].values[0],
                                                   df['opt_nit'].values[0],
                                                   df['opt_success'].values[0],
                                                   df['opt_message'].values[0])

            outfile.write(line)

        outfile.close()


def write_rate_fit_summary(list_of_ratefit_pk_files, output_fpath,
                           file_delim_string='_hx_rate_fit.pickle'):

    with open(output_fpath, 'w') as outfile:

        header = 'fname,prot_name,sequence,backexchange,backexchange_res_subtract,rate_fit_rmse\n'
        outfile.write(header)

        for ind, pkfpath in enumerate(list_of_ratefit_pk_files):

            pkfname = os.path.split(pkfpath)[-1].strip(file_delim_string)
            pkobj = load_pickle_object(pkfpath)

            line = '{},{},{},{},{},{}\n'.format(pkfname,
                                                pkobj['exp_data']['protein_name'],
                                                pkobj['exp_data']['protein_sequence'],
                                                pkobj['back_exchange']['backexchange_value'],
                                                pkobj['back_exchange_res_subtract'],
                                                pkobj['bayesfit_output']['rmse']['total'])

            outfile.write(line)

        outfile.close()


def write_dg_fit_summary(list_of_dg_pk_files, output_fpath, file_delim_string):

    with open(output_fpath, 'w') as outfile:

        header = 'fname,prot_name,sequence,opt_val,pair_energy,full_burial_corr,hbond_burial_corr,hbond_rank_factor,distance_to_nonpolar_res_corr,distance_to_sec_struct_corr,top_stdev,comp_deltag_rmse\n'

        outfile.write(header)

        for ind, pkfpath in enumerate(list_of_dg_pk_files):

            pkfname = os.path.split(pkfpath)[-1].strip(file_delim_string)
            pkobj = load_pickle_object(pkfpath)

            line = '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(pkfname,
                                                                  pkobj['protein_name'],
                                                                  pkobj['protein_full_sequence'],
                                                                  pkobj['anneal_data']['opt_val'],
                                                                  pkobj['anneal_data']['pair_energy'],
                                                                  pkobj['anneal_data']['full_burial_corr'],
                                                                  pkobj['anneal_data']['hbond_burial_corr'],
                                                                  pkobj['anneal_data']['hbond_rank_factor'],
                                                                  pkobj['anneal_data']['distance_to_nonpolar_res_corr'],
                                                                  pkobj['anneal_data']['distance_to_sec_struct_corr'],
                                                                  pkobj['anneal_data']['top_stdev'],
                                                                  pkobj['anneal_data']['comp_deltaG_rmse_term'])
            outfile.write(line)

        outfile.close()


if __name__ == '__main__':

    pass
