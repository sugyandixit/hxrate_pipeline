import pandas as pd
import numpy as np
import hxdata
import matplotlib.pyplot as plt
from hx_rate_fit import calc_back_exchange
from scipy.stats import linregress
import methods


def gen_list_of_backexchange(list_of_dist_files,
                             list_of_seq,
                             d2o_frac,
                             d2o_purity,
                             check_saturation=True):
    """

    :param list_of_dist_files:
    :param list_of_seq:
    :param d2o_frac:
    :param d2o_purity:
    :return:
    """

    backexchange_list = []

    for ind, (fpath, seq) in enumerate(zip(list_of_dist_files, list_of_seq)):

        tp, dist = hxdata.load_data_from_hdx_ms_dist_(fpath=fpath)

        bkexch_obj = calc_back_exchange(sequence=seq,
                                        experimental_isotope_dist=dist[-1],
                                        d2o_fraction=d2o_frac,
                                        d2o_purity=d2o_purity)
        backexchange_list.append(bkexch_obj.backexchange_value)

    return backexchange_list


def plot_backexchange_correlation(low_ph_corr_bkexch,
                                  high_ph_corr_bkexch,
                                  low_ph_nocorr_backexch,
                                  high_ph_nocorr_backexch,
                                  corr_slope,
                                  corr_intercept,
                                  corr_r,
                                  output_path):
    """
    plot backexchange correlation with low and high ph data
    :param low_ph_corr_bkexch:
    :param high_ph_corr_bkexch:
    :param low_ph_saturate_backexch:
    :param high_ph_saturate_backexch:
    :param low_ph_others:
    :param high_ph_others:
    :param corr_slope:
    :param corr_intercept:
    :param corr_r:
    :param output_path:
    :return:
    """

    x_ = np.linspace(start=min(low_ph_corr_bkexch) - 0.05, stop=max(low_ph_corr_bkexch) + 0.05, num=20)
    y_ = corr_slope * x_ + corr_intercept

    title = 'Slope: %.4f | Intercept: %.4f | R: %.4f' % (corr_slope, corr_intercept, corr_r)

    plt.scatter(low_ph_nocorr_backexch, high_ph_nocorr_backexch, color='aquamarine', label='unsatisfied')
    plt.scatter(low_ph_corr_bkexch, high_ph_corr_bkexch, color='darkcyan', label='satisfied')
    plt.plot(x_, y_, ls='--', color='darkcyan')
    plt.xlabel('low ph d2o saturation level')
    plt.ylabel('high ph d2o saturation level')
    plt.grid()
    plt.title(title)
    plt.savefig(output_path)
    plt.close()


def gen_low_high_ph_bkexchange_(merge_sample_list_fpath,
                                low_ph_d2o_frac,
                                low_ph_d2o_pur,
                                high_ph_d2o_frac,
                                high_ph_d2o_pur,
                                bkexchange_bounds,
                                plot_output_path,
                                output_path):
    """

    :param merge_sample_list_fpath:
    :param low_ph_d2o_frac:
    :param low_ph_d2o_pur:
    :param high_ph_d2o_frac:
    :param high_ph_d2o_pur:
    :param output_path:
    :return:
    """

    merge_sample_df = pd.read_csv(merge_sample_list_fpath)

    seq_array = merge_sample_df['sequence'].values

    prot_name_array = merge_sample_df['protein_name'].values

    high_ph_fpath_list = merge_sample_df['high_ph_data_fpath'].values
    low_ph_fpath_list = merge_sample_df['low_ph_data_fpath'].values

    check_saturation_bool_high = [check_saturation_from_file(fpath=x,
                                                             mass_rate_threshold=0.03,
                                                             dist_indices=[-1, -2]) for x in high_ph_fpath_list]
    check_saturation_bool_low = [check_saturation_from_file(fpath=x,
                                                            mass_rate_threshold=0.03,
                                                            dist_indices=[-1, -2]) for x in low_ph_fpath_list]

    check_saturation_bool = []
    for bool_low, bool_high in zip(check_saturation_bool_low, check_saturation_bool_high):
        if bool_low:
            if bool_high:
                check_saturation_bool.append(True)
            else:
                check_saturation_bool.append(False)
        else:
            check_saturation_bool.append(False)

    low_ph_backexchange_arr = np.array(gen_list_of_backexchange(list_of_dist_files=low_ph_fpath_list,
                                                                list_of_seq=seq_array,
                                                                d2o_frac=low_ph_d2o_frac,
                                                                d2o_purity=low_ph_d2o_pur))
    high_ph_backexchange_arr = np.array(gen_list_of_backexchange(list_of_dist_files=high_ph_fpath_list,
                                                                 list_of_seq=seq_array,
                                                                 d2o_frac=high_ph_d2o_frac,
                                                                 d2o_purity=high_ph_d2o_pur))

    print('heho')

    out_dict = dict()
    out_dict['low_ph_backexchange_array'] = low_ph_backexchange_arr
    out_dict['high_ph_backexchange_array'] = high_ph_backexchange_arr

    corr_include_indices = []
    no_corr_indices = []
    for ind, (low_ph_bkexch_, high_ph_bkexch_) in enumerate(zip(low_ph_backexchange_arr, high_ph_backexchange_arr)):
        if check_saturation_bool[ind]:
            if bkexchange_bounds[0] <= low_ph_bkexch_ <= bkexchange_bounds[1]:
                if bkexchange_bounds[0] <= high_ph_bkexch_ <= bkexchange_bounds[1]:
                    corr_include_indices.append(ind)
                else:
                    no_corr_indices.append(ind)
            else:
                no_corr_indices.append(ind)
        else:
            no_corr_indices.append(ind)

    corr_include_indices = np.array(corr_include_indices)
    no_corr_indices = np.array(no_corr_indices)

    low_ph_bkexch_corr = low_ph_backexchange_arr[corr_include_indices]
    high_ph_bkexch_corr = high_ph_backexchange_arr[corr_include_indices]

    low_ph_bkexch_nocorr = low_ph_backexchange_arr[no_corr_indices]
    high_ph_bkexch_nocorr = high_ph_backexchange_arr[no_corr_indices]

    linreg = linregress(x=low_ph_bkexch_corr,
                        y=high_ph_bkexch_corr)

    low_ph_bkexch_linreg = low_ph_backexchange_arr * linreg[0] + linreg[1]

    plot_backexchange_correlation(low_ph_bkexch_corr,
                                  high_ph_bkexch_corr,
                                  low_ph_nocorr_backexch=low_ph_bkexch_nocorr,
                                  high_ph_nocorr_backexch=high_ph_bkexch_nocorr,
                                  corr_slope=linreg[0],
                                  corr_intercept=linreg[1],
                                  corr_r=linreg[2],
                                  output_path=plot_output_path)

    write_low_high_backexchange_array(low_ph_protein_name=merge_sample_df['protein_name_low_ph'].values,
                                      high_ph_protein_name=merge_sample_df['protein_name_high_ph'].values,
                                      low_ph_backexchange_array=low_ph_backexchange_arr,
                                      high_ph_backexchange_array=high_ph_backexchange_arr,
                                      corr_include_indices=corr_include_indices,
                                      low_ph_linreg_bkexchange=low_ph_bkexch_linreg,
                                      output_path=output_path)

    return out_dict


def check_for_exchange_saturation(dist_list, mass_rate_threshold=0.03):
    """

    :param dist_list:
    :param mass_rate_threshold:
    :return:
    """
    gauss_fit_list = methods.gauss_fit_to_isotope_dist_array(isotope_dist=dist_list)
    centroids = [x.centroid for x in gauss_fit_list]
    mass_rate = abs((centroids[1] - centroids[0])/centroids[0])
    if mass_rate > mass_rate_threshold:
        return False
    else:
        return True


def check_saturation_from_file(fpath, mass_rate_threshold=0.03, dist_indices=[-1, -2]):
    """

    :param fpath:
    :param mass_rate_threshold:
    :param dist_indices:
    :return:
    """

    tp, dist_list = hxdata.load_data_from_hdx_ms_dist_(fpath)
    dist_to_check = dist_list[dist_indices]
    check_bool = check_for_exchange_saturation(dist_list=dist_to_check,
                                               mass_rate_threshold=mass_rate_threshold)
    return check_bool


def write_low_high_backexchange_array(low_ph_protein_name,
                                      high_ph_protein_name,
                                      low_ph_backexchange_array,
                                      high_ph_backexchange_array,
                                      corr_include_indices,
                                      low_ph_linreg_bkexchange,
                                      output_path):
    """

    :param low_ph_protein_name:
    :param high_ph_protein_name:
    :param low_ph_backexchange_array:
    :param high_ph_backexchange_array:
    :param corr_include_indices:
    :param output_path:
    :return:
    """

    corr_include_arr = np.zeros(len(low_ph_backexchange_array))
    corr_include_arr[corr_include_indices] = 1

    header = 'low_ph_protein_name,high_ph_protein_name,low_ph_backexchange,high_ph_backexchange,corr_include,low_ph_backexchange_linreg\n'
    data_string = ''

    for ind, (low_prot_name, high_prot_name, low_backexchange, high_backexchange, corr_include_num, low_ph_linreg) in enumerate(zip(low_ph_protein_name,
                                                                                                   high_ph_protein_name,
                                                                                                   low_ph_backexchange_array,
                                                                                                   high_ph_backexchange_array,
                                                                                                                     corr_include_arr,
                                                                                                                                    low_ph_linreg_bkexchange)):

        data_string += '{},{},{},{},{},{}\n'.format(low_prot_name, high_prot_name, low_backexchange, high_backexchange, corr_include_num, low_ph_linreg)

    with open(output_path, 'w') as outfile:
        outfile.write(header + data_string)
        outfile.close()


if __name__ == '__main__':

    merge_sample_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/merged_data/merge_sample_list.csv'

    output_dir = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/merged_data'

    d2o_frac = 0.95
    d2o_pur = 0.95

    gen_low_high_ph_bkexchange_(merge_sample_list_fpath=merge_sample_fpath,
                                low_ph_d2o_frac=d2o_frac,
                                low_ph_d2o_pur=d2o_pur,
                                high_ph_d2o_frac=d2o_frac,
                                high_ph_d2o_pur=d2o_pur,
                                bkexchange_bounds=[0.15, 0.30],
                                output_path=output_dir+'/merge_backexchange_v3.csv',
                                plot_output_path=output_dir+'/merge_backexchange_v3.pdf')
