import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy import interpolate
import matplotlib.pyplot as plt
import methods
import hx_rate_fit
import hxdata


def gen_mse_with_factor_shift(factor: float,
                              low_tp: np.ndarray,
                              low_centroids: np.ndarray,
                              high_tp: np.ndarray,
                              high_centroids: np.ndarray) -> float:
    """
    gen comparison arrays when the high tp is shifted by some factor
    :param factor: factor by which to shift the high timepoints
    :param low_tp: low ph timepoints not including zero
    :param low_centroids: low ph centroinds not including for zero
    :param high_tp: high ph timepoints not including zero
    :param high_centroids: high ph centroids not including for zero
    :return: mse
    """

    # shift the tp by some factor
    high_tp_refac = high_tp * factor

    # convert the time to log space
    log_low_tp = np.log(low_tp)
    log_high_tp = np.log(high_tp_refac)

    # generate index where to slice the timepoint for comparison that is common on both high and low data
    tp_diff = np.subtract(log_low_tp[-1], log_high_tp)
    high_ind = 0
    for ind, diff in enumerate(tp_diff):
        if diff < 0:
            high_ind = ind
            break

    # make sure high end is not zero. That means it didn't find any common timepoints for comparison. in that case
    # return the full centroids for comparison
    if high_ind == 0:
        print('no common timepoints for comparison. Returning the entire array as it is for computing mse.')
        high_centroids_comp = high_centroids
        low_centroids_comp = low_centroids
    else:
        # generate a new time axis that's common for both high and low ph data
        new_axis = log_high_tp[:high_ind]
        high_centroids_comp = high_centroids[:high_ind]

        # create an interpolation function for low data
        low_interp_func = interpolate.interp1d(x=log_low_tp, y=low_centroids)

        # generate interpolated centroids on the new axis
        low_centroids_comp = low_interp_func(x=new_axis)

    # computing mse
    try:
        mse = mean_squared_error(y_true=high_centroids_comp, y_pred=low_centroids_comp)
    except ValueError:
        mse = 100

    return mse


def optimize_factor_for_alignment(low_tp: np.ndarray,
                                  low_centroids: np.ndarray,
                                  high_tp: np.ndarray,
                                  high_centroids: np.ndarray,
                                  opt_method: str = 'Powell',
                                  opt_init_guess: float = 10.0) -> dict:
    """

    :param low_tp: non zero low ph timepoints
    :param low_centroids: non zero low ph centroids
    :param high_tp: non zero high ph timepoints
    :param high_centroids: non zero high ph centroids
    :param opt_method: optimize method
    :param opt_init_guess: initial guess for the factor
    :return: optimize dictionary
    """

    # optimize for factor that best aligns the centroids from low to high ph data

    opt = minimize(fun=gen_mse_with_factor_shift,
                   x0=np.array([opt_init_guess]),
                   args=(low_tp, low_centroids, high_tp, high_centroids),
                   method=opt_method)

    # store important info in a dictionary for output
    out_dict = dict()
    out_dict['opt_method'] = opt_method
    out_dict['opt_init_guess'] = opt_init_guess
    out_dict['opt_mse'] = opt.fun
    out_dict['opt_nfev'] = opt.nfev
    out_dict['opt_nit'] = opt.nit
    out_dict['opt_success'] = opt.success
    out_dict['opt_message'] = opt.message
    out_dict['opt_x'] = opt.x[0]

    return out_dict


def generate_backexchange_array(exp_dist: np.ndarray,
                                timepoints: np.ndarray,
                                sequence: str,
                                d2o_fraction: float,
                                d2o_purity: float,
                                user_backexchange: float,
                                backexchange_correction_arr: np.ndarray) -> np.ndarray:
    """
    generate backexchange array by calculating backexchange and using correction rates
    :param exp_dist: experimental distribution used to calculate the backexchange
    :param timepoints: timepoints array
    :param sequence: protein sequence
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param user_backexchange: user backexchange
    :param backexchange_correction_arr: correction arr
    :return: backexchange array
    """

    # generate backexchange object. Either calculate backexchange using the exp distribution or set a user backexchange
    backexchange = hx_rate_fit.calc_back_exchange(sequence=sequence,
                                                  experimental_isotope_dist=exp_dist,
                                                  d2o_fraction=d2o_fraction,
                                                  d2o_purity=d2o_purity,
                                                  usr_backexchange=user_backexchange)

    # generate backexchange array based on correction rates. If no correction rate, use the same backexchange for all timepoints
    if backexchange_correction_arr is None:
        backexchange_array = np.array([backexchange.backexchange_value for x in timepoints])
    else:

        backexchange_array = methods.gen_corr_backexchange(mass_rate_array=backexchange_correction_arr,
                                                       fix_backexchange_value=backexchange.backexchange_value)

    return backexchange_array


def gen_merged_dstirbution(sequence: str,
                           low_tp: np.ndarray,
                           low_dist: np.ndarray,
                           low_d2o_frac: float,
                           low_d2o_purity: float,
                           low_user_backexchange: float,
                           low_backexchange_correction_arr: np.ndarray,
                           high_tp: np.ndarray,
                           high_dist: np.ndarray,
                           high_d2o_frac: float,
                           high_d2o_purity: float,
                           high_user_backexchange: float,
                           high_backexchange_correction_arr: np.ndarray) -> dict:
    """

    :param sequence:
    :param low_tp:
    :param low_dist:
    :param low_d2o_frac:
    :param low_d2o_purity:
    :param low_user_backexchange:
    :param low_backexchange_correction_arr:
    :param high_tp:
    :param high_dist:
    :param high_d2o_frac:
    :param high_d2o_purity:
    :param high_user_backexchange:
    :param high_backexchange_correction_arr:
    :return:
    """

    # fit gaussians to both distributions
    low_gauss_fit_list = methods.gauss_fit_to_isotope_dist_array(low_dist)
    high_gauss_fit_list = methods.gauss_fit_to_isotope_dist_array(high_dist)

    final_dist = low_dist[-1]

    # get backexchange array
    low_backexchange_array = generate_backexchange_array(exp_dist=low_dist[-1],
                                                         timepoints=low_tp,
                                                         sequence=sequence,
                                                         d2o_fraction=low_d2o_frac,
                                                         d2o_purity=low_d2o_purity,
                                                         user_backexchange=low_user_backexchange,
                                                         backexchange_correction_arr=low_backexchange_correction_arr)
    high_backexchange_array = generate_backexchange_array(exp_dist=high_dist[-1],
                                                          timepoints=high_tp,
                                                          sequence=sequence,
                                                          d2o_fraction=high_d2o_frac,
                                                          d2o_purity=high_d2o_purity,
                                                          user_backexchange=high_user_backexchange,
                                                          backexchange_correction_arr=high_backexchange_correction_arr)

    # generate centroids array for both distributions
    low_centroids = np.array([x.centroid for x in low_gauss_fit_list])
    high_centroids = np.array([x.centroid for x in high_gauss_fit_list])

    # generate centroids with backexhcnage corrections
    low_centroids_corr = methods.correct_centroids_using_backexchange(centroids=low_centroids,
                                                                      backexchange_array=low_backexchange_array)
    high_centroids_corr = methods.correct_centroids_using_backexchange(centroids=high_centroids,
                                                                       backexchange_array=high_backexchange_array)

    # get the factor by using the optimization function. exclude zero timepoints
    opt_ = optimize_factor_for_alignment(low_tp=low_tp[1:],
                                         low_centroids=low_centroids_corr[1:],
                                         high_tp=high_tp[1:],
                                         high_centroids=high_centroids_corr[1:])

    # generate the high ph timepoints with factor applied
    high_tp_factor = high_tp * opt_['opt_x']

    # merge the low and high ph timepoints together
    merge_tp = np.concatenate([low_tp, high_tp_factor[1:]])  # don't inlcude the zero timepoint from high ph data

    # generate a sorting key
    tp_sort_index = np.argsort(merge_tp)

    # generate a sorted timepoint array
    sorted_merged_tp = merge_tp[tp_sort_index]

    # generate a sorted merged distributions
    merge_dist = np.concatenate([low_dist, high_dist[1:, :]], axis=0)
    sorted_merged_dist = merge_dist[tp_sort_index, :]

    # generate a sorted merged backexchange array
    merged_backexchange = np.concatenate([low_backexchange_array, high_backexchange_array[1:]])
    sorted_merged_backexchange = merged_backexchange[tp_sort_index]
    merged_backexchange_correction_array = methods.gen_backexchange_correction_from_backexchange_array(backexchange_array=sorted_merged_backexchange)

    out_dict = dict()
    out_dict['low_tp'] = low_tp
    out_dict['low_centroids'] = low_centroids
    out_dict['low_centroids_corr'] = low_centroids_corr
    out_dict['high_tp'] = high_tp
    out_dict['high_tp_factor'] = high_tp_factor
    out_dict['high_centroids'] = high_centroids
    out_dict['high_centroids_corr'] = high_centroids_corr
    out_dict['factor_optimization'] = opt_
    out_dict['merged_timepoints'] = sorted_merged_tp
    out_dict['merged_dist'] = sorted_merged_dist
    out_dict['merged_backexchange'] = sorted_merged_backexchange
    out_dict['merged_backexchange_correction_array'] = merged_backexchange_correction_array

    return out_dict


def get_backexchange_corr_arr(bakcexchange_corr_fpath):
    """
    given backexchange correction filepath
    :param bakcexchange_corr_fpath: backexchange correction filepath
    :return: backexchange correction
    """
    df = pd.read_csv(bakcexchange_corr_fpath)
    backexchange_corr_arr = df.iloc[:, 1].values
    return backexchange_corr_arr


def plot_merge_centroids(low_tp, low_centroids, high_tp_refactored, high_centroids, factor, mse, output_path):
    """

    :param low_tp:
    :param low_centroids:
    :param high_tp_refactored:
    :param high_centroids:
    :param output_path:
    :return:
    """
    plt.scatter(np.log(low_tp), low_centroids, marker='o', color='blue', edgecolor='black', label='low_ph', s=60)
    plt.scatter(np.log(high_tp_refactored), high_centroids, marker='o', color='orange', edgecolor='black',
                label='high_ph', s=60)
    plt.xlabel('log time (seconds)')
    plt.ylabel('Center of mass')
    plt.grid()
    plt.legend(loc='best')
    plt.title('Merge low and high ph data (Factor: %.4f | MSE: %.4f)' % (factor, mse))
    plt.savefig(output_path)
    plt.close()


def gen_high_low_merged_from_to_file(sequence: str,
                                     low_ph_data_fpath: str,
                                     low_d2o_frac: float,
                                     low_d2o_purity: float,
                                     low_user_backexchange: float or None,
                                     low_backexchange_corr_fpath: str,
                                     high_ph_data_fpath: str,
                                     high_d2o_frac: float,
                                     high_d2o_purity: float,
                                     high_user_backexchange: float or None,
                                     high_backexchange_corr_fpath: str,
                                     merged_backexchange_fpath: str or None,
                                     merged_data_fpath: str or None,
                                     merged_backexchange_correction_fpath: str or None,
                                     factor_fpath: str or None,
                                     merge_plot_fpath: str or None,
                                     return_flag: bool = False):

    # read the ph data
    low_tp, low_dists = hxdata.load_data_from_hdx_ms_dist_(low_ph_data_fpath)
    high_tp, high_dists = hxdata.load_data_from_hdx_ms_dist_(high_ph_data_fpath)

    # get the backexchange correction array
    if low_backexchange_corr_fpath is not None:
        low_backexchange_corr_arr = get_backexchange_corr_arr(low_backexchange_corr_fpath)
    else:
        low_backexchange_corr_arr = None

    if high_backexchange_corr_fpath is not None:
        high_backexchange_corr_arr = get_backexchange_corr_arr(high_backexchange_corr_fpath)
    else:
        high_backexchange_corr_arr = None

    merged_data_dict = gen_merged_dstirbution(sequence=sequence,
                                              low_tp=low_tp,
                                              low_dist=low_dists,
                                              low_d2o_frac=low_d2o_frac,
                                              low_d2o_purity=low_d2o_purity,
                                              low_user_backexchange=low_user_backexchange,
                                              low_backexchange_correction_arr=low_backexchange_corr_arr,
                                              high_tp=high_tp,
                                              high_dist=high_dists,
                                              high_d2o_frac=high_d2o_frac,
                                              high_d2o_purity=high_d2o_purity,
                                              high_user_backexchange=high_user_backexchange,
                                              high_backexchange_correction_arr=high_backexchange_corr_arr)

    # write the merged timepoints and isotope distribution
    if merged_data_fpath is not None:
        hxdata.write_isotope_dist_timepoints(timepoints=merged_data_dict['merged_timepoints'],
                                             isotope_dist_array=merged_data_dict['merged_dist'],
                                             output_path=merged_data_fpath)

    # write the merged backexchange
    if merged_backexchange_fpath is not None:
        hxdata.write_backexchange_array(timepoints=merged_data_dict['merged_timepoints'],
                                        backexchange_array=merged_data_dict['merged_backexchange'],
                                        output_path=merged_backexchange_fpath)

    # write the merged backexchange correction
    if merged_backexchange_correction_fpath is not None:
        hxdata.write_backexchange_correction_array(timepoints=merged_data_dict['merged_timepoints'],
                                                   backexchange_correction_array=merged_data_dict['merged_backexchange_correction_array'],
                                                   output_path=merged_backexchange_correction_fpath)

    # write factor and optimization info to a file
    if factor_fpath is not None:
        hxdata.write_merge_factor(merge_factor=merged_data_dict['factor_optimization']['opt_x'],
                                  opt_mse=merged_data_dict['factor_optimization']['opt_mse'],
                                  opt_nfev=merged_data_dict['factor_optimization']['opt_nfev'],
                                  opt_nit=merged_data_dict['factor_optimization']['opt_nit'],
                                  opt_success=merged_data_dict['factor_optimization']['opt_success'],
                                  opt_message=merged_data_dict['factor_optimization']['opt_message'],
                                  output_path=factor_fpath)

    if merge_plot_fpath is not None:
        plot_merge_centroids(low_tp=merged_data_dict['low_tp'][1:],
                             low_centroids=merged_data_dict['low_centroids_corr'][1:],
                             high_tp_refactored=merged_data_dict['high_tp_factor'][1:],
                             high_centroids=merged_data_dict['high_centroids_corr'][1:],
                             factor=merged_data_dict['factor_optimization']['opt_x'],
                             mse=merged_data_dict['factor_optimization']['opt_mse'],
                             output_path=merge_plot_fpath)

    if return_flag:
        return merged_data_dict


def gen_parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-seq', '--sequence', help='protein sequence one letter code')
    parser.add_argument('-ldata', '--lowphdata', help='low ph data filepath')
    parser.add_argument('-ldf', '--lowdfrac', help='low d2o fraction')
    parser.add_argument('-ldp', '--lowdpur', help='low d2o purity')
    parser.add_argument('-lbk', '--lowbkex', help='low ph backexchange user value')
    parser.add_argument('-lbkc', '--lowbkexcorr', help='low ph backexchange correction filepath')
    parser.add_argument('-hdata', '--highphdata', help='high ph data filepath')
    parser.add_argument('-hdf', '--highdfrac', help='high d2o fraction')
    parser.add_argument('-hdp', '--highdpur', help='high d2o purity')
    parser.add_argument('-hbk', '--highbkex', help='high ph backexchange user value')
    parser.add_argument('-hbkc', '--highbkexcorr', help='high ph backexchange correction filepath')
    parser.add_argument('-mbk', '--mergebkex', help='merge backexchange output filepath .csv')
    parser.add_argument('-mbkc', '--mergebkexcorr', help='merge backexchange correction output filepath .csv')
    parser.add_argument('-mdp', '--mergedatapath', help='merge distribution output filepath .csv')
    parser.add_argument('-mpp', '--mergeplotpath', help='merge plot output filepath .pdf')
    parser.add_argument('-mfp', '--mergefactorpath', help='merge factor output filepath .csv')

    return parser


def run_from_parser():

    parser_ = gen_parse_args()
    options = parser_.parse_args()

    gen_high_low_merged_from_to_file(sequence=options.sequence,
                                     low_ph_data_fpath=options.ldata,
                                     low_d2o_frac=float(options.ldf),
                                     low_d2o_purity=float(options.ldp),
                                     low_user_backexchange=float(options.lbk),
                                     low_backexchange_corr_fpath=options.lbkc,
                                     high_ph_data_fpath=options.hdata,
                                     high_d2o_frac=float(options.hdf),
                                     high_d2o_purity=float(options.hdp),
                                     high_user_backexchange=float(options.hbk),
                                     high_backexchange_corr_fpath=options.hbkc,
                                     merged_backexchange_fpath=options.mbk,
                                     merged_backexchange_correction_fpath=options.mbkc,
                                     merged_data_fpath=options.mdp,
                                     factor_fpath=options.mfp,
                                     merge_plot_fpath=options.mpp,
                                     return_flag=False)


if __name__ == '__main__':

    run_from_parser()

    # common_backexchange_filepath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bkexch_corr_output/common_2.csv_bkexchange.csv'
    # common_backexchange_df = pd.read_csv(common_backexchange_filepath)
    # # common_backexchange_df['comp_bool'] = np.array([1 for x in range(len(common_backexchange_df['sequence'].values))])
    #
    # low_backexchange_corr_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/lib15_ph6_sample.csv_backexchange_correction.csv'
    # high_backexchange_corr_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/lib15_ph7_sample.csv_backexchange_correction.csv'
    #
    # comp_df = common_backexchange_df[common_backexchange_df['comp_bool'] == 1]
    #
    # low_fpath_dir = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/lib15_ph6'
    # high_fpath_dir = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/lib15_ph7'
    #
    # d2o_frac = 0.95
    # d2o_pur = 0.95
    #
    # output_top_dir = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/merged_data_ph6_ph7'
    #
    # import glob
    # import os
    #
    # for ind, (low_name, high_name, seq) in enumerate(zip(comp_df['ph6'].values, comp_df['ph7'].values, comp_df['sequence'].values)):
    #
    #     low_phdata_fpath = glob.glob(low_fpath_dir + '/' + low_name + '*.csv')[0]
    #     high_phdata_fpath = glob.glob(high_fpath_dir + '/' + high_name + '*.csv')[0]
    #
    #     output_dir = hxdata.make_new_dir(output_top_dir + '/' + low_name + '_' + high_name)
    #     merged_data_path = os.path.join(output_dir, low_name + '_' + high_name + '_merged_data.csv')
    #     merged_backexchange_path = os.path.join(output_dir, low_name + '_' + high_name + '_merged_backexchange.csv')
    #     merged_backexchange_corr_path = os.path.join(output_dir, low_name + '_' + high_name + '_merged_backexchange_correction.csv')
    #     factor_path = os.path.join(output_dir, low_name + '_' + high_name + '_factor.csv')
    #     factor_plot_path = os.path.join(output_dir, low_name + '_' + high_name + '_factor_plot.pdf')
    #
    #     gen_high_low_merged_from_to_file(sequence=seq,
    #                                      low_ph_data_fpath=low_phdata_fpath,
    #                                      low_d2o_frac=d2o_frac,
    #                                      low_d2o_purity=d2o_pur,
    #                                      low_user_backexchange=None,
    #                                      low_backexchange_corr_fpath=low_backexchange_corr_fpath,
    #                                      high_ph_data_fpath=high_phdata_fpath,
    #                                      high_d2o_frac=d2o_frac,
    #                                      high_d2o_purity=d2o_pur,
    #                                      high_user_backexchange=None,
    #                                      high_backexchange_corr_fpath=high_backexchange_corr_fpath,
    #                                      merged_backexchange_fpath=merged_backexchange_path,
    #                                      merged_backexchange_correction_fpath=merged_backexchange_corr_path,
    #                                      merged_data_fpath=merged_data_path,
    #                                      factor_fpath=factor_path,
    #                                      merge_plot_fpath=factor_plot_path)
