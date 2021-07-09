# hx_rate_fitting code
# authors: Gabe Rocklin and Suggie

import multiprocessing as mp
import numpy as np
from dataclasses import dataclass
from scipy.special import expit
from scipy.optimize import fmin_powell
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from methods import isotope_dist_from_PoiBin, gen_temp_rates, gen_theoretical_isotope_dist_for_all_timepoints, \
    normalize_mass_distribution_array, hx_rate_fitting_optimization, compute_rmse_exp_thr_iso_dist
from hxdata import load_data_from_hdx_ms_dist_, write_pickle_object, write_hx_rate_output, write_isotope_dist_timepoints
import time


@dataclass
class HXRate(object):
    """
    class container to store hx rate fitting data
    """
    back_exchange: object = None
    optimization_cost: float = None
    optimization_func_evals: int = None
    optimization_init_rate_guess: np.ndarray = None
    hx_rates: np.ndarray = None
    thr_isotope_dist_array: np.ndarray = None
    fit_rmse_each_timepoint: np.ndarray = None
    total_fit_rmse: float = None


@dataclass
class BackExchange(object):
    """
    class container to store backexchange data
    """
    backexchange_value: float = None
    fit_rmse: float = None
    theoretical_isotope_dist: np.ndarray = None
    backexchange_array: np.ndarray = None


def plot_exp_thr_dist(exp_dist_array: np.ndarray,
                      thr_dist_array: np.ndarray,
                      timepoints_array: np.ndarray,
                      output_path: str,
                      rmse_each_timepoint: np.ndarray = None,
                      total_rmse: float = None):
    """
    plot the experimental and theoretical isotope distribution at each time point
    :param exp_dist_array: exp dist array
    :param thr_dist_array: thr dist array
    :param timepoints_array: timepoints array
    :param rmse_each_timepoint: rmse at each timepoint. if None, will compute
    :param total_rmse: total rmse. if None, will compute
    :param output_path: output path
    :return:
    """

    num_columns = 3
    num_rows = 0

    for num in range(len(exp_dist_array)):
        if num % num_columns == 0:
            num_rows += 1

    fig = plt.figure(figsize=(15, num_rows * 2.0))
    gs = gridspec.GridSpec(ncols=num_columns, nrows=num_rows, figure=fig)

    n_rows = 0
    n_cols = 0

    for num, (timepoint, exp_dist, thr_dist) in enumerate(zip(timepoints_array, exp_dist_array, thr_dist_array)):

        if rmse_each_timepoint is None:
            rmse_tp = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=exp_dist,
                                                    thr_isotope_dist=thr_dist,
                                                    squared=False)
        else:
            rmse_tp = rmse_each_timepoint[num]

        ax = fig.add_subplot(gs[n_rows, n_cols])
        plt.plot(exp_dist, color='blue', label='exp')
        plt.plot(thr_dist, color='red', label='thr (%.4f)' % rmse_tp)
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.xticks(range(0, len(exp_dist) + 5, 5))
        ax.set_xticklabels(range(0, len(exp_dist) + 5, 5), fontsize=8)
        plt.grid(axis='x', alpha=0.25)
        ax.tick_params(length=3, pad=3)
        plt.legend(loc='best', fontsize='small')
        plt.title('timepoint %i' % timepoint)

        if (n_cols+1) % num_columns == 0:
            n_rows += 1
            n_cols = 0
        else:
            n_cols += 1

    if total_rmse is None:
        exp_isotope_dist_concat = np.concatenate(exp_dist_array)
        thr_isotope_dist_concat = np.concatenate(thr_dist_array)
        thr_isotope_dist_concat[np.isnan(thr_isotope_dist_concat)] = 0
        total_rmse = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=exp_isotope_dist_concat,
                                                   thr_isotope_dist=thr_isotope_dist_concat,
                                                   squared=False)

    plot_title = 'EXP vs THR Isotope Distribution (Fit RMSE: %.4f)' % total_rmse
    plt.suptitle(plot_title)
    plt.subplots_adjust(hspace=0.5, wspace=0.1, top=0.95)
    plt.savefig(output_path)
    plt.close()


def plot_hx_rates(hx_rates: np.ndarray,
                  output_path: str):
    """
    plot the hx rates in increasing order
    :param hx_rates:
    :param output_path:
    :return:
    """
    fig, ax = plt.subplots()
    plt.plot(np.arange(len(hx_rates)), np.sort(hx_rates), marker='o', ls='-', color='black', markerfacecolor='red')
    plt.xticks(range(0, len(hx_rates) + 2, 2))
    ax.set_xticklabels(range(0, len(hx_rates) + 2, 2), fontsize=8)
    plt.grid(axis='x', alpha=0.25)
    plt.xlabel('Residues (Ranked from most to least protected)')
    plt.ylabel('Rate')
    plt.subplots_adjust(hspace=0.5, wspace=0.1, top=0.95)
    plt.savefig(output_path)
    plt.close()



def calc_back_exchange(sequence: str,
                       experimental_isotope_dist: np.ndarray,
                       d2o_fraction: float,
                       d2o_purity: float,
                       timepoint: float = 1e9) -> object:
    """
    calculate back exchange from the experimental isotope distribution
    :param sequence: protein sequence
    :param experimental_isotope_dist: experimental isotope distribution to be used for backexchange calculation
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param timepoint: hdx timepoint in seconds. default to 1e9 for full deuteration
    :return: backexchange data object
    """

    # set high rates for calculating back exchange
    rates = gen_temp_rates(sequence=sequence,
                           rate_value=1e2)

    num_bins = len(experimental_isotope_dist)

    opt = fmin_powell(lambda x: compute_rmse_exp_thr_iso_dist(exp_isotope_dist=experimental_isotope_dist,
                                                              thr_isotope_dist=isotope_dist_from_PoiBin(sequence=sequence,
                                                                            timepoint=timepoint,
                                                                            inv_backexchange=expit(x),
                                                                            rates=rates,
                                                                            d2o_fraction=d2o_fraction,
                                                                            d2o_purity=d2o_purity,
                                                                            num_bins=num_bins),
                                                              squared=False),
                      x0=2,
                      disp=True)

    back_exchange = 1 - expit(opt)[0]

    thr_iso_dist_full_deut = isotope_dist_from_PoiBin(sequence=sequence,
                                                      timepoint=timepoint,
                                                      inv_backexchange=1 - back_exchange,
                                                      rates=rates,
                                                      d2o_fraction=d2o_fraction,
                                                      d2o_purity=d2o_purity,
                                                      num_bins=num_bins)

    fit_rmse = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=experimental_isotope_dist,
                                             thr_isotope_dist=thr_iso_dist_full_deut,
                                             squared=False)

    backexchange_obj = BackExchange(backexchange_value=back_exchange,
                                    fit_rmse=fit_rmse,
                                    theoretical_isotope_dist=thr_iso_dist_full_deut)

    return backexchange_obj


def fit_rate(sequence: str,
             time_points: np.ndarray,
             norm_mass_distribution_array: np.ndarray,
             d2o_fraction: float,
             d2o_purity: float,
             opt_iter: int = 500,
             opt_temp: float = 0.00003,
             opt_step_size: float = 0.02,
             multi_proc: bool = True,
             number_of_cores: int = 6,
             rate_init_list: list = None,
             free_energy_values: np.ndarray = None,
             temperature: float = None,
             backexchange_value: float = None) -> object:

    # initialize hxrate data object
    hxrate = HXRate()

    # calculate back exchange first
    back_exchange = calc_back_exchange(sequence=sequence,
                                       experimental_isotope_dist=norm_mass_distribution_array[-1],
                                       d2o_fraction=d2o_fraction,
                                       d2o_purity=d2o_purity)

    # to input backexchange value manually
    if backexchange_value is not None:
        back_exchange.backexchange_value = backexchange_value

    # generate an array of backexchange with same backexchange value to an array of length of timepoints
    back_exchange.backexchange_array = np.array([back_exchange.backexchange_value for x in time_points])

    # store backexchange object in hxrate object
    hxrate.back_exchange = back_exchange

    inv_back_exchange_array = np.subtract(1, back_exchange.backexchange_array)

    # generate a temporary rates to determine what residues can't exchange
    temp_rates = gen_temp_rates(sequence=sequence, rate_value=1)

    # get the indices of residues that don't exchange
    zero_indices = np.where(temp_rates == 0)[0]

    # calculate the number of residues that can exchange
    num_rates = len(temp_rates) - len(zero_indices)

    num_bins_ = len(norm_mass_distribution_array[0])

    # get init rate list as initial guesses for rate fit optimization
    if rate_init_list is not None:
        init_rates_list = rate_init_list
    else:
        init_rates_list = [np.array([-2 for x in range(num_rates)]),
                           np.array([-5 for x in range(num_rates)]),
                           np.linspace(-7, 0, num_rates),
                           np.linspace(-7, -2, num_rates),
                           np.linspace(-8, -4, num_rates),
                           np.linspace(1, -12, num_rates)]

    # create a list to store the opt object
    store_opt_object = [0]

    if multi_proc:
        st_time = time.time()

        pool = mp.Pool(processes=number_of_cores)

        results = []

        for init_rate in init_rates_list:

            kwds_pool = {'init_rate_guess': init_rate,
                         'exp_isotope_dist_array': norm_mass_distribution_array,
                         'sequence': sequence,
                         'timepoints': time_points,
                         'inv_backexchange_array': inv_back_exchange_array,
                         'd2o_fraction': d2o_fraction,
                         'd2o_purity': d2o_purity,
                         'num_bins': num_bins_,
                         'free_energy_values': free_energy_values,
                         'temperature': temperature,
                         'opt_iter': opt_iter,
                         'opt_temp': opt_temp,
                         'opt_step_size': opt_step_size,
                         'return_tuple': True}

            pool_result = pool.apply_async(hx_rate_fitting_optimization, kwds=kwds_pool)
            results.append(pool_result)

        pool.close()

        tuple_results_list = [p.get() for p in results]
        opt_object_list = [tup[0] for tup in tuple_results_list]
        opt_obj_fun_list = np.array([x.fun for x in opt_object_list])
        min_fun_ind = np.argmin(opt_obj_fun_list)
        store_opt_object[0] = opt_object_list[min_fun_ind]
        init_rate_used = tuple_results_list[min_fun_ind][1]

        finish_time = time.time()

        time_took = finish_time - st_time

    else:

        st_time = time.time()

        opt_cost = 10
        init_rate_final_ind = -1

        for ind, init_rate in enumerate(init_rates_list[:1]):

            opt_ = hx_rate_fitting_optimization(exp_isotope_dist_array=norm_mass_distribution_array,
                                                sequence=sequence,
                                                timepoints=time_points,
                                                inv_backexchange_array=inv_back_exchange_array,
                                                d2o_fraction=d2o_fraction,
                                                d2o_purity=d2o_purity,
                                                num_bins=num_bins_,
                                                free_energy_values=free_energy_values,
                                                temperature=temperature,
                                                init_rate_guess=init_rate,
                                                opt_iter=opt_iter,
                                                opt_temp=opt_temp,
                                                opt_step_size=opt_step_size)

            new_opt_cost = opt_.fun
            if new_opt_cost < opt_cost:
                init_rate_final_ind = ind
                store_opt_object[0] = opt_
            opt_cost = new_opt_cost

        init_rate_used = init_rates_list[init_rate_final_ind]

        finish_time = time.time()
        time_took = finish_time - st_time

    opt_object = store_opt_object[0]
    hxrate.optimization_cost = opt_object.fun
    hxrate.optimization_func_evals = opt_object.nfev
    hxrate.optimization_init_rate_guess = init_rate_used
    hxrate.hx_rates = opt_object.x

    # generate theoretical isotope distribution array
    hxrate.thr_isotope_dist_array = gen_theoretical_isotope_dist_for_all_timepoints(sequence=sequence,
                                                                                    timepoints=time_points,
                                                                                    rates=np.exp(hxrate.hx_rates),
                                                                                    inv_backexchange_array=inv_back_exchange_array,
                                                                                    d2o_fraction=d2o_fraction,
                                                                                    d2o_purity=d2o_purity,
                                                                                    num_bins=num_bins_,
                                                                                    free_energy_values=free_energy_values,
                                                                                    temperature=temperature)

    # compute the fit mse for each distribution
    hxrate.fit_rmse_each_timepoint = np.zeros(len(time_points))
    for ind, (exp_dist, thr_dist) in enumerate(zip(norm_mass_distribution_array, hxrate.thr_isotope_dist_array)):
        hxrate.fit_rmse_each_timepoint[ind] = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=exp_dist,
                                                                            thr_isotope_dist=thr_dist,
                                                                            squared=False)

    # compute the total fit mse distribution
    exp_isotope_dist_concat = np.concatenate(norm_mass_distribution_array)
    thr_isotope_dist_concat = np.concatenate(hxrate.thr_isotope_dist_array)
    thr_isotope_dist_concat[np.isnan(thr_isotope_dist_concat)] = 0
    hxrate.total_fit_rmse = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=exp_isotope_dist_concat,
                                                          thr_isotope_dist=thr_isotope_dist_concat,
                                                          squared=False)

    print('time took for hx rate fitting optimization: ', time_took)

    return hxrate


def fit_rate_from_to_file(sequence: str,
                          hx_ms_dist_fpath: str,
                          d2o_fraction: float,
                          d2o_purity: float,
                          opt_iter: int,
                          opt_temp: float,
                          opt_step_size: float,
                          multi_proc: bool = False,
                          free_energy_values: np.ndarray = None,
                          temperature: float = None,
                          hx_rate_output_path: str = None,
                          hx_rate_csv_output_path: str = None,
                          hx_isotope_dist_output_path: str = None,
                          hx_rate_plot_path: str = None,
                          hx_isotope_dist_plot_path: str = None,
                          return_flag: bool = False) -> object:
    # todo: add param descriptions for the function here

    # get the mass distribution data at each hdx timepoint
    timepoints, mass_dist = load_data_from_hdx_ms_dist_(hx_ms_dist_fpath)

    # normalize the mass distribution
    norm_dist = normalize_mass_distribution_array(mass_dist_array=mass_dist)

    # fit rate
    hxrate_object = fit_rate(sequence=sequence,
                             time_points=timepoints,
                             norm_mass_distribution_array=norm_dist,
                             d2o_fraction=d2o_fraction,
                             d2o_purity=d2o_purity,
                             opt_iter=opt_iter,
                             opt_temp=opt_temp,
                             opt_step_size=opt_step_size,
                             multi_proc=multi_proc,
                             free_energy_values=free_energy_values,
                             temperature=temperature)

    # convert hxrate object to dict and save as a pickle file
    if hx_rate_output_path is not None:
        hxrate_object.back_exchange = vars(hxrate_object.back_exchange)
        hxrate_vars = vars(hxrate_object)
        write_pickle_object(hxrate_vars, hx_rate_output_path)

    # write hxrate as a csv file
    if hx_rate_csv_output_path is not None:
        write_hx_rate_output(hxrate_object.hx_rates, hx_rate_csv_output_path)

    # write hxrate distributions as a csv file
    if hx_isotope_dist_output_path is not None:
        write_isotope_dist_timepoints(timepoints=timepoints,
                                      isotope_dist_array=hxrate_object.thr_isotope_dist_array,
                                      output_path=hx_isotope_dist_output_path)

    # plot hxrate as .pdf file
    if hx_rate_plot_path is not None:
        plot_hx_rates(hx_rates=hxrate_object.hx_rates, output_path=hx_rate_plot_path)

    # plot hxrate dist as .pdf file
    if hx_isotope_dist_plot_path is not None:
        plot_exp_thr_dist(exp_dist_array=norm_dist,
                          thr_dist_array=hxrate_object.thr_isotope_dist_array,
                          timepoints_array=timepoints,
                          output_path=hx_isotope_dist_plot_path,
                          rmse_each_timepoint=hxrate_object.fit_rmse_each_timepoint,
                          total_rmse=hxrate_object.total_fit_rmse)

    if return_flag:
        return hxrate_object


if __name__ == '__main__':

    d2o_fraction_ = 0.95
    d2o_purity_ = 0.95
    opt_iter_ = 50
    opt_temp_ = 0.00003
    opt_step_size_ = 0.02
    multi_proc_ = True

    import pandas as pd

    sample_fpath = '../../workfolder/sample.csv'
    sample_df = pd.read_csv(sample_fpath)
    prot_name = sample_df['name'].values[0]
    prot_seq = sample_df['sequence'].values[0]
    hx_ms_dist_fpath = sample_df['hx_dist_fpath'].values[0]

    output_dirpath = '../../workfolder/output_hxrate/'

    fit_rate_from_to_file(sequence=prot_seq,
                          hx_ms_dist_fpath=hx_ms_dist_fpath,
                          d2o_purity=d2o_purity_,
                          d2o_fraction=d2o_fraction_,
                          opt_iter=opt_iter_,
                          opt_temp=opt_temp_,
                          opt_step_size=opt_step_size_,
                          multi_proc=multi_proc_,
                          free_energy_values=None,
                          temperature=None,
                          hx_rate_output_path=output_dirpath + 'hx_rate_object.pickle',
                          hx_rate_csv_output_path=output_dirpath + 'hx_rate.csv',
                          hx_isotope_dist_output_path=output_dirpath + 'hxrate_isotope_dist.csv',
                          hx_rate_plot_path=output_dirpath + 'hx_rate.pdf',
                          hx_isotope_dist_plot_path=output_dirpath + 'hx_rate_isotope_dist.pdf')
