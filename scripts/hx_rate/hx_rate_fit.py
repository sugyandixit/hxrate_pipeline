# hx_rate_fitting code
# authors: Gabe Rocklin and Suggie

import multiprocessing as mp
import numpy as np
from dataclasses import dataclass
from scipy.special import expit
from scipy.optimize import fmin_powell
from methods import isotope_dist_from_PoiBin, gen_temp_rates, gen_theoretical_isotope_dist_for_all_timepoints, \
    normalize_mass_distribution_array, hx_rate_fitting_optimization, compute_rmse_exp_thr_iso_dist, \
    gauss_fit_to_isotope_dist_array, convert_hxrate_object_to_dict, plot_hx_rate_fitting_
from hxdata import load_data_from_hdx_ms_dist_, write_pickle_object, write_hx_rate_output, write_isotope_dist_timepoints
import time


@dataclass
class ExpData(object):
    """
    class container to store exp data
    """
    protein_sequence: str = None
    timepoints: np.ndarray = None
    exp_isotope_dist_array: np.ndarray = None
    gauss_fit: list = None


@dataclass
class HXRate(object):
    """
    class container to store hx rate fitting data
    """
    exp_data: object = None
    back_exchange: object = None
    optimization_cost: float = None
    optimization_func_evals: int = None
    optimization_init_rate_guess: np.ndarray = None
    hx_rates: np.ndarray = None
    thr_isotope_dist_array: np.ndarray = None
    thr_isotope_dist_gauss_fit: list = None
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

    print('\nCALCULATING BACK EXCHANGE ... ')

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

    # todo: add param descritpions

    # initialize hxrate data object
    hxrate = HXRate()

    # store exp data in the object
    expdata = ExpData(protein_sequence=sequence,
                      timepoints=time_points,
                      exp_isotope_dist_array=norm_mass_distribution_array)

    # fit gaussian to exp data
    expdata.gauss_fit = gauss_fit_to_isotope_dist_array(norm_mass_distribution_array)

    # store expdata object in hxrate data object
    hxrate.exp_data = expdata

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

    print('\nHX RATE FITTING ... ')

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

        # set the max number of cores to be equal to the number of initial rate guesses
        if number_of_cores > len(init_rates_list):
            number_of_cores = len(init_rates_list)

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

    # fit gaussian to the thr isotope dist
    hxrate.thr_isotope_dist_gauss_fit = gauss_fit_to_isotope_dist_array(hxrate.thr_isotope_dist_array)

    return hxrate


def fit_rate_from_to_file(sequence: str,
                          hx_ms_dist_fpath: str,
                          d2o_fraction: float,
                          d2o_purity: float,
                          opt_iter: int,
                          opt_temp: float,
                          opt_step_size: float,
                          multi_proc: bool = False,
                          number_of_cores: int = 6,
                          free_energy_values: np.ndarray = None,
                          temperature: float = None,
                          hx_rate_output_path: str = None,
                          hx_rate_csv_output_path: str = None,
                          hx_isotope_dist_output_path: str = None,
                          hx_rate_plot_path: str = None,
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
                             number_of_cores=number_of_cores,
                             free_energy_values=free_energy_values,
                             temperature=temperature)

    # convert hxrate object to dict and save as a pickle file

    # write hxrate as a csv file
    if hx_rate_csv_output_path is not None:
        write_hx_rate_output(hxrate_object.hx_rates, hx_rate_csv_output_path)

    # write hxrate distributions as a csv file
    if hx_isotope_dist_output_path is not None:
        write_isotope_dist_timepoints(timepoints=timepoints,
                                      isotope_dist_array=hxrate_object.thr_isotope_dist_array,
                                      output_path=hx_isotope_dist_output_path)

    # plot hxrate output as .pdf file
    if hx_rate_plot_path is not None:

        exp_centroid_arr = np.array([x.centroid for x in hxrate_object.exp_data.gauss_fit])
        exp_width_arr = np.array([x.width for x in hxrate_object.exp_data.gauss_fit])

        thr_centroid_arr = np.array([x.centroid for x in hxrate_object.thr_isotope_dist_gauss_fit])
        thr_width_arr = np.array([x.width for x in hxrate_object.thr_isotope_dist_gauss_fit])

        plot_hx_rate_fitting_(hx_rates=hxrate_object.hx_rates,
                              exp_isotope_dist=norm_dist,
                              thr_isotope_dist=hxrate_object.thr_isotope_dist_array,
                              exp_isotope_centroid_array=exp_centroid_arr,
                              thr_isotope_centroid_array=thr_centroid_arr,
                              exp_isotope_width_array=exp_width_arr,
                              thr_isotope_width_array=thr_width_arr,
                              timepoints=time_points,
                              fit_rmse_timepoints=hxrate_object.fit_rmse_each_timepoint,
                              fit_rmse_total=hxrate_object.total_fit_rmse,
                              backexchange=hxrate_object.back_exchange.backexchange_value,
                              output_path=hx_rate_plot_path)

    # save hxrate object into a pickle object
    if hx_rate_output_path is not None:
        hxrate_dict = convert_hxrate_object_to_dict(hxrate_object=hxrate_object)
        write_pickle_object(hxrate_dict, hx_rate_output_path)

    if return_flag:
        return hxrate_object


if __name__ == '__main__':

    d2o_fraction_ = 0.95
    d2o_purity_ = 0.95
    opt_iter_ = 30
    opt_temp_ = 0.00003
    opt_step_size_ = 0.02
    multi_proc_ = True
    num_cores = 6

    import pandas as pd

    sample_fpath = '../../workfolder/sample.csv'
    sample_df = pd.read_csv(sample_fpath)
    prot_name = sample_df['name'].values[0]
    prot_seq = sample_df['sequence'].values[0]
    hx_ms_dist_fpath_ = sample_df['hx_dist_fpath'].values[0]

    output_dirpath = '../../workfolder/output_hxrate/'

    fit_rate_from_to_file(sequence=prot_seq,
                          hx_ms_dist_fpath=hx_ms_dist_fpath_,
                          d2o_purity=d2o_purity_,
                          d2o_fraction=d2o_fraction_,
                          opt_iter=opt_iter_,
                          opt_temp=opt_temp_,
                          opt_step_size=opt_step_size_,
                          multi_proc=multi_proc_,
                          number_of_cores=num_cores,
                          free_energy_values=None,
                          temperature=None,
                          hx_rate_output_path=output_dirpath + 'hx_rate_object.pickle',
                          hx_rate_csv_output_path=output_dirpath + 'hx_rate.csv',
                          hx_isotope_dist_output_path=output_dirpath + 'hxrate_isotope_dist.csv',
                          hx_rate_plot_path=output_dirpath + 'hx_rate.pdf',
                          hx_isotope_dist_plot_path=output_dirpath + 'hx_rate_isotope_dist.pdf')
