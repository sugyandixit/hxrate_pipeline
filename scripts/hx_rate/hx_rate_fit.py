# hx_rate_fitting code
# authors: Gabe Rocklin and Suggie

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import basinhopping
from methods import theoretical_isotope_dist, calc_intrinsic_hx_rates, back_exchange_


def calc_back_exchange(sequence: str,
                       experimental_isotope_dist: np.ndarray,
                       temperature: float,
                       ph: float,
                       d2o_fraction: float = 0.95,
                       d2o_purity: float = 0.95) -> float:
    """
    calculate back exchange from the experimental isotope distribution
    :param sequence: length of the protein sequence
    :param experimental_isotope_dist: experimental isotope distribution to use for calculating backexchange
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param temperature: temperature in kelvin
    :param ph: pH value
    :return: back_exchange value
    """

    sequence_length = len(sequence)
    thr_isotope_dist = theoretical_isotope_dist(sequence=sequence)

    intrinsic_rates = calc_intrinsic_hx_rates(sequence_str=sequence,
                                              Temperature=temperature,
                                              pH=ph)

    intrinsic_rates[:2] = 0

    back_exchange = back_exchange_(sequence_length=sequence_length,
                                   theoretical_isotope_distribution=thr_isotope_dist,
                                   experimental_isotope_distribution=experimental_isotope_dist,
                                   intrinsic_rates=intrinsic_rates,
                                   temperature=temperature,
                                   d2o_fraction=d2o_fraction,
                                   d2o_purity=d2o_purity)

    return back_exchange


def fit_rate(sequence: str,
             time_points: np.ndarray,
             exp_mass_distribution_array: np.ndarray,
             d2o_fraction: float,
             d2o_purity: float,
             opt_iter: int,
             opt_temp: float,
             opt_step_size: float,
             multi_proc: bool) -> object:
    print('test')
    


# def fit_hx_rates_optimize(sequence, timepoints, thr_isotope_dist_list, exp_isotope_dist_list, num_rates, backexchange,
#                           d2o_fraction=0.95, d2o_purity=0.95, num_bins=None, n_iter=250, temp=0.00003, step_size=0.02,
#                           multi_fit_mode=False):
#     """
#     calculate the hx rates using the time points and the list of experimental isotope distribution
#     :param timepoints: timepoints
#     :param exp_isotope_dist_list: list of experimental isotope distributions
#     :param backexchange: backexchange rate
#     :return: fitted hx rates
#     """
#
#     backexchange_arr = np.array([backexchange for x in timepoints])
#     if len(backexchange_arr) != len(timepoints):
#         backexchange_arr = np.reshape(backexchange_arr, (len(timepoints),))
#
#     exp_isotope_dist_concat = np.concatenate(exp_isotope_dist_list)
#
#     init_rate_set = 0
#
#     if multi_fit_mode:
#
#         init_rates_list = [np.array([-2 for x in range(num_rates)]),
#                            np.array([-5 for x in range(num_rates)]),
#                            np.linspace(-7, 0, num_rates),
#                            np.linspace(-7, -2, num_rates),
#                            np.linspace(-8, -4, num_rates),
#                            np.linspace(1, -12, num_rates)]
#
#         prev_fun = 10
#
#         for ind, init_rate in enumerate(init_rates_list):
#
#             print('multi_fit_rate_set: ' + str(ind+1))
#
#             opt_ = basinhopping(lambda rates: hx_rate_fit_rmse(timepoints=timepoints,
#                                                                rates=rates,
#                                                                thr_isotope_dist_list=thr_isotope_dist_list,
#                                                                exp_isotope_dist_concat=exp_isotope_dist_concat,
#                                                                num_bins=num_bins,
#                                                                backexchange_arr=backexchange_arr,
#                                                                d2o_purity=d2o_purity,
#                                                                d2o_fraction=d2o_fraction),
#                                 x0=init_rate,
#                                 niter=n_iter,
#                                 T=temp,
#                                 stepsize=step_size,
#                                 minimizer_kwargs={'options': {'maxiter': 1}})
#
#             new_fun = opt_.fun
#
#             if new_fun < prev_fun:
#                 opt = opt_
#                 init_rate_set = ind
#
#             prev_fun = new_fun
#
#     else:
#         # init_rates = np.linspace(-8, -4, num_rates)
#         init_rates = np.linspace(-4, -2, num_rates)
#         opt = basinhopping(lambda rates: hx_rate_fit_rmse(timepoints=timepoints,
#                                                           rates=rates,
#                                                           thr_isotope_dist_list=thr_isotope_dist_list,
#                                                           exp_isotope_dist_concat=exp_isotope_dist_concat,
#                                                           num_bins=num_bins,
#                                                           backexchange_arr=backexchange_arr,
#                                                           d2o_purity=d2o_purity,
#                                                           d2o_fraction=d2o_fraction),
#                            x0=init_rates,
#                            niter=n_iter,
#                            T=temp,
#                            stepsize=step_size,
#                            minimizer_kwargs={'options': {'maxiter': 1}})
#
#     return opt, init_rate_set


@dataclass
class HXDistData(object):
    """
    data container object to store data for HX rate fitting
    """

    prot_name: str
    prot_sequence: str
    ph: float
    d2o_frac: float
    d2o_purity: float
    temperature: float
    timepoints: np.ndarray
    mass_distribution_array: np.ndarray
    nterm_seq_add: str = None
    cter_seq_add: str = None

    def normalize_distribution(self) -> np.ndarray:
        norm_dist = []
        for dist in self.mass_distribution_array:
            norm_ = dist/max(dist)
            norm_dist.append(norm_)
        norm_dist = np.asarray(norm_dist)
        return norm_dist


@dataclass
class HXRate(object):
    """
    class container to store hx rate fitting data
    """
    prot_name: str
    prot_seq: str
    ph: float
    d2o_fraction: float
    d2o_purity: float
    temperature: float
    back_exchange: float = None
    back_exchange_list: list = None
    intrinsic_rates: np.ndarray = None
    basin_hopping_temp: float = None
    basin_hopping_step_size: float = None
    basin_hopping_num_iteration: float = None
    basin_hopping_cost_value: float = None
    fit_rates: np.ndarray = None


if __name__ == '__main__':

    temp_ = 298
    ph_ = 5.9
    d2o_fraction_ = 0.95
    d2o_purity_ = 0.95

    hx_dist_fpath = '../../workfolder/input_hx_dist/HEEH_rd4_0097_hx_mass_dist.csv'
    sample_csv_fpath = '../../workfolder/sample.csv'
    sample_df = pd.read_csv(sample_csv_fpath)
    prot_name = sample_df['name'].values[0]
    prot_seq = sample_df['sequence'].values[0]

    from hxdata import load_data_from_hdx_ms_dist_

    tp, mdist = load_data_from_hdx_ms_dist_(hx_dist_fpath)

    distdata = HXDistData(prot_name=prot_name,
                          prot_sequence=prot_seq,
                          ph=ph_,
                          d2o_frac=d2o_fraction_,
                          d2o_purity=d2o_purity_,
                          temperature=temp_,
                          timepoints=tp,
                          mass_distribution_array=mdist)

    norm_dist_ = distdata.normalize_distribution()
