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
                       d2o_fraction: float,
                       d2o_purity: float) -> float:
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
                                              temperature=temperature,
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


# def fit_rate(sequence: str,
#              time_points: np.ndarray,
#              norm_mass_distribution_array: np.ndarray,
#              temperature: float,
#              ph: float,
#              d2o_fraction: float,
#              d2o_purity: float,
#              opt_iter: int,
#              opt_temp: float,
#              opt_step_size: float,
#              multi_proc: bool = True,
#              rate_init_list: list = None) -> object:
#
#     # initialize HXRate object to collect data
#     hx_rate = HXRate(prot_seq=sequence,
#                      ph=ph,
#                      d2o_fraction=d2o_fraction,
#                      d2o_purity=d2o_purity,
#                      temperature=temperature)
#
#     # calculate back exchange first
#
#     hx_rate.back_exchange = calc_back_exchange(sequence=sequence,
#                                                experimental_isotope_dist=norm_mass_distribution_array[-1],
#                                                temperature=temperature,
#                                                ph=ph,
#                                                d2o_fraction=d2o_fraction,
#                                                d2o_purity=d2o_purity)
#
#     # populate backexchange in an array with length equals number of time points
#     hx_rate.back_exchange_array = np.array([hx_rate.back_exchange for x in time_points])
#     if len(hx_rate.back_exchange_array) != len(time_points):
#         hx_rate.back_exchange_array = np.reshape(hx_rate.back_exchange_array, (len(time_points),))
#
#     inv_back_exchange_array = np.subtract(1, hx_rate.back_exchange_array)
#
#     # concatenate exp mass distribution
#     exp_isotope_dist_concat = np.concatenate(norm_mass_distribution_array)
#
#     # calculate intrinsic rates
#     intrinsic_rates = calc_intrinsic_hx_rates(sequence_str=sequence,
#                                               Temperature=temperature,
#                                               pH=ph)
#
#     # assign the first two residue rates 0
#     intrinsic_rates[:2] = 0
#
#     # get the number of rates that have non zero intrinsic rates
#     len_rates = len([x for x in intrinsic_rates if x != 0])
#
#     # get init rate list as initial guesses for rate fit optimization
#     if rate_init_list is not None:
#         init_rates_list = rate_init_list
#     else:
#         init_rates_list = [np.array([-2 for x in range(len_rates)]),
#                            np.array([-5 for x in range(len_rates)]),
#                            np.linspace(-7, 0, len_rates),
#                            np.linspace(-7, -2, len_rates),
#                            np.linspace(-8, -4, len_rates),
#                            np.linspace(1, -12, len_rates)]
#
#     # set an initial optimization cost value
#     opt_cost = 10
#
#     if multi_proc:
#
#         print('put multi processing code here')
#
#     else:
#
#         for init_rate in init_rates_list:
#
#             opt_ = basinhopping(lambda rates: hx_rate_fit_rmse(timepoints=time_points,
#                                                                rates=rates,
#                                                                thr_isotope_dist_list=thr_isotope_dist_list,
#                                                                exp_isotope_dist_concat=exp_isotope_dist_concat,
#                                                                num_bins=num_bins,
#                                                                backexchange_arr=inv_back_exchange_array,
#                                                                d2o_purity=d2o_purity,
#                                                                d2o_fraction=d2o_fraction),
#                                 x0=init_rate,
#                                 niter=n_iter,
#                                 T=temp,
#                                 stepsize=step_size,
#                                 minimizer_kwargs={'options': {'maxiter': 1}})
#
#             new_opt_cost = opt_.fun
#             if new_opt_cost < opt_cost:
#                 opt = opt_
#             opt_cost = new_opt_cost
#
#     return hx_rate


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

    prot_seq: str
    ph: float
    d2o_fraction: float
    d2o_purity: float
    temperature: float
    prot_name: str = None
    back_exchange: float = None
    back_exchange_array: np.ndarray = None
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

    import hxdata
    tp, mdist = hxdata.load_sample_data()

    distdata = HXDistData(prot_name=prot_name,
                          prot_sequence=prot_seq,
                          ph=ph_,
                          d2o_frac=d2o_fraction_,
                          d2o_purity=d2o_purity_,
                          temperature=temp_,
                          timepoints=tp,
                          mass_distribution_array=mdist)

    norm_dist_ = distdata.normalize_distribution()

    # back_exch = calc_back_exchange(sequence=,
    #                                experimental_isotope_dist=,
    #                                temperature=,
    #                                ph=,
    #                                )
