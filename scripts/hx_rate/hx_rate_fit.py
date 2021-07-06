# hx_rate_fitting code
# authors: Gabe Rocklin and Suggie

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from scipy.special import expit
from scipy.optimize import basinhopping, fmin_powell
from methods import isotope_dist_from_PoiBin


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
    rates = np.array([1e2]*len(sequence), dtype=float)

    # set the rates for the first two residues as 0
    rates[:2] = 0

    # set the rate for proline to be 0
    if 'P' in sequence:
        amino_acid_list = [x for x in sequence]
        for ind, amino_acid in enumerate(amino_acid_list):
            if amino_acid == 'P':
                rates[ind] = 0

    num_bins = len(experimental_isotope_dist)

    opt = fmin_powell(lambda x: mean_squared_error(experimental_isotope_dist,
                                                   isotope_dist_from_PoiBin(sequence=sequence,
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

    fit_rmse = mean_squared_error(y_true=experimental_isotope_dist,
                                  y_pred=thr_iso_dist_full_deut,
                                  squared=False)

    backexchange_obj = BackExchange(backexchange_value=back_exchange,
                                    fit_rmse=fit_rmse,
                                    theoretical_isotope_dist=thr_iso_dist_full_deut)

    return backexchange_obj


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


@dataclass
class BackExchange(object):
    """
    class container to store backexchange data
    """
    backexchange_value: float = None
    fit_rmse: float = None
    theoretical_isotope_dist: np.ndarray = None
    backexchange_array: np.ndarray = None


if __name__ == '__main__':

    temp_ = 298
    ph_ = 5.9
    d2o_fraction_ = 0.95
    d2o_purity_ = 0.95

    import hxdata
    prot_name, prot_seq, tp, mdist = hxdata.load_sample_data()

    distdata = HXDistData(prot_name=prot_name,
                          prot_sequence=prot_seq,
                          ph=ph_,
                          d2o_frac=d2o_fraction_,
                          d2o_purity=d2o_purity_,
                          temperature=temp_,
                          timepoints=tp,
                          mass_distribution_array=mdist)

    norm_dist_ = distdata.normalize_distribution()

    back_exch = calc_back_exchange(sequence=prot_seq,
                                   experimental_isotope_dist=norm_dist_[-1],
                                   d2o_fraction=d2o_fraction_,
                                   d2o_purity=d2o_purity_)
