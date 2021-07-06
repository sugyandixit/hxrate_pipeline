# hx_rate_fitting code
# authors: Gabe Rocklin and Suggie

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from scipy.special import expit
from scipy.optimize import basinhopping, fmin_powell
from methods import isotope_dist_from_PoiBin, gen_temp_rates, calc_intrinsic_hx_rates


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


def fit_rate(sequence: str,
             time_points: np.ndarray,
             norm_mass_distribution_array: np.ndarray,
             d2o_fraction: float,
             d2o_purity: float,
             opt_iter: int,
             opt_temp: float,
             opt_step_size: float,
             multi_proc: bool = True,
             rate_init_list: list = None,
             free_energy_values: np.ndarray = None,
             temperature: float = None,
             backexchange_value: float = None) -> object:

    # initialize hxrate data object
    hxrate = HXRate()

    intrinsic_rates = calc_intrinsic_hx_rates(sequence_str=sequence,
                                              temperature=298,
                                              ph=5.9)
    intrinsic_rates[:2] = 0

    print('heho')

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

    # concatenate exp mass distribution
    exp_isotope_dist_concat = np.concatenate(norm_mass_distribution_array)

    # generate a temporary rates to determine what residues can't exchange
    temp_rates = gen_temp_rates(sequence=sequence, rate_value=1)

    # get the indices of residues that don't exchange
    zero_indices = np.where(temp_rates == 0)[0]

    # calculate the number of residues that can exchange
    num_rates = len(temp_rates) - len(zero_indices)

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

    # set an initial optimization cost value
    opt_cost = 10

    # if multi_proc:
    #
    #     print('put multi processing code here')
    #
    # else:
    #
    #     for init_rate in init_rates_list:
    #
    #         opt_ = basinhopping(lambda rates: hx_rate_fit_rmse(timepoints=time_points,
    #                                                            rates=rates,
    #                                                            thr_isotope_dist_list=thr_isotope_dist_list,
    #                                                            exp_isotope_dist_concat=exp_isotope_dist_concat,
    #                                                            num_bins=num_bins,
    #                                                            backexchange_arr=inv_back_exchange_array,
    #                                                            d2o_purity=d2o_purity,
    #                                                            d2o_fraction=d2o_fraction),
    #                             x0=init_rate,
    #                             niter=n_iter,
    #                             T=temp,
    #                             stepsize=step_size,
    #                             minimizer_kwargs={'options': {'maxiter': 1}})
    #
    #         new_opt_cost = opt_.fun
    #         if new_opt_cost < opt_cost:
    #             opt = opt_
    #         opt_cost = new_opt_cost
    #
    # return hx_rate


@dataclass
class HXDistData(object):
    """
    data container object to store data for HX rate fitting
    """

    prot_name: str
    prot_sequence: str
    d2o_frac: float
    d2o_purity: float
    timepoints: np.ndarray
    mass_distribution_array: np.ndarray
    nterm_seq_add: str = None
    cter_seq_add: str = None
    ph: float = None
    temperature: float = None

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

    back_exchange: object = None
    fit_rates: np.ndarray = None
    fit_thr_isotope_dist: np.ndarray = None
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

    fit_rate(sequence=prot_seq,
             time_points=tp,
             norm_mass_distribution_array=norm_dist_,
             d2o_fraction=d2o_fraction_,
             d2o_purity=d2o_purity_,
             opt_iter=100,
             opt_temp=100,
             opt_step_size=100)