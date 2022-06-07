import numpy as np
from scipy.special import expit
from scipy.optimize import fmin_powell
from dataclasses import dataclass
from methods import gen_corr_backexchange, gen_backexchange_correction_from_backexchange_array, gen_temp_rates, \
    isotope_dist_from_PoiBin, compute_rmse_exp_thr_iso_dist


@dataclass
class BackExchange(object):
    """
    class container to store backexchange data
    """
    timepoints: np.ndarray = None
    experimental_isotope_dist: np.ndarray = None
    backexchange_value: float = None
    fit_rmse: float = None
    theoretical_isotope_dist: np.ndarray = None
    backexchange_correction_dict: dict = None
    backexchange_array: np.ndarray = None

    def gen_tp_backexchange_array(self):
        """
        generate timepoint specific backexchange array
        :param timepoints:
        :return:
        """
        if self.backexchange_correction_dict is None:
            self.backexchange_array = np.array([self.backexchange_value for _ in range(len(self.timepoints))])
        else:
            tp_backexchange_corr_array = np.zeros(len(self.timepoints))
            for ind, tp in enumerate(self.timepoints):
                tp_backexchange_corr_array[ind] = self.backexchange_correction_dict[tp]
            self.backexchange_array = gen_corr_backexchange(mass_rate_array=tp_backexchange_corr_array,
                                                            fix_backexchange_value=self.backexchange_value)

    def gen_backexchange_correction_dict(self):
        """
        generate backexchange correction dict
        :return:
        """
        backexch_corr = gen_backexchange_correction_from_backexchange_array(backexchange_array=self.backexchange_array)
        corr_dict = dict()
        for ind, (tp, bkex_corr) in enumerate(zip(self.timepoints, backexch_corr)):
            corr_dict[tp] = bkex_corr
        self.backexchange_correction_dict = corr_dict


def calc_back_exchange(sequence: str,
                       experimental_isotope_dist: np.ndarray,
                       timepoints_array: np.ndarray,
                       d2o_fraction: float,
                       d2o_purity: float,
                       bkex_tp: float = 1e9,
                       usr_backexchange: float = None,
                       backexchange_array: np.ndarray = None,
                       backexchange_corr_dict: dict = None) -> object:
    """
    calculate back exchange from the experimental isotope distribution
    :param sequence: protein sequence
    :param experimental_isotope_dist: experimental isotope distribution to be used for backexchange calculation
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param timepoint: hdx timepoint in seconds. default to 1e9 for full deuteration
    :param usr_backexchange: if usr backexchange provided, will generate backexchange object based on that backexchange
    value
    :param backexchange_array: provide backexchange array
    :return: backexchange data object
    """
    # set high rates for calculating back exchange
    rates = gen_temp_rates(sequence=sequence,
                           rate_value=1e4)

    # set the number of bins for isotope distribtuion
    num_bins = len(experimental_isotope_dist)

    # initiate Backexchange Object
    backexchange_obj = BackExchange(experimental_isotope_dist=experimental_isotope_dist,
                                    timepoints=timepoints_array,
                                    backexchange_array=backexchange_array,
                                    backexchange_correction_dict=backexchange_corr_dict)

    if backexchange_array is None:
        if usr_backexchange is None:
            print('\nCALCULATING BACK EXCHANGE ... ')

            opt = fmin_powell(lambda x: compute_rmse_exp_thr_iso_dist(exp_isotope_dist=experimental_isotope_dist,
                                                                      thr_isotope_dist=isotope_dist_from_PoiBin(sequence=sequence,
                                                                                                                timepoint=bkex_tp,
                                                                                                                inv_backexchange=expit(x),
                                                                                                                rates=rates,
                                                                                                                d2o_fraction=d2o_fraction,
                                                                                                                d2o_purity=d2o_purity,
                                                                                                                num_bins=num_bins),
                                                                      squared=False),
                              x0=0.05,
                              disp=True)

            backexchange_obj.backexchange_value = 1 - expit(opt)[0]

            # check if backexchange value is 1
            # if backexchange value is 1, set it to default of 0.98
            if backexchange_obj.backexchange_value == 1:
                print('\nIsotopeDistribution close to undeuterated state led to backexchange value of 1. Setting it to a default value of 0.98')
                backexchange_obj.backexchange_value = 0.98

        else:
            print('\nSETTING USER BACK EXCHANGE ... ')
            backexchange_obj.backexchange_value = usr_backexchange

        # generate timepoint specific backexchange array
        backexchange_obj.gen_tp_backexchange_array()

    else:
        print('\nSETTING BACK EXCHANGE VALUE FROM THE BACKEXCHANGE ARRAY ... ')
        backexchange_obj.backexchange_value = backexchange_array[-1]
        backexchange_obj.backexchange_array = backexchange_array

        print('\nGENERATING BACKEXCHANGE CORRECTION DICTIONARY ... ')
        backexchange_obj.gen_backexchange_correction_dict()

    backexchange_obj.theoretical_isotope_dist = isotope_dist_from_PoiBin(sequence=sequence,
                                                                         timepoint=bkex_tp,
                                                                         inv_backexchange=1 - backexchange_obj.backexchange_value,
                                                                         rates=rates,
                                                                         d2o_fraction=d2o_fraction,
                                                                         d2o_purity=d2o_purity,
                                                                         num_bins=num_bins)

    backexchange_obj.fit_rmse = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=experimental_isotope_dist,
                                                              thr_isotope_dist=backexchange_obj.theoretical_isotope_dist,
                                                              squared=False)

    return backexchange_obj
