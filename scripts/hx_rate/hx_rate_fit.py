# hx_rate_fitting code
# authors: Gabe Rocklin and Suggie

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import fmin_powell, basinhopping
from scipy.special import expit
from sklearn.metrics import mean_squared_error
from methods import isotope_dist_from_PoiBin_bkexch, theoretical_isotope_dist_from_sequence, calc_intrinsic_hx_rates


def get_data_from_mz_dist_file(fpath):
    """
    get hdx mass distribution data
    :param fpath: input .csv path for distribution
    :return: timepoints, mass_distribution list
    """
    df = pd.read_csv(fpath)
    timepoints = np.asarray(df.columns.values[1:], dtype=float)
    mass_distiribution = df.iloc[:, 1:].values.T
    return timepoints, mass_distiribution


def calc_back_exchange(sequence_length, theoretical_isotope_dist, experimental_isotope_dist, intrinsic_rates,
                       temperature, d2o_fraction=0.95, d2o_purity=0.95):
    """
    calculate back exchange from the experimental isotope distribution
    :param sequence_length: length of the protein sequence
    :param theoretical_isotope_dist: theoretical isotope distribution of the protein
    :param experimental_isotope_dist: experimental isotope distribution to use for calculating backexchange
    :param intrinsic_rates: intrinsic rates
    :param d2o_fraction:
    :param d2o_purity:
    :param temperature:
    """

    intrinsic_rates[:2] = 0

    num_bins_ = len(experimental_isotope_dist)

    opt = fmin_powell(lambda x: mean_squared_error(experimental_isotope_dist,
                                                   isotope_dist_from_PoiBin_bkexch(sequence_length=sequence_length,
                                                                                   isotope_dist=theoretical_isotope_dist,
                                                                                   timepoint=1e9,
                                                                                   rates=intrinsic_rates,
                                                                                   num_bins=num_bins_,
                                                                                   backexchange=expit(x),
                                                                                   d2o_fraction=d2o_fraction,
                                                                                   d2o_purity=d2o_purity,
                                                                                   temp=temperature),
                                                   squared=False), x0=2, disp=True)

    back_exchange = 1 - expit(opt)[0]

    return back_exchange


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

    def normalize_distribution(self):
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

    temp = 298
    ph = 5.9
    d2o_fraction = 0.95
    d2o_purity = 0.95

    hx_dist_fpath = '../../workfolder/input_hx_dist/HEEH_rd4_0097_hx_mass_dist.csv'
    sample_csv_fpath = '../../workfolder/sample.csv'
    sample_df = pd.read_csv(sample_csv_fpath)
    prot_name = sample_df['name'].values[0]
    prot_seq = sample_df['sequence'].values[0]
    tp, mdist = get_data_from_mz_dist_file(hx_dist_fpath)

    distdata = HXDistData(prot_name=prot_name,
                          prot_sequence=prot_seq,
                          ph=ph,
                          d2o_frac=d2o_fraction,
                          d2o_purity=d2o_purity,
                          temperature=temp,
                          timepoints=tp,
                          mass_distribution_array=mdist)

    norm_dist = distdata.normalize_distribution()

    intrinsic_rates = calc_intrinsic_hx_rates(sequence_str=prot_seq,
                                              Temperature=temp,
                                              pH=ph)

    back_exch = calc_back_exchange(sequence_length=len(prot_seq),
                                   theoretical_isotope_dist=theoretical_isotope_dist_from_sequence(prot_seq),
                                   experimental_isotope_dist=norm_dist[-1],
                                   intrinsic_rates=intrinsic_rates,
                                   temperature=temp,
                                   d2o_fraction=d2o_fraction,
                                   d2o_purity=d2o_purity)
    print('heho')
