# hx_rate_fitting code
# authors: Gabe Rocklin and Suggie

import numpy as np
import pandas as pd
import molmass
from dataclasses import dataclass
from scipy.optimize import fmin_powell, basinhopping
from scipy.special import expit
from sklearn.metrics import mean_squared_error


def PoiBin(success_probabilities):
    """
    poisson binomial probability distribution function
    :param success_probabilities:
    :return:
    """
    number_trials = success_probabilities.size

    omega = 2 * np.pi / (number_trials + 1)

    chi = np.empty(number_trials + 1, dtype=complex)
    chi[0] = 1
    half_number_trials = int(
        number_trials / 2 + number_trials % 2)
    # set first half of chis:

    # idx_array = np.arange(1, half_number_trials + 1)
    exp_value = np.exp(omega * np.arange(1, half_number_trials + 1) * 1j)
    xy = 1 - success_probabilities + success_probabilities * exp_value[:, np.newaxis]
    # sum over the principal values of the arguments of z:
    argz_sum = np.arctan2(xy.imag, xy.real).sum(axis=1)
    # get d value:
    # exparg = np.log(np.abs(xy)).sum(axis=1)
    d_value = np.exp(np.log(np.abs(xy)).sum(axis=1))
    # get chi values:
    chi[1:half_number_trials + 1] = d_value * np.exp(argz_sum * 1j)

    # set second half of chis:
    chi[half_number_trials + 1:number_trials + 1] = np.conjugate(
        chi[1:number_trials - half_number_trials + 1][::-1])
    chi /= number_trials + 1
    xi = np.fft.fft(chi)
    return xi.real


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


def calculate_theoretical_isotope_dist_from_sequence(sequence, num_isotopes=None):
    """
    calculate theoretical isotope distribtuion from a given one letter sequence of protein chain
    :param sequence: protein sequence in one letter code
    :param num_isotopes: number of isotopes to include. If none, includes all
    :return: isotope distribution
    """
    seq_formula = molmass.Formula(sequence)
    isotope_dist = np.array([x[1] for x in seq_formula.spectrum().values()])
    isotope_dist = isotope_dist/max(isotope_dist)
    if num_isotopes:
        if num_isotopes < len(isotope_dist):
            isotope_dist = isotope_dist[:num_isotopes]
        else:
            fill_arr = np.zeros(num_isotopes - len(isotope_dist))
            isotope_dist = np.append(isotope_dist, fill_arr)
    return isotope_dist


def cal_hx_prob_with_backexchange(timepoints, rate_constant, backexchange, d2o_purity, d2o_fraction):
    """
    calculate the exchange probability for each time point given the rate constant, backexchange, and d20 purity and
    fraction
    :param timepoints: array of time points (seconds)
    :param rate_constant: array of rate constants or a single rate
    :param backexchange: backexchange rate
    :param d2o_purity: purity of d2o
    :param d2o_fraction:
    :return: exchange probabilities
    """
    prob = (1.0 - np.exp(-rate_constant * timepoints)) * (d2o_fraction * d2o_purity * backexchange)
    return prob


def hx_rates_probability_distribution_with_fes(timepoints, rates, backexchange, d2o_fraction, d2o_purity,
                                                fes, temp):
    """

    :param timepoints:
    :param rates:
    :param backexchange:
    :param d2o_fraction:
    :param d2o_purity:
    :param blank_fes:
    :return:
    """
    r_constant = 0.0019872036
    fractions = np.exp(-fes / (r_constant * temp)) / (1.0 + np.exp(-fes / (r_constant * temp)))
    hx_probabs = cal_hx_prob_with_backexchange(timepoints=timepoints,
                                               rate_constant=rates*fractions,
                                               backexchange=backexchange,
                                               d2o_purity=d2o_purity,
                                               d2o_fraction=d2o_fraction)
    pmf_hx_probabs = PoiBin(hx_probabs)
    return pmf_hx_probabs


def isotope_dist_from_PoiBin_bkexch(sequence_length, isotope_dist, timepoint, rates, num_bins, backexchange,
                                                       d2o_fraction, d2o_purity, temp):
    """
    returns the convolved isotopic distribution from the pfm of hx rates probabilities
    :param isotope_dist: isotope distribution
    :param timepoints: time points array in seconds
    :param rates: measured rates array
    :param num_bins: number of bins
    :param backexchange: back exchange rate as above
    :param d2o_fraction: fraction of d2o
    :param d2o_purity: purity of d2o
    :return: isotope distribution
    """
    fes = np.zeros(sequence_length)
    pmf_hx_prob_fes = hx_rates_probability_distribution_with_fes(timepoints=timepoint,
                                                                 rates=rates,
                                                                 backexchange=backexchange,
                                                                 d2o_fraction=d2o_fraction,
                                                                 d2o_purity=d2o_purity,
                                                                 fes=fes,
                                                                 temp=temp)
    isotope_dist_poibin_convol = np.convolve(pmf_hx_prob_fes, isotope_dist)
    isotope_dist_poibin_convol_norm = isotope_dist_poibin_convol[:num_bins]/max(isotope_dist_poibin_convol[:num_bins])
    return isotope_dist_poibin_convol_norm


def calc_back_exchange_without_free_energies(sequence_length, theoretical_isotope_dist, experimental_isotope_dist,
                                             intrinsic_rates, temperature, d2o_fraction=0.95, d2o_purity=0.95):
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

    back_exchange = expit(opt)

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
    temp: float
    timepoints: np.ndarray
    mass_distribution_array: np.ndarray

    def normalize_distribution(self):
        norm_dist = []
        for dist in self.mass_distribution_array:
            norm_ = dist/max(dist)
            norm_dist.append(norm_)
        norm_dist = np.asarray(norm_dist)
        return norm_dist


if __name__ == '__main__':

    hx_dist_fpath = '../workfolder/input_hx_dist/HEEH_rd4_0097_hx_mass_dist.csv'
    sample_csv_fpath = '../workfolder/sample.csv'
    sample_df = pd.read_csv(sample_csv_fpath)
    prot_name = sample_df['name'].values[0]
    prot_seq = sample_df['sequence'].values[0]
    tp, mdist = get_data_from_mz_dist_file(hx_dist_fpath)
    distdata = HXDistData(prot_name=prot_name,
                          prot_sequence=prot_seq,
                          ph=6.9,
                          d2o_frac=0.95,
                          d2o_purity=0.95,
                          temp=298,
                          timepoints=tp,
                          mass_distribution_array=mdist)
