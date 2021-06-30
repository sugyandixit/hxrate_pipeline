# hx_rate_fitting code
# original Author: Gabe Rocklin
# reformatted: Suggie

import numpy as np
import pandas as pd
import molmass
from dataclasses import dataclass


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


if __name__ == '__main__':

    hx_dist_fpath = '/Users/smd4193/PycharmProjects/hxrate/workfolder/input_hx_dist/HEEH_rd4_0097_hx_mass_dist.csv'
    tp, mdist = get_data_from_mz_dist_file(hx_dist_fpath)
    distdata = HXDistData(prot_name='EEEEE',
                          prot_sequence='LLLLL',
                          ph=6.9,
                          d2o_frac=0.95,
                          d2o_purity=0.95,
                          temp=298,
                          timepoints=tp,
                          mass_distribution_array=mdist)
