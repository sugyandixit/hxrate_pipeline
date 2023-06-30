import math
import matplotlib.pyplot as plt
import numpy as np
import molmass
import matplotlib.gridspec as gridspec
import psutil as psutil
from scipy.optimize import basinhopping, curve_fit
from sklearn.metrics import mean_squared_error
from scipy.ndimage import center_of_mass
from dataclasses import dataclass
from scipy.stats import linregress
import os

# global variables
r_constant = 0.0019872036


@dataclass
class GaussFit(object):
    """
    class container to store gauss fit data
    """
    fit_success: bool
    gauss_fit_dist: np.ndarray
    y_baseline: float
    amplitude: float
    centroid: float
    width: float
    r_sq: float
    rmse: float


def calculate_intrinsic_exchange_rates(sequence_str, Temperature, pH, nterm_mode='NT', cterm_mode='CT'):
    """
    calculates the intrinsic h exchange rates based on the amino acid sequence for a polypeptide chain
    # calculate the instrinsic exchange rate
    # taken directly from https://gitlab.com/mcpe/psx/blob/master/Code/IntrinsicExchange.py
    # changed the raw values based on the paper J. Am. Soc. Mass Spectrom. (2018) 29;1936-1939
    :param sequence: sequence of the protein (needs to include additional nterm and cterm residues as well)
    :param temp: temperature
    :param ph: ph
    :return: list of intrinsic exchange rates for each residue
    """

    sequence = [x for x in sequence_str]
    sequence.insert(0, nterm_mode)
    sequence.append(cterm_mode)

    # ka, kb, and kw values
    ka = (10.0 ** 1.62)/60
    kb = (10.0 ** 10.18)/60  # changed this value to the one reported on the paper on JASMS! (10.00 init)
    kw = (10.0 ** -1.5)/60

    # Temperature correction
    R = 1.987
    # gabe has the temp correction factor with 278
    # the excel sheet from the englander lab also has 278.0
    TemperatureCorrection = (1.0/Temperature - 1.0/278.0) / R
    Temperature_Corr_2 = (1.0/Temperature - 1.0/293.0) / R
    # TemperatureCorrection = (1.0 / Temperature - 1.0 / 293.0) / R  # disregarding this temperature correction formula

    # Activation energies (in cal/mol)
    AcidActivationEnergy = 14000.0
    BaseActivationEnergy = 17000.0
    SolventActivationEnergy = 19000.0

    AspActivationEnergy = 1000.0
    GluActivationEnergy = 1083.0
    HisActivationEnergy = 7500.0

    # Corrections based on activation energies
    # AcidTemperatureCorrection = math.exp(- TemperatureCorrection * AcidActivationEnergy)
    # BaseTemperatureCorrection = math.exp(- TemperatureCorrection * BaseActivationEnergy)
    # SolventTemperatureCorrection = math.exp(- TemperatureCorrection * SolventActivationEnergy)

    AspTemperatureCorrection = math.exp(- TemperatureCorrection * AspActivationEnergy)
    GluTemperatureCorrection = math.exp(- TemperatureCorrection * GluActivationEnergy)
    HisTemperatureCorrection = math.exp(- TemperatureCorrection * HisActivationEnergy)

    # Corrected pH in D2O
    # pH += 0.4

    # pK-values
    pKD = 15.05
    # pKAsp = 4.48 * AspTemperatureCorrection
    pKAsp = math.log10(10**(-1*4.48)*AspTemperatureCorrection)*-1
    pKGlu = math.log10(10**(-1*4.93)*GluTemperatureCorrection)*-1
    pKHis = math.log10(10**(-1*7.42)*HisTemperatureCorrection)*-1


    # create dictionary to store the amino acids L and R reference values for both acid and base

    MilneAcid = {}

    # MilneAcid["NTerminal"] = (None, RhoAcidNTerm)
    # MilneAcid["CTerminal"] = (LambdaAcidCTerm, None)

    MilneAcid["A"] = (0.00, 0.00)
    MilneAcid["C"] = (-0.54, -0.46)
    MilneAcid["C2"] = (-0.74, -0.58)
    MilneAcid["D0"] = (0.90, 0.58)  # added this item from the JASMS paper
    MilneAcid["D+"] = (-0.90, -0.12)
    MilneAcid["E0"] = (-0.90, 0.31)  # added this item according to the JASMS paper
    MilneAcid["E+"] = (-0.60, -0.27)
    MilneAcid["F"] = (-0.52, -0.43)
    MilneAcid["G"] = (-0.22, 0.22)
    MilneAcid["H0"] = [0.00, 0.00]  # added this item according to the JASMS paper
    MilneAcid["H+"] = (-0.80, -0.51)  # added this item according to the JASMS paper
    MilneAcid["I"] = (-0.91, -0.59)
    MilneAcid["K"] = (-0.56, -0.29)
    MilneAcid["L"] = (-0.57, -0.13)
    MilneAcid["M"] = (-0.64, -0.28)
    MilneAcid["N"] = (-0.58, -0.13)
    MilneAcid["P"] = (0.00, -0.19)
    MilneAcid["Pc"] = (0.00, -0.85)
    MilneAcid["Q"] = (-0.47, -0.27)
    MilneAcid["R"] = (-0.59, -0.32)
    MilneAcid["S"] = (-0.44, -0.39)
    MilneAcid["T"] = (-0.79, -0.47)
    MilneAcid["V"] = (-0.74, -0.30)
    MilneAcid["W"] = (-0.40, -0.44)
    MilneAcid["Y"] = (-0.41, -0.37)

    # Dictionary for base values (format is (lambda, rho))
    MilneBase = {}

    # MilneBase["NTerminal"] = (None, RhoBaseNTerm)
    # MilneBase["CTerminal"] = (LambdaBaseCTerm, None)

    MilneBase["A"] = (0.00, 0.00)
    MilneBase["C"] = (0.62, 0.55)
    MilneBase["C2"] = (0.55, 0.46)
    MilneBase['D0'] = (0.10, -0.18)  # added this item according to the JASMS paper
    MilneBase["D+"] = (0.69, 0.60)
    MilneBase["E0"] = (-0.11, -0.15)  # added this item according to the JASMS paper
    MilneBase["E+"] = (0.24, 0.39)
    MilneBase["F"] = (-0.24, 0.06)
    MilneBase["G"] = (0.27, 0.17)  # old value
    # MilneBase["G"] = (-0.03, 0.17)  # changed this value according to the JASMS paper
    MilneBase["H0"] = (-0.10, 0.14)  # added this item according to the JASMS paper
    MilneBase["H+"] = (0.80, 0.83)  # added this item according to the JASMS paper
    MilneBase["I"] = (-0.73, -0.23)
    MilneBase["K"] = (-0.04, 0.12)
    MilneBase["L"] = (-0.58, -0.21)
    MilneBase["M"] = (-0.01, 0.11)
    MilneBase["N"] = (0.49, 0.32)
    MilneBase["P"] = (0.00, -0.24)
    MilneBase["Pc"] = (0.00, 0.60)
    MilneBase["Q"] = (0.06, 0.20)
    MilneBase["R"] = (0.08, 0.22)
    MilneBase["S"] = (0.37, 0.30)
    MilneBase["T"] = (-0.07, 0.20)
    MilneBase["V"] = (-0.70, -0.14)
    MilneBase["W"] = (-0.41, -0.11)
    MilneBase["Y"] = (-0.27, 0.05)

    # Default values
    MilneAcid["?"] = (0.00, 0.00)
    MilneBase["?"] = (0.00, 0.00)

    LambdaProtonatedAcidAsp = math.log10(
        10.0 ** (MilneAcid['D+'][0] - pH) / (10.0 ** -pKAsp + 10.0 ** -pH) + 10.0 ** (MilneAcid['D0'][0] - pKAsp) / (
                10.0 ** -pKAsp + 10.0 ** -pH))
    LambdaProtonatedAcidGlu = math.log10(
        10.0 ** (MilneAcid['E+'][0] - pH) / (10.0 ** -pKGlu + 10.0 ** -pH) + 10.0 ** (MilneAcid['E0'][0] - pKGlu) / (
                10.0 ** -pKGlu + 10.0 ** -pH))
    LambdaProtonatedAcidHis = math.log10(
        10.0 ** (MilneAcid['H+'][0] - pH) / (10.0 ** -pKHis + 10.0 ** -pH) + 10.0 ** (MilneAcid['H0'][0] - pKHis) / (
                10.0 ** -pKHis + 10.0 ** -pH))

    RhoProtonatedAcidAsp = math.log10(
        10.0 ** (MilneAcid['D+'][1] - pH) / (10.0 ** -pKAsp + 10.0 ** -pH) + 10.0 ** (MilneAcid['D0'][1] - pKAsp) / (
                10.0 ** -pKAsp + 10.0 ** -pH))
    RhoProtonatedAcidGlu = math.log10(
        10.0 ** (MilneAcid['E+'][1] - pH) / (10.0 ** -pKGlu + 10.0 ** -pH) + 10.0 ** (MilneAcid['E0'][1] - pKGlu) / (
                10.0 ** -pKGlu + 10.0 ** -pH))
    RhoProtonatedAcidHis = math.log10(
        10.0 ** (MilneAcid['H+'][1] - pH) / (10.0 ** -pKHis + 10.0 ** -pH) + 10.0 ** (MilneAcid['H0'][1] - pKHis) / (
                10.0 ** -pKHis + 10.0 ** -pH))

    LambdaProtonatedBaseAsp = math.log10(
        10.0 ** (MilneBase['D+'][0] - pH) / (10.0 ** -pKAsp + 10.0 ** -pH) + 10.0 ** (MilneBase['D0'][0] - pKAsp) / (
                10.0 ** -pKAsp + 10.0 ** -pH))
    LambdaProtonatedBaseGlu = math.log10(
        10.0 ** (MilneBase['E+'][0] - pH) / (10.0 ** -pKGlu + 10.0 ** -pH) + 10.0 ** (MilneBase['E0'][0] - pKGlu) / (
                10.0 ** -pKGlu + 10.0 ** -pH))
    LambdaProtonatedBaseHis = math.log10(
        10.0 ** (MilneBase['H+'][0] - pH) / (10.0 ** -pKHis + 10.0 ** -pH) + 10.0 ** (MilneBase['H0'][0] - pKHis) / (
                10.0 ** -pKHis + 10.0 ** -pH))

    RhoProtonatedBaseAsp = math.log10(
        10.0 ** (MilneBase['D+'][1] - pH) / (10.0 ** -pKAsp + 10.0 ** -pH) + 10.0 ** (MilneBase['D0'][1] - pKAsp) / (
                10.0 ** -pKAsp + 10.0 ** -pH))
    RhoProtonatedBaseGlu = math.log10(
        10.0 ** (MilneBase['E+'][1] - pH) / (10.0 ** -pKGlu + 10.0 ** -pH) + 10.0 ** (MilneBase['E0'][1] - pKGlu) / (
                10.0 ** -pKGlu + 10.0 ** -pH))
    RhoProtonatedBaseHis = math.log10(
        10.0 ** (MilneBase['H+'][1] - pH) / (10.0 ** -pKHis + 10.0 ** -pH) + 10.0 ** (MilneBase['H0'][1] - pKHis) / (
                10.0 ** -pKHis + 10.0 ** -pH))

    MilneAcid["D"] = (LambdaProtonatedAcidAsp, RhoProtonatedAcidAsp)
    MilneAcid["E"] = (LambdaProtonatedAcidGlu, RhoProtonatedAcidGlu)
    MilneAcid["H"] = (LambdaProtonatedAcidHis, RhoProtonatedAcidHis)

    MilneBase["D"] = (LambdaProtonatedBaseAsp, RhoProtonatedBaseAsp)
    MilneBase["E"] = (LambdaProtonatedBaseGlu, RhoProtonatedBaseGlu)
    MilneBase["H"] = (LambdaProtonatedBaseHis, RhoProtonatedBaseHis)

    # Termini
    RhoAcidNTerm = -1.32
    LambdaAcidCTerm = math.log10(10.0 ** (0.05 - pH) / (10.0 ** -pKGlu + 10.0 ** -pH) + 10.0 ** (0.96 - pKGlu) / (
            10.0 ** -pKGlu + 10.0 ** -pH))

    RhoBaseNTerm = 1.62
    LambdaBaseCTerm = -1.80

    MilneAcid["NT"] = (None, RhoAcidNTerm)
    MilneAcid["CT"] = (LambdaAcidCTerm, None)

    MilneBase["NT"] = (None, RhoBaseNTerm)
    MilneBase["CT"] = (LambdaBaseCTerm, None)

    # N terminal methylation
    LambdaAcidNMe = math.log10(135.5/(ka*60))
    LambdaBaseNMe = math.log10(2970000000/(kb*60))

    MilneAcid["NMe"] = (LambdaAcidNMe, None)
    MilneBase["NMe"] = (LambdaBaseNMe, None)

    # Acetylation
    MilneAcid["Ac"] = (None, 0.29)
    MilneBase["Ac"] = (None, -0.20)

    # Ion concentrations
    DIonConc = 10.0 ** -pH
    ODIonConc = 10.0 ** (pH - pKD)

    # Loop over the chain starting with 0 for initial residue
    IntrinsicEnchangeRates = [0.0]
    IntrinsicEnchangeRates_min = [0.0]

    # Account for middle residues
    for i in range(2, len(sequence) - 1):
        Residue = sequence[i]

        if Residue in ("P", "Pc"):
            IntrinsicEnchangeRates.append(0.0)

        else:
            # Identify neighbors
            LeftResidue = sequence[i - 1]
            RightResidue = sequence[i + 1]

            if RightResidue == "CT":
                Fa = 10.0 ** (MilneAcid[LeftResidue][1] + MilneAcid[Residue][0] + MilneAcid["CT"][0])
                Fb = 10.0 ** (MilneBase[LeftResidue][1] + MilneBase[Residue][0] + MilneBase["CT"][0])

            elif i == 2:
                Fa = 10.0 ** (MilneAcid["NT"][1] + MilneAcid[LeftResidue][1] + MilneAcid[Residue][0])
                Fb = 10.0 ** (MilneBase["NT"][1] + MilneBase[LeftResidue][1] + MilneBase[Residue][0])

            else:
                Fa = 10.0 ** (MilneAcid[LeftResidue][1] + MilneAcid[Residue][0])
                Fb = 10.0 ** (MilneBase[LeftResidue][1] + MilneBase[Residue][0])

            # Contributions from acid, base, and water

            Fta = math.exp(-1*AcidActivationEnergy*Temperature_Corr_2)
            Ftb = math.exp(-1*BaseActivationEnergy*Temperature_Corr_2)
            Ftw = math.exp(-1*SolventActivationEnergy*Temperature_Corr_2)

            kaT = Fa * ka * DIonConc * Fta
            kbT = Fb * kb * ODIonConc * Ftb
            kwT = Fb * kw * Ftw
            # kaT = Fa * ka * AcidTemperatureCorrection * DIonConc
            # kbT = Fb * kb * BaseTemperatureCorrection * ODIonConc
            # kwT = Fb * kw * SolventTemperatureCorrection

            # Collect exchange rates
            IntrinsicExchangeRate = kaT + kbT + kwT

            # To compare with the excel sheet from Englander Lab
            IntrinsicExchangeRate_min = IntrinsicExchangeRate * 60

            # Construct list
            IntrinsicEnchangeRates.append(IntrinsicExchangeRate)

            # To compare with the excel sheet from Englander lab
            IntrinsicEnchangeRates_min.append(IntrinsicExchangeRate_min)


    IntrinsicEnchangeRates = np.array(IntrinsicEnchangeRates)
    # IntrinsicEnchangeRates_min = np.array(IntrinsicEnchangeRates_min)

    return IntrinsicEnchangeRates


def PoiBin(success_probabilities):
    """
    poisson binomial probability distribution function
    :param success_probabilities:
    :return: probabiliy distribution
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


def gen_corr_backexchange(mass_rate_array, fix_backexchange_value):
    """

    :param mass_rate_array:
    :param fix_backexchange_value:
    :return:
    """
    backexchange_array = np.zeros(len(mass_rate_array))

    for ind, mass_rate in enumerate(mass_rate_array):
        corr_bkexch = (((1 - fix_backexchange_value)/(-1 - 0)) * mass_rate) + fix_backexchange_value
        backexchange_array[ind] = corr_bkexch

    return backexchange_array


def gen_backexchange_correction_from_backexchange_array(backexchange_array):
    """
    gen mass correction array from backexchange array
    :param backexchange_array:
    :return:
    """

    correction_array = np.zeros(len(backexchange_array))
    fix_backexchange_value = backexchange_array[-1]
    for ind, bk_val in enumerate(backexchange_array):
        mass_rate = ((bk_val - fix_backexchange_value) * (-1 - 0))/(1 - fix_backexchange_value)
        correction_array[ind] = mass_rate

    return correction_array


def estimate_gauss_param(ydata: np.ndarray,
                         xdata: np.ndarray,
                         baseline: float = 0.0,
                         width_ht: float = 0.7) -> list:
    ymax = np.max(ydata)
    maxindex = np.nonzero(ydata == ymax)[0]
    peakmax_x = xdata[maxindex][0]
    norm_arr = ydata/max(ydata)
    bins_for_width = norm_arr[norm_arr > width_ht]
    width_bin = len(bins_for_width)
    init_guess = [baseline, ymax, peakmax_x, width_bin]
    return init_guess


def gauss_func(x, y0, A, xc, w):
    """
    gaussian function with baseline
    :param x: xdata
    :param y0: baseline
    :param A: amplitude
    :param xc: centroid
    :param w: width
    :return: gauss(x)
    """
    rxc = ((x - xc) ** 2) / (2 * (w ** 2))
    y = y0 + A * (np.exp(-rxc))
    return y


def fit_gaussian(data: np.ndarray) -> object:
    """
    fit gaussian to data
    :param data: xdata to fit gaussian
    :return: gauss fit object
    """
    xdata = np.arange(len(data))
    guess_params = estimate_gauss_param(ydata=data,
                                        xdata=xdata)

    # initialize gauss fit object with fit success as false
    mean = sum(xdata * data) / sum(data)
    sigma = np.sqrt(sum(data * (xdata - mean) ** 2) / sum(data))
    gaussfit = GaussFit(fit_success=False,
                        gauss_fit_dist=data,
                        y_baseline=guess_params[0],
                        amplitude=guess_params[1],
                        centroid=center_of_mass_(data_array=data),
                        width=sigma,
                        r_sq=0.00,
                        rmse=100.0)

    try:

        # fit gaussian
        popt, pcov = curve_fit(gauss_func, xdata, data, p0=guess_params, maxfev=100000)

        # if the centroid is smaller than 0, return the false gaussfit object
        if popt[2] < 0.0:
            return gaussfit

        # if the width is smaller than 0, return the false gauss fit object
        if popt[3] < 0.0 or popt[3] > len(data):
            return gaussfit

        # for successful gaussian fit
        else:
            gaussfit.fit_success = True
            gaussfit.y_baseline = popt[0]
            gaussfit.amplitude = popt[1]
            gaussfit.centroid = popt[2]
            gaussfit.width = popt[3]
            gaussfit.gauss_fit_dist = gauss_func(xdata, *popt)
            gaussfit.rmse = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=data,
                                                          thr_isotope_dist=gaussfit.gauss_fit_dist,
                                                          squared=True)
            slope, intercept, rvalue, pvalue, stderr = linregress(data, gaussfit.gauss_fit_dist)
            gaussfit.r_sq = rvalue**2

            return gaussfit

    except RuntimeError:
        return gaussfit



def gauss_fit_to_isotope_dist_array(isotope_dist: np.ndarray) -> list:
    """
    fit gaussian to each isotope dist array and store the output in a list
    :param isotope_dist: 2d array of isotope dist
    :return: gauss fit list
    """

    gauss_fit_list = []

    for dist in isotope_dist:

        gauss_fit = fit_gaussian(data=dist)
        gauss_fit_list.append(gauss_fit)

    return gauss_fit_list


def convert_hxrate_object_to_dict(hxrate_object):

    # first convert the gauss fit objects to dicts
    exp_gauss_list = []
    for gauss_obj in hxrate_object.exp_data.gauss_fit:
        exp_gauss_list.append(vars(gauss_obj))

    # replace with the dict list
    hxrate_object.exp_data.gauss_fit = exp_gauss_list

    thr_gauss_list = []
    if hxrate_object.thr_isotope_dist_gauss_fit is not None:
        for gauss_obj_ in hxrate_object.thr_isotope_dist_gauss_fit:
            thr_gauss_list.append(vars(gauss_obj_))

    # replace with dict list
    hxrate_object.thr_isotope_dist_gauss_fit = thr_gauss_list

    # convert expdata object to dict
    hxrate_object.exp_data = vars(hxrate_object.exp_data)

    # convert backexchange object to dict
    hxrate_object.back_exchange = vars(hxrate_object.back_exchange)

    # convert merge object to dict
    hxrate_object.merge_data = vars(hxrate_object.merge_data)

    # convert hxrate_pipeline object to dict
    hxrate_dict = vars(hxrate_object)

    return hxrate_dict


def make_new_dir(dirpath):
    """
    make a new directory if the directory doesn't already exists
    :param dirpath: directory path
    :return: directory path
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


def gen_temp_rates(sequence: str, rate_value: float = 1e2) -> np.ndarray:
    """
    generate template rates
    :param sequence: protein sequence
    :param rate_value: temporary rate value
    :return: an array of rates with first two residues and proline residues assigned to 0.0
    """
    rates = np.array([rate_value]*len(sequence), dtype=float)

    # set the rates for the first two residues as 0
    rates[:2] = 0

    # set the rate for proline to be 0
    if 'P' in sequence:
        amino_acid_list = [x for x in sequence]
        for ind, amino_acid in enumerate(amino_acid_list):
            if amino_acid == 'P':
                rates[ind] = 0

    return rates


def center_of_mass_(data_array):
    com = center_of_mass(data_array)[0]
    return com


def correct_centroids_using_backexchange(centroids: np.ndarray,
                                         backexchange_array: np.ndarray,
                                         include_zero_dist: bool = False) -> np.ndarray:
    """

    :param centroids: uncorrected centroids
    :param backexchange_array: backexchange array for each timepoint
    :return: corrected centroids
    """

    # generate corr centroids zero arrays and fill with corrections. timepoint 0 doesn't need any correction
    corr_centroids = np.zeros(len(centroids))

    # apply correction to centroids based on backexchange
    for ind, (centr, bkexch) in enumerate(zip(centroids, backexchange_array)):
        corr_centroids[ind] = centr/(1-bkexch)

    if include_zero_dist:
        corr_centroids[0] = centroids[0]

    return corr_centroids


# @jit(parallel=True)
def normalize_mass_distribution_array(mass_dist_array: np.ndarray) -> np.ndarray:
    norm_dist = np.zeros(np.shape(mass_dist_array))
    for ind, dist in enumerate(mass_dist_array):
        norm_ = dist/max(dist)
        norm_dist[ind] = norm_
    return norm_dist


def theoretical_isotope_dist(sequence, num_isotopes=None):
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


# @jit(nopython=True)
def calc_hx_prob(timepoint: float,
                 rate_constant: np.ndarray,
                 inv_back_exchange: float,
                 d2o_purity: float,
                 d2o_fraction: float) -> np.ndarray:
    """
    calculate the exchange probability for each residue at the timepoint given the rate constant, backexchange, d2o purity and fraction
    :param timepoint: timepoint in seconds
    :param rate_constant: array of rate_constant
    :param inv_back_exchange: 1 - backexchange value
    :param d2o_purity: d2o purity
    :param d2o_fraction: d2o fraction
    :return: array of probabilities
    """

    prob = (1.0 - np.exp(-rate_constant * timepoint)) * (d2o_fraction * d2o_purity * inv_back_exchange)
    return prob


# @jit(nopython=True)
def hx_rates_probability_distribution(timepoint: float,
                                      rates: np.ndarray,
                                      inv_backexchange: float,
                                      d2o_fraction: float,
                                      d2o_purity: float,
                                      free_energy_values: np.ndarray = None,
                                      temperature: float = None) -> np.ndarray:
    """
    generate rate of hx probabilities for all residues
    :param timepoint: hdx timepoint in seconds
    :param rates: rates
    :param inv_backexchange: 1 - backexchange
    :param d2o_fraction: d2o fractyion
    :param d2o_purity: d2o purity
    :param free_energy_values: free energy values
    :param temperature: temperature
    :return: hx probablities of all resdiues given the hdx rates
    """

    fractions = np.array([1 for x in rates])
    if free_energy_values is not None:
        if temperature is None:
            raise ValueError('You need to specify temperature (K) in order to use free energy values')
        else:
            fractions = np.exp(-free_energy_values / (r_constant * temperature)) / (1.0 + np.exp(-free_energy_values / (r_constant * temperature)))

    rate_constant_values = rates * fractions

    hx_probabs = calc_hx_prob(timepoint=timepoint,
                              rate_constant=rate_constant_values,
                              inv_back_exchange=inv_backexchange,
                              d2o_purity=d2o_purity,
                              d2o_fraction=d2o_fraction)

    return hx_probabs


def isotope_dist_from_PoiBin(sequence: str,
                             timepoint: float,
                             inv_backexchange: float,
                             rates: np.ndarray,
                             d2o_fraction: float,
                             d2o_purity: float,
                             num_bins: float,
                             free_energy_values: np.ndarray = None,
                             temperature: float = None) -> np.ndarray:
    """
    generate theoretical isotope distribution based on hdx rates, timepoint, and other conditions
    :param sequence: protein sequence str
    :param timepoint: timepoint in seconds
    :param inv_backexchange: 1 - backexchange
    :param rates: hdx rates
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param num_bins: number of bins to include for isotope distribution
    :param free_energy_values: free energy values to be used for calculating isotope distribution
    :param temperature: temperature to be used when free energy is not None
    :return: isotope distribution normalized
    """

    hx_probs = hx_rates_probability_distribution(timepoint=timepoint,
                                                     rates=rates,
                                                     inv_backexchange=inv_backexchange,
                                                     d2o_fraction=d2o_fraction,
                                                     d2o_purity=d2o_purity,
                                                     free_energy_values=free_energy_values,
                                                     temperature=temperature)

    pmf_hx_probs = PoiBin(hx_probs)

    seq_isotope_dist = theoretical_isotope_dist(sequence=sequence, num_isotopes=num_bins)

    isotope_dist_poibin = np.convolve(pmf_hx_probs, seq_isotope_dist)[:num_bins]
    isotope_dist_poibin_norm = isotope_dist_poibin/max(isotope_dist_poibin)

    return isotope_dist_poibin_norm


# @jit(parallel=True)
def gen_theoretical_isotope_dist_for_all_timepoints(sequence: str,
                                                    timepoints: np.ndarray,
                                                    rates: np.ndarray,
                                                    inv_backexchange_array: np.ndarray,
                                                    d2o_fraction: float,
                                                    d2o_purity: float,
                                                    num_bins: int,
                                                    free_energy_values: np.ndarray = None,
                                                    temperature: float = None) -> np.ndarray:
    """

    :param sequence: protein sequence
    :param timepoints: array of hdx timepoints
    :param rates: rates
    :param inv_backexchange_array: inv backexchange array with length equals to length of timepoints
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param num_bins: number of bins for isotope distribution
    :param free_energy_values: free energy values array (optional)
    :param temperature: temperature in Kelvin (optional)
    :return: array of theoretical isotope distributions for each timepoint
    """

    out_array = np.zeros((len(timepoints), num_bins))

    for ind, (tp, inv_backexch) in enumerate(zip(timepoints, inv_backexchange_array)):

        isotope_dist = isotope_dist_from_PoiBin(sequence=sequence,
                                                timepoint=tp,
                                                inv_backexchange=inv_backexch,
                                                rates=rates,
                                                d2o_fraction=d2o_fraction,
                                                d2o_purity=d2o_purity,
                                                num_bins=num_bins,
                                                free_energy_values=free_energy_values,
                                                temperature=temperature)
        out_array[ind] = isotope_dist

    return out_array


def compute_rmse_exp_thr_iso_dist(exp_isotope_dist: np.ndarray,
                                  thr_isotope_dist: np.ndarray,
                                  squared: bool = False) -> float:
    """
    compute the mean squared error between exp and thr isotope distribution only with values of exp_dist that are higher
    than 0
    :param exp_isotope_dist: exp isotope dist numpy 2d array
    :param thr_isotope_dist: thr isotope dist numpy 2d array
    :param squared: boolean. Default False
    :return: rmse
    """
    exp_isotope_dist_comp = exp_isotope_dist[exp_isotope_dist > 0]
    thr_isotope_dist_comp = thr_isotope_dist[exp_isotope_dist > 0]
    rmse = mean_squared_error(exp_isotope_dist_comp, thr_isotope_dist_comp, squared=squared)
    return rmse



# @jit(parallel=True)
def mse_exp_thr_isotope_dist_all_timepoints(exp_isotope_dist_array: np.ndarray,
                                             sequence: str,
                                             timepoints: np.ndarray,
                                             inv_backexchange_array: np.ndarray,
                                             rates: np.ndarray,
                                             d2o_fraction: float,
                                             d2o_purity: float,
                                             num_bins: int,
                                             free_energy_values: np.ndarray = None,
                                             temperature: float = None) -> float:
    """

    :param exp_isotope_dist_array:
    :param sequence:
    :param timepoints:
    :param inv_backexchange_array:
    :param rates:
    :param d2o_fraction:
    :param d2o_purity:
    :param num_bins:
    :param free_energy_values:
    :param temperature:
    :return:
    """

    theoretical_isotope_dist_all_timepoints = gen_theoretical_isotope_dist_for_all_timepoints(sequence=sequence,
                                                                                              timepoints=timepoints,
                                                                                              rates=np.exp(rates),
                                                                                              inv_backexchange_array=inv_backexchange_array,
                                                                                              d2o_fraction=d2o_fraction,
                                                                                              d2o_purity=d2o_purity,
                                                                                              num_bins=num_bins,
                                                                                              free_energy_values=free_energy_values,
                                                                                              temperature=temperature)

    exp_isotope_dist_concat = np.concatenate(exp_isotope_dist_array)
    thr_isotope_dist_concat = np.concatenate(theoretical_isotope_dist_all_timepoints)
    thr_isotope_dist_concat[np.isnan(thr_isotope_dist_concat)] = 0

    rmse = compute_rmse_exp_thr_iso_dist(exp_isotope_dist_concat, thr_isotope_dist_concat, squared=False)

    return rmse


def basin_hop_print_func(x, f, accepted):
    print('%s,%.4f,%d' % (psutil.Process().pid, f, int(accepted)))

def basin_hop_header_print_func():
    print('\nprocess_id,function_value,accepted')


def hx_rate_fitting_optimization(init_rate_guess: np.ndarray,
                                 exp_isotope_dist_array: np.ndarray,
                                 sequence: str,
                                 timepoints: np.ndarray,
                                 inv_backexchange_array: np.ndarray,
                                 d2o_fraction: float,
                                 d2o_purity: float,
                                 num_bins: int,
                                 free_energy_values: np.ndarray,
                                 temperature: float,
                                 opt_iter: int,
                                 opt_temp: float,
                                 opt_step_size: float,
                                 return_tuple: bool = False) -> object:
    """
    rate fitting otimization routine
    :param exp_isotope_dist_array: experimental isotope distribution array
    :param sequence: protein sequence
    :param timepoints: timepoints array
    :param inv_backexchange_array: inverse backexchange array
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param num_bins: number of bins to include for theoretical isotope distribution
    :param free_energy_values: free energy values
    :param temperature: temperature in K
    :param init_rate_guess: initial rate guess array
    :param opt_iter: # of basinhopping optimization iterations
    :param opt_temp: optimization temperature
    :param opt_step_size: optimization step size
    :param return_tuple: bool. if True returns a tuple of (opt_, init_rate_guess), else returns only opt_
    :return: optimization object
    """

    minimizer_kwargs_ = {'method': 'BFGS',
                         'options': {'maxiter': 1,
                                     'disp': False}}

    opt_ = basinhopping(lambda rates: mse_exp_thr_isotope_dist_all_timepoints(exp_isotope_dist_array=exp_isotope_dist_array,
                                                                              sequence=sequence,
                                                                              timepoints=timepoints,
                                                                              rates=rates,
                                                                              inv_backexchange_array=inv_backexchange_array,
                                                                              d2o_fraction=d2o_fraction,
                                                                              d2o_purity=d2o_purity,
                                                                              num_bins=num_bins,
                                                                              free_energy_values=free_energy_values,
                                                                              temperature=temperature),
                        x0=init_rate_guess,
                        niter=opt_iter,
                        T=opt_temp,
                        stepsize=opt_step_size,
                        minimizer_kwargs=minimizer_kwargs_,
                        disp=False,
                        callback=basin_hop_print_func)

    if return_tuple:
        return opt_, init_rate_guess
    else:
        return opt_


def plot_hx_rate_fitting_(prot_name: str,
                          hx_rates: np.ndarray,
                          exp_isotope_dist: np.ndarray,
                          thr_isotope_dist: np.ndarray,
                          exp_isotope_centroid_array: np.ndarray,
                          thr_isotope_centroid_array: np.ndarray,
                          exp_isotope_width_array: np.ndarray,
                          thr_isotope_width_array: np.ndarray,
                          timepoints: np.ndarray,
                          fit_rmse_timepoints: np.ndarray,
                          fit_rmse_total: float,
                          backexchange: float,
                          backexchange_array: np.ndarray,
                          d2o_fraction: float,
                          d2o_purity: float,
                          output_path: str):
    """
    generate several plots for visualizing the hx rate fitting output
    :param prot_name: protein name
    :param hx_rates: in ln scale
    :param exp_isotope_dist: exp isotope dist array
    :param thr_isotope_dist: thr isotope dist array from hx rates
    :param exp_isotope_centroid_array: exp isotope centroid in an array
    :param thr_isotope_centroid_array: thr isotope centroid in an array
    :param exp_isotope_width_array: exp isotope width in an array
    :param thr_isotope_width_array: thr isotope width in an array
    :param timepoints: time points
    :param fit_rmse_timepoints: fit rmse for each timepoint
    :param fit_rmse_total: total fit rmse
    :param backexchange: backexchange value
    :param backexchange_array: backexchange array
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param output_path: plot saveing output path
    :return:
    """

    # define figure size
    num_columns = 2
    num_rows = len(timepoints)
    fig_size = (25, 1.0 * num_rows)

    font_size = 10

    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(nrows=num_rows, ncols=num_columns)

    plt.rcParams.update({'font.size': font_size})

    if fit_rmse_timepoints is None:
        fit_rmse_tp = np.zeros(len(timepoints))
    else:
        fit_rmse_tp = fit_rmse_timepoints

    #######################################################
    #######################################################
    # start plotting the exp and thr isotope dist
    for num, (timepoint, exp_dist, thr_dist, exp_centroid, bkexch) in enumerate(zip(timepoints, exp_isotope_dist, thr_isotope_dist, exp_isotope_centroid_array, backexchange_array)):

        if fit_rmse_timepoints is None:
            rmse_tp = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=exp_dist,
                                                    thr_isotope_dist=thr_dist,
                                                    squared=False)
            fit_rmse_tp[num] = rmse_tp
        else:
            rmse_tp = fit_rmse_tp[num]

        # plot exp and thr isotope dist
        ax = fig.add_subplot(gs[num, 0])
        plt.plot(exp_dist, color='blue', marker='o', ls='-', markersize=3)
        plt.plot(thr_dist, color='red')
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        plt.xticks(range(0, len(exp_dist) + 5, 5))
        ax.set_xticklabels(range(0, len(exp_dist) + 5, 5))
        plt.grid(axis='x', alpha=0.25)
        ax.tick_params(length=3, pad=3)

        # put the rmse on the right side of the plot and delta centroid
        if num == 0:
            delta_centroid = 0.0
        else:
            prev_centroid = exp_isotope_centroid_array[num - 1]
            delta_centroid = exp_centroid - prev_centroid

        # delta_centroid_text = 'dmz=%.2f' % delta_centroid

        plt.text(1.0, 1.2, "fit rmse = %.4f\nd_mz = %.2f\nbkexch = %.2f" % (rmse_tp, delta_centroid, bkexch*100),
                 horizontalalignment="right",
                 verticalalignment="top",
                 transform=ax.transAxes)

        # put timepoint on  the left side of the plot
        plt.text(0.01, 1.2, '%s t %i' % (num, timepoint),
                 horizontalalignment="left",
                 verticalalignment="top",
                 transform=ax.transAxes)

        # put the centroid information by the peak max
        plt.text(
            exp_centroid,
            1.1,
            "%.1f" % exp_centroid,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8)
    #######################################################
    #######################################################

    # 8 plots on the second row
    num_plots_second_row = 8
    second_plot_row_thickness = int(len(timepoints)/num_plots_second_row)
    second_plot_indices = [(num*second_plot_row_thickness) for num in range(num_plots_second_row)]

    #######################################################
    #######################################################
    # plot timepoint specific backexchange
    ax0 = fig.add_subplot(gs[second_plot_indices[0]: second_plot_indices[1], 1])

    plt.scatter(x=np.arange(len(timepoints))[1:], y=backexchange_array[1:]*100, color='black')
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    plt.xticks(range(-1, len(timepoints) + 1, 1))
    ax0.set_xticklabels(range(-1, len(timepoints) + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Back Exchange (%)')
    ax0.tick_params(length=3, pad=3)

    #######################################################
    #######################################################
    # plot fit rmse

    ax1 = fig.add_subplot(gs[second_plot_indices[1]: second_plot_indices[2], 1])
    plt.scatter(np.arange(len(timepoints)), fit_rmse_tp, color='black')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax1.set_xticklabels(range(0, len(timepoints) + 1, 1))
    if max(fit_rmse_tp) <= 0.15:
        y_ticks = np.round(np.linspace(0, 0.15, num=16), 2)
        plt.yticks(y_ticks)
        ax1.set_yticklabels(y_ticks)
    else:
        plt.axhline(y=0.15, ls='--', color='black')
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Fit RMSE')
    ax1.tick_params(length=3, pad=3)

    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot center of mass exp and thr

    timepoints_v2 = np.array([x for x in timepoints])
    timepoints_v2[0] = timepoints_v2[2] - timepoints_v2[1]

    ax2 = fig.add_subplot(gs[second_plot_indices[2]: second_plot_indices[3], 1])
    ax2.plot(timepoints_v2, exp_isotope_centroid_array, marker='o', ls='-', color='blue')
    ax2.plot(timepoints_v2, thr_isotope_centroid_array, marker='o', ls='-', color='red')
    ax2.set_xscale('log')
    ax2.set_xticks(timepoints_v2)
    ax2.set_xticklabels([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('log(timepoint)')
    plt.ylabel('Centroid')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot center of mass exp and thr corrected using backexchange

    exp_isotope_centroid_array_corr = correct_centroids_using_backexchange(centroids=exp_isotope_centroid_array,
                                                                           backexchange_array=backexchange_array,
                                                                           include_zero_dist=True)
    thr_isotope_centroid_array_corr = correct_centroids_using_backexchange(centroids=thr_isotope_centroid_array,
                                                                           backexchange_array=backexchange_array,
                                                                           include_zero_dist=True)

    ax2 = fig.add_subplot(gs[second_plot_indices[3]: second_plot_indices[4], 1])
    ax2.plot(timepoints_v2, exp_isotope_centroid_array_corr, marker='o', ls='-', color='blue')
    ax2.plot(timepoints_v2, thr_isotope_centroid_array_corr, marker='o', ls='-', color='red')
    ax2.set_xscale('log')
    ax2.set_xticks(timepoints_v2)
    ax2.set_xticklabels([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('log(timepoint)')
    plt.ylabel('Corrected Centroid')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the error in center of mass between exp and thr distributions

    com_difference = np.subtract(exp_isotope_centroid_array_corr, thr_isotope_centroid_array_corr)

    ax3 = fig.add_subplot(gs[second_plot_indices[4]: second_plot_indices[5], 1])
    ax3.scatter(np.arange(len(timepoints)), com_difference, color='black')
    plt.axhline(y=0, ls='--', color='black', alpha=0.50)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax3.set_xticklabels(range(0, len(timepoints) + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Centroid difference (E-T)')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the width of exp and thr distributions

    ax4 = fig.add_subplot(gs[second_plot_indices[5]: second_plot_indices[6], 1])
    ax4.scatter(np.arange(len(timepoints)), exp_isotope_width_array, color='blue')
    ax4.scatter(np.arange(len(timepoints)), thr_isotope_width_array, color='red')
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax4.set_xticklabels(range(0, len(timepoints) + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Width')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the error in center of mass between exp and thr distributions

    width_difference = np.subtract(exp_isotope_width_array, thr_isotope_width_array)

    ax5 = fig.add_subplot(gs[second_plot_indices[6]: second_plot_indices[7], 1])
    ax5.scatter(np.arange(len(timepoints)), width_difference, color='black')
    plt.axhline(y=0, ls='--', color='black', alpha=0.50)
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax5.set_xticklabels(range(0, len(timepoints) + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Width difference (E-T)')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the rates in log10 scale
    hx_rates_log10 = np.log10(np.exp(hx_rates))

    ax6 = fig.add_subplot(gs[second_plot_indices[7]:, 1])
    plt.plot(np.arange(len(hx_rates_log10)), np.sort(hx_rates_log10), marker='o', ls='-', color='red',
             markerfacecolor='red', markeredgecolor='black')
    plt.xticks(range(0, len(hx_rates_log10) + 2, 2))
    ax6.set_xticklabels(range(0, len(hx_rates_log10) + 2, 2))
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Residues (Ranked from slowest to fastest exchanging)')
    plt.ylabel('Rate: log k (1/s)')
    #######################################################
    #######################################################

    # adjust some plot properties and add title
    plt.subplots_adjust(hspace=1.2, wspace=0.1, top=0.96)

    title_1 = 'Fit RMSE: %.4f | Backexchange: %.2f %% | D2O Purity: %.1f %% | D2O_Fraction: %.1f %%' %(fit_rmse_total,
                                                                                                       backexchange*100,
                                                                                                       d2o_purity*100,
                                                                                                       d2o_fraction*100)

    plot_title = prot_name + ' (' + title_1 + ')'

    plt.suptitle(plot_title)

    plt.figtext(0.498, 0.968, "EXP", color='blue', ha='right', fontsize=8)
    plt.figtext(0.502, 0.968, "FIT", color='red', ha='left', fontsize=8)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()



def plot_hx_rate_fitting_bayes(prot_name: str,
                               hx_rates: np.ndarray,
                               hx_rates_error: np.ndarray,
                               exp_isotope_dist: np.ndarray,
                               thr_isotope_dist: np.ndarray,
                               exp_isotope_centroid_array: np.ndarray,
                               thr_isotope_centroid_array: np.ndarray,
                               exp_isotope_width_array: np.ndarray,
                               thr_isotope_width_array: np.ndarray,
                               timepoints: np.ndarray,
                               fit_rmse_timepoints: np.ndarray,
                               fit_rmse_total: float,
                               backexchange: float,
                               backexchange_array: np.ndarray,
                               d2o_fraction: float,
                               d2o_purity: float,
                               output_path: str):
    """
    generate several plots for visualizing the hx rate fitting output
    :param prot_name: protein name
    :param hx_rates: in ln scale
    :param exp_isotope_dist: exp isotope dist array
    :param thr_isotope_dist: thr isotope dist array from hx rates
    :param exp_isotope_centroid_array: exp isotope centroid in an array
    :param thr_isotope_centroid_array: thr isotope centroid in an array
    :param exp_isotope_width_array: exp isotope width in an array
    :param thr_isotope_width_array: thr isotope width in an array
    :param timepoints: time points
    :param fit_rmse_timepoints: fit rmse for each timepoint
    :param fit_rmse_total: total fit rmse
    :param backexchange: backexchange value
    :param backexchange_array: backexchange array
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param output_path: plot saveing output path
    :return:
    """

    # define figure size
    num_columns = 2

    # this is a constant
    num_plots_second_row = 8

    num_rows = len(timepoints)
    row_width = 1.0
    if len(timepoints) < num_plots_second_row:
        num_rows = num_plots_second_row
        row_width = 2.5
    fig_size = (25, row_width * num_rows)

    font_size = 10

    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(nrows=num_rows, ncols=num_columns)

    plt.rcParams.update({'font.size': font_size})

    if fit_rmse_timepoints is None:
        fit_rmse_tp = np.zeros(len(timepoints))
    else:
        fit_rmse_tp = fit_rmse_timepoints

    #######################################################
    #######################################################
    # start plotting the exp and thr isotope dist
    for num, (timepoint, exp_dist, thr_dist, exp_centroid, bkexch) in enumerate(zip(timepoints, exp_isotope_dist, thr_isotope_dist, exp_isotope_centroid_array, backexchange_array)):

        if fit_rmse_timepoints is None:
            rmse_tp = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=exp_dist,
                                                    thr_isotope_dist=thr_dist,
                                                    squared=False)
            fit_rmse_tp[num] = rmse_tp
        else:
            rmse_tp = fit_rmse_tp[num]

        # plot exp and thr isotope dist
        ax = fig.add_subplot(gs[num, 0])
        plt.plot(exp_dist, color='blue', marker='o', ls='-', markersize=3)
        plt.plot(thr_dist, color='red')
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        plt.xticks(range(0, len(exp_dist) + 5, 5))
        ax.set_xticklabels(range(0, len(exp_dist) + 5, 5))
        plt.grid(axis='x', alpha=0.25)
        ax.tick_params(length=3, pad=3)

        # put the rmse on the right side of the plot and delta centroid
        if num == 0:
            delta_centroid = 0.0
        else:
            prev_centroid = exp_isotope_centroid_array[num - 1]
            delta_centroid = exp_centroid - prev_centroid

        # delta_centroid_text = 'dmz=%.2f' % delta_centroid

        plt.text(1.0, 1.2, "fit rmse = %.4f\nd_mz = %.2f\nbkexch = %.2f" % (rmse_tp, delta_centroid, bkexch*100),
                 horizontalalignment="right",
                 verticalalignment="top",
                 transform=ax.transAxes)

        # put timepoint on  the left side of the plot
        plt.text(0.01, 1.2, '%s t %i' % (num, timepoint),
                 horizontalalignment="left",
                 verticalalignment="top",
                 transform=ax.transAxes)

        # put the centroid information by the peak max
        plt.text(
            exp_centroid,
            1.1,
            "%.1f" % exp_centroid,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8)
    #######################################################
    #######################################################

    # 8 plots on the second row
    second_plot_row_thickness = int(len(timepoints)/num_plots_second_row)
    if len(timepoints) < num_plots_second_row:
        second_plot_row_thickness = 1
    second_plot_indices = [(num*second_plot_row_thickness) for num in range(num_plots_second_row)]

    #######################################################
    #######################################################
    # plot timepoint specific backexchange
    ax0 = fig.add_subplot(gs[second_plot_indices[0]: second_plot_indices[1], 1])

    plt.scatter(x=np.arange(len(timepoints))[1:], y=backexchange_array[1:]*100, color='black')
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    plt.xticks(range(-1, len(timepoints) + 1, 1))
    ax0.set_xticklabels(range(-1, len(timepoints) + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Back Exchange (%)')
    ax0.tick_params(length=3, pad=3)

    #######################################################
    #######################################################
    # plot fit rmse

    ax1 = fig.add_subplot(gs[second_plot_indices[1]: second_plot_indices[2], 1])
    plt.scatter(np.arange(len(timepoints)), fit_rmse_tp, color='black')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax1.set_xticklabels(range(0, len(timepoints) + 1, 1))
    if max(fit_rmse_tp) <= 0.15:
        y_ticks = np.round(np.linspace(0, 0.15, num=16), 2)
        plt.yticks(y_ticks)
        ax1.set_yticklabels(y_ticks)
    else:
        plt.axhline(y=0.15, ls='--', color='black')
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Fit RMSE')
    ax1.tick_params(length=3, pad=3)

    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot center of mass exp and thr

    timepoints_v2 = np.array([x for x in timepoints])
    timepoints_v2[0] = timepoints_v2[2] - timepoints_v2[1]

    ax2 = fig.add_subplot(gs[second_plot_indices[2]: second_plot_indices[3], 1])
    ax2.plot(timepoints_v2, exp_isotope_centroid_array, marker='o', ls='-', color='blue')
    ax2.plot(timepoints_v2, thr_isotope_centroid_array, marker='o', ls='-', color='red')
    ax2.set_xscale('log')
    ax2.set_xticks(timepoints_v2)
    ax2.set_xticklabels([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('log(timepoint)')
    plt.ylabel('Centroid')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot center of mass exp and thr corrected using backexchange

    exp_isotope_centroid_array_corr = correct_centroids_using_backexchange(centroids=exp_isotope_centroid_array,
                                                                           backexchange_array=backexchange_array,
                                                                           include_zero_dist=True)
    thr_isotope_centroid_array_corr = correct_centroids_using_backexchange(centroids=thr_isotope_centroid_array,
                                                                           backexchange_array=backexchange_array,
                                                                           include_zero_dist=True)

    ax2 = fig.add_subplot(gs[second_plot_indices[3]: second_plot_indices[4], 1])
    ax2.plot(timepoints_v2, exp_isotope_centroid_array_corr, marker='o', ls='-', color='blue')
    ax2.plot(timepoints_v2, thr_isotope_centroid_array_corr, marker='o', ls='-', color='red')
    ax2.set_xscale('log')
    ax2.set_xticks(timepoints_v2)
    ax2.set_xticklabels([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('log(timepoint)')
    plt.ylabel('Corrected Centroid')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the error in center of mass between exp and thr distributions

    com_difference = np.subtract(exp_isotope_centroid_array_corr, thr_isotope_centroid_array_corr)

    ax3 = fig.add_subplot(gs[second_plot_indices[4]: second_plot_indices[5], 1])
    ax3.scatter(np.arange(len(timepoints)), com_difference, color='black')
    plt.axhline(y=0, ls='--', color='black', alpha=0.50)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax3.set_xticklabels(range(0, len(timepoints) + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Centroid difference (E-T)')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the width of exp and thr distributions

    ax4 = fig.add_subplot(gs[second_plot_indices[5]: second_plot_indices[6], 1])
    ax4.scatter(np.arange(len(timepoints)), exp_isotope_width_array, color='blue')
    ax4.scatter(np.arange(len(timepoints)), thr_isotope_width_array, color='red')
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax4.set_xticklabels(range(0, len(timepoints) + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Width')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the error in center of mass between exp and thr distributions

    width_difference = np.subtract(exp_isotope_width_array, thr_isotope_width_array)

    ax5 = fig.add_subplot(gs[second_plot_indices[6]: second_plot_indices[7], 1])
    ax5.scatter(np.arange(len(timepoints)), width_difference, color='black')
    plt.axhline(y=0, ls='--', color='black', alpha=0.50)
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax5.set_xticklabels(range(0, len(timepoints) + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Width difference (E-T)')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the rates in log10 scale
    hx_rates_log10 = np.log10(np.exp(hx_rates))
    hx_rates_error_log10 = np.log10(np.exp(hx_rates_error))

    ax6 = fig.add_subplot(gs[second_plot_indices[7]:, 1])
    plt.errorbar(x=np.arange(len(hx_rates_log10)), y=hx_rates_log10, yerr=hx_rates_error_log10, marker='o', ls='-',
                 color='red', markerfacecolor='red', markeredgecolor='black')
    # plt.plot(np.arange(len(hx_rates_log10)), hx_rates_log10[sort_ind], marker='o', ls='-', color='red',
    #          markerfacecolor='red', markeredgecolor='black')
    plt.xticks(range(0, len(hx_rates_log10) + 2, 2))
    ax6.set_xticklabels(range(0, len(hx_rates_log10) + 2, 2))
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Residues (Ranked from slowest to fastest exchanging)')
    plt.ylabel('Rate: log k (1/s)')
    #######################################################
    #######################################################

    # adjust some plot properties and add title
    plt.subplots_adjust(hspace=1.2, wspace=0.1, top=0.96)

    title_1 = 'Fit RMSE: %.4f | Backexchange: %.2f %% | D2O Purity: %.1f %% | D2O_Fraction: %.1f %%' %(fit_rmse_total,
                                                                                                       backexchange*100,
                                                                                                       d2o_purity*100,
                                                                                                       d2o_fraction*100)

    plot_title = prot_name + ' (' + title_1 + ')'

    plt.suptitle(plot_title)

    plt.figtext(0.498, 0.968, "EXP", color='blue', ha='right', fontsize=8)
    plt.figtext(0.502, 0.968, "FIT", color='red', ha='left', fontsize=8)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_hx_rates(hx_rates: np.ndarray,
                  output_path: str,
                  log_rates: bool = True):
    """
    plot the hx rates in increasing order
    :param hx_rates:
    :param output_path:
    :param log_rates: If True, plot the rates in log10 scale. If not, plots in ln scale.
    :return:
    """

    y_label = 'ln k (1/s)'

    if log_rates:
        hx_rates = np.log10(np.exp(hx_rates))
        y_label = 'log10 k (1/s)'

    fig, ax = plt.subplots()
    plt.plot(np.arange(len(hx_rates)), np.sort(hx_rates), marker='o', ls='-', color='black', markerfacecolor='red')
    plt.xticks(range(0, len(hx_rates) + 2, 2))
    ax.set_xticklabels(range(0, len(hx_rates) + 2, 2), fontsize=8)
    plt.grid(axis='x', alpha=0.25)
    plt.xlabel('Residues (Ranked from slowest to fastest exchanging)')
    plt.ylabel(y_label)
    plt.subplots_adjust(hspace=0.5, wspace=0.1, top=0.95)
    plt.savefig(output_path)
    plt.close()


def plot_exp_thr_dist(exp_dist_array: np.ndarray,
                      thr_dist_array: np.ndarray,
                      timepoints_array: np.ndarray,
                      backexchange: float,
                      output_path: str,
                      rmse_each_timepoint: np.ndarray = None,
                      total_rmse: float = None):
    """
    plot the experimental and theoretical isotope distribution at each time point
    :param exp_dist_array: exp dist array
    :param thr_dist_array: thr dist array
    :param timepoints_array: timepoints array
    :param backexchange: backexchange float value
    :param rmse_each_timepoint: rmse at each timepoint. if None, will compute
    :param total_rmse: total rmse. if None, will compute
    :param output_path: output path
    :return:
    """

    num_columns = 0
    num_rows = 5

    for num in range(len(exp_dist_array)):
        if num % num_rows == 0:
            num_columns += 1

    fig = plt.figure(figsize=(num_columns * 3, num_rows * 3.0))
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
        plt.xlabel('Added mass units')
        plt.grid(axis='x', alpha=0.25)
        ax.tick_params(length=3, pad=3)
        plt.legend(loc='best', fontsize='small')
        plt.title('timepoint %i' % timepoint)

        if (n_rows+1) % num_rows == 0:
            n_cols += 1
            n_rows = 0
        else:
            n_rows += 1

    if total_rmse is None:
        exp_isotope_dist_concat = np.concatenate(exp_dist_array)
        thr_isotope_dist_concat = np.concatenate(thr_dist_array)
        thr_isotope_dist_concat[np.isnan(thr_isotope_dist_concat)] = 0
        total_rmse = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=exp_isotope_dist_concat,
                                                   thr_isotope_dist=thr_isotope_dist_concat,
                                                   squared=False)

    plot_title = 'EXP vs THR Isotope Distribution (Fit RMSE: %.4f | BACKEXCHANGE: %.2f)' % (total_rmse, backexchange*100)
    plt.suptitle(plot_title)
    plt.subplots_adjust(hspace=0.5, wspace=0.1, top=0.95)
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':

    pass
