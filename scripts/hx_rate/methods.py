import numpy as np
import math
import molmass
from scipy.optimize import fmin_powell
from scipy.special import expit
from sklearn.metrics import mean_squared_error


def calc_intrinsic_hx_rates(sequence_str: str,
                            temperature: float,
                            ph: float,
                            nterm_mode: str = 'NT',
                            cterm_mode: str = 'CT'):
    """
    calculates the intrinsic h exchange rates based on the amino acid sequence for a polypeptide chain
    # calculate the instrinsic exchange rate
    # taken directly from https://gitlab.com/mcpe/psx/blob/master/Code/IntrinsicExchange.py
    # changed the raw values based on the paper J. Am. Soc. Mass Spectrom. (2018) 29;1936-1939
    :param sequence_str: sequence of the protein (needs to include additional nterm and cterm residues as well)
    :param temperature: temperature
    :param ph: ph
    :param nterm_mode:
    :param cterm_mode:
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
    TemperatureCorrection = (1.0/temperature - 1.0/278.0) / R
    Temperature_Corr_2 = (1.0/temperature - 1.0/293.0) / R
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
        10.0 ** (MilneAcid['D+'][0] - ph) / (10.0 ** -pKAsp + 10.0 ** -ph) + 10.0 ** (MilneAcid['D0'][0] - pKAsp) / (
                10.0 ** -pKAsp + 10.0 ** -ph))
    LambdaProtonatedAcidGlu = math.log10(
        10.0 ** (MilneAcid['E+'][0] - ph) / (10.0 ** -pKGlu + 10.0 ** -ph) + 10.0 ** (MilneAcid['E0'][0] - pKGlu) / (
                10.0 ** -pKGlu + 10.0 ** -ph))
    LambdaProtonatedAcidHis = math.log10(
        10.0 ** (MilneAcid['H+'][0] - ph) / (10.0 ** -pKHis + 10.0 ** -ph) + 10.0 ** (MilneAcid['H0'][0] - pKHis) / (
                10.0 ** -pKHis + 10.0 ** -ph))

    RhoProtonatedAcidAsp = math.log10(
        10.0 ** (MilneAcid['D+'][1] - ph) / (10.0 ** -pKAsp + 10.0 ** -ph) + 10.0 ** (MilneAcid['D0'][1] - pKAsp) / (
                10.0 ** -pKAsp + 10.0 ** -ph))
    RhoProtonatedAcidGlu = math.log10(
        10.0 ** (MilneAcid['E+'][1] - ph) / (10.0 ** -pKGlu + 10.0 ** -ph) + 10.0 ** (MilneAcid['E0'][1] - pKGlu) / (
                10.0 ** -pKGlu + 10.0 ** -ph))
    RhoProtonatedAcidHis = math.log10(
        10.0 ** (MilneAcid['H+'][1] - ph) / (10.0 ** -pKHis + 10.0 ** -ph) + 10.0 ** (MilneAcid['H0'][1] - pKHis) / (
                10.0 ** -pKHis + 10.0 ** -ph))

    LambdaProtonatedBaseAsp = math.log10(
        10.0 ** (MilneBase['D+'][0] - ph) / (10.0 ** -pKAsp + 10.0 ** -ph) + 10.0 ** (MilneBase['D0'][0] - pKAsp) / (
                10.0 ** -pKAsp + 10.0 ** -ph))
    LambdaProtonatedBaseGlu = math.log10(
        10.0 ** (MilneBase['E+'][0] - ph) / (10.0 ** -pKGlu + 10.0 ** -ph) + 10.0 ** (MilneBase['E0'][0] - pKGlu) / (
                10.0 ** -pKGlu + 10.0 ** -ph))
    LambdaProtonatedBaseHis = math.log10(
        10.0 ** (MilneBase['H+'][0] - ph) / (10.0 ** -pKHis + 10.0 ** -ph) + 10.0 ** (MilneBase['H0'][0] - pKHis) / (
                10.0 ** -pKHis + 10.0 ** -ph))

    RhoProtonatedBaseAsp = math.log10(
        10.0 ** (MilneBase['D+'][1] - ph) / (10.0 ** -pKAsp + 10.0 ** -ph) + 10.0 ** (MilneBase['D0'][1] - pKAsp) / (
                10.0 ** -pKAsp + 10.0 ** -ph))
    RhoProtonatedBaseGlu = math.log10(
        10.0 ** (MilneBase['E+'][1] - ph) / (10.0 ** -pKGlu + 10.0 ** -ph) + 10.0 ** (MilneBase['E0'][1] - pKGlu) / (
                10.0 ** -pKGlu + 10.0 ** -ph))
    RhoProtonatedBaseHis = math.log10(
        10.0 ** (MilneBase['H+'][1] - ph) / (10.0 ** -pKHis + 10.0 ** -ph) + 10.0 ** (MilneBase['H0'][1] - pKHis) / (
                10.0 ** -pKHis + 10.0 ** -ph))

    MilneAcid["D"] = (LambdaProtonatedAcidAsp, RhoProtonatedAcidAsp)
    MilneAcid["E"] = (LambdaProtonatedAcidGlu, RhoProtonatedAcidGlu)
    MilneAcid["H"] = (LambdaProtonatedAcidHis, RhoProtonatedAcidHis)

    MilneBase["D"] = (LambdaProtonatedBaseAsp, RhoProtonatedBaseAsp)
    MilneBase["E"] = (LambdaProtonatedBaseGlu, RhoProtonatedBaseGlu)
    MilneBase["H"] = (LambdaProtonatedBaseHis, RhoProtonatedBaseHis)

    # Termini
    RhoAcidNTerm = -1.32
    LambdaAcidCTerm = math.log10(10.0 ** (0.05 - ph) / (10.0 ** -pKGlu + 10.0 ** -ph) + 10.0 ** (0.96 - pKGlu) / (
            10.0 ** -pKGlu + 10.0 ** -ph))

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
    DIonConc = 10.0 ** -ph
    ODIonConc = 10.0 ** (ph - pKD)

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


def cal_hx_prob_with_backexchange(timepoints, rate_constant, inv_backexchange, d2o_purity, d2o_fraction):
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
    prob = (1.0 - np.exp(-rate_constant * timepoints)) * (d2o_fraction * d2o_purity * inv_backexchange)
    return prob


def hx_rates_probability_distribution_with_fes(timepoints, rates, inv_backexchange, d2o_fraction, d2o_purity,
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
                                               inv_backexchange=inv_backexchange,
                                               d2o_purity=d2o_purity,
                                               d2o_fraction=d2o_fraction)
    pmf_hx_probabs = PoiBin(hx_probabs)
    return pmf_hx_probabs


def isotope_dist_from_PoiBin_bkexch(sequence_length, isotope_dist, timepoint, rates, num_bins, inv_backexchange,
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
                                                                 inv_backexchange=inv_backexchange,
                                                                 d2o_fraction=d2o_fraction,
                                                                 d2o_purity=d2o_purity,
                                                                 fes=fes,
                                                                 temp=temp)
    isotope_dist_poibin_convol = np.convolve(pmf_hx_prob_fes, isotope_dist)
    isotope_dist_poibin_convol_norm = isotope_dist_poibin_convol[:num_bins]/max(isotope_dist_poibin_convol[:num_bins])
    return isotope_dist_poibin_convol_norm


def back_exchange_(sequence_length: int,
                            theoretical_isotope_distribution: np.ndarray,
                            experimental_isotope_distribution: np.ndarray,
                            intrinsic_rates: np.ndarray,
                            temperature: float,
                            d2o_fraction: float,
                            d2o_purity: float) -> float:
    """

    """
    num_bins_ = len(experimental_isotope_distribution)

    opt = fmin_powell(lambda x: mean_squared_error(experimental_isotope_distribution,
                                                   isotope_dist_from_PoiBin_bkexch(sequence_length=sequence_length,
                                                                                   isotope_dist=theoretical_isotope_distribution,
                                                                                   timepoint=1e9,
                                                                                   rates=intrinsic_rates,
                                                                                   num_bins=num_bins_,
                                                                                   inv_backexchange=expit(x),
                                                                                   d2o_fraction=d2o_fraction,
                                                                                   d2o_purity=d2o_purity,
                                                                                   temp=temperature),
                                                   squared=False), x0=2, disp=True)

    back_exchange = 1 - expit(opt)[0]

    return back_exchange


if __name__ == '__main__':

    pass
