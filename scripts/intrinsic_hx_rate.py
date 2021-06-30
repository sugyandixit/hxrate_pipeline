import numpy as np
import math

def calc_intrinsic_hx_rates(sequence_str, Temperature, pH, nterm_mode='NT', cterm_mode='CT'):
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


if __name__ == '__main__':

    pass
