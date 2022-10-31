import copy
import numpy as np
from dataclasses import dataclass
from sklearn.neighbors import KernelDensity
from scipy import stats
from scipy.optimize import minimize
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from methods import calculate_intrinsic_exchange_rates_suggie
from hxdata import load_pickle_object, write_pickle_object
import matplotlib.pyplot as plt


@dataclass
class HXRate(object):
    """

    """
    mean: np.ndarray
    median: np.ndarray = None
    std: np.ndarray = None
    ci_5: np.ndarray = None
    ci_95: np.ndarray = None


@dataclass
class IntrinsicRate(object):
    """"""

    intrinsic_rates: np.ndarray

    def median(self):
        """

        :return:
        """
        intrates_nonzero = self.gen_int_rates_nonzero()
        return np.median(intrates_nonzero)

    def mean(self):
        """

        :return:
        """
        intrates_nonzero = self.gen_int_rates_nonzero()
        return np.mean(intrates_nonzero)

    def density(self, x, par0, par1):
        """

        :param x:
        :param par0:
        :param par1:
        :return:
        """
        return -stats.norm.pdf(x, par0, par1)

    def mode(self):
        """

        :return:
        """
        intrates_nonzero = self.gen_int_rates_nonzero()
        kde = KernelDensity(bandwidth=0.1).fit(intrates_nonzero)
        cdist = kde.sample(1000)
        norm_fit = stats.norm.fit(cdist)
        mode = minimize(self.density, (0), args=(norm_fit[0], norm_fit[1])).x
        return mode

    def gen_int_rates_nonzero(self):
        """

        :return:
        """
        int_rates = self.intrinsic_rates
        int_rates[1] = 0.0
        intrinsic_rates_nonzero_ind = np.nonzero(int_rates)
        intrinsic_rates_nonzero = int_rates[intrinsic_rates_nonzero_ind]
        return intrinsic_rates_nonzero


@dataclass
class DGOutput(object):
    """

    """
    intrinsic_rates: np.ndarray = None
    intrinsic_rates_median: float = None
    netcharge: float = None
    free_energy: np.ndarray = None
    sorted_free_energy: np.ndarray = None
    hxrate_output: dict = None

    def check_empty_data(self):
        """

        :return:
        """
        if self.sorted_free_energy is not None:
            return True
        else:
            raise TypeError('Object is empty')

    def to_csv(self, filepath):
        """

        :param filepath:
        :return:
        """
        if self.check_empty_data():
            with open(filepath, 'w') as outfile:
                outfile.write('ind,dg\n')
                for ind, dg_val in enumerate(self.sorted_free_energy):
                    line = '{},{}\n'.format(ind, dg_val)
                    outfile.write(line)

                outfile.close()

    def to_dict(self):
        """

        :return:
        """
        dict_obj = copy.deepcopy(self)
        dict_out = vars(dict_obj)
        return dict_out

    def to_pickle(self, filepath):
        """

        :param filepath:
        :return:
        """
        if self.check_empty_data():
            write_pickle_object(obj=self.to_dict(), filepath=filepath)

    def plot_dg(self, filepath):
        """

        :param filepath:
        :return:
        """
        if self.check_empty_data():
            fig, ax = plt.subplots(figsize=(7, 5))
            plt.plot(self.sorted_free_energy, marker='o', ls='-', markerfacecolor='red', markeredgecolor='black',
                     color='red')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xticks(range(0, len(self.sorted_free_energy)+1, 2))
            plt.xlabel('Residue (ranked from most to least stable)')
            plt.ylabel('dG (kcal/mol)')
            plt.grid(alpha=0.25)
            ax.tick_params(length=3, pad=3)

            plt.subplots_adjust(hspace=0.2, wspace=0.1, top=0.94)

            # title = 'dG distribution'
            #
            # plt.suptitle(title)

            plt.savefig(filepath, bbox_inches="tight")
            plt.close()


def calc_net_charge_at_ph(protein_sequence, pH=6.15):
    """
    calculate net charge at certain pH
    :param protein_sequence:
    :param pH:
    :return:
    """
    analyses_seq = ProteinAnalysis(protein_sequence)
    net_charge = analyses_seq.charge_at_pH(pH)
    return net_charge


def corr_fe_with_net_charge(fe, net_charge):
    """

    :param fe:
    :param net_charge:
    :return:
    """
    fe_corr = fe - (net_charge * -0.12)
    return fe_corr


def calc_free_energy_from_hx_rates(hx_instrinsic_rate, hx_meas_rate, temperature=295.0, r_constant=1.987204e-3):
    """
    calculate free energy of opening (change in protein energy landscape upon opening or unfolding)
    :param hx_instrinsic_rate: hx intrinsic rate
    :param hx_meas_rate: hx measured rate
    :param r_constant: 1.987204e-3 kcal mol-1 K-1
    :param temperature: temperature in Kelvin
    :return: free energy of opening = - R T ln (k_op)
    """
    k_op = hx_meas_rate/(hx_instrinsic_rate - hx_meas_rate)
    free_energy = -r_constant * temperature * np.log(k_op)
    return free_energy


def gen_sorted_array(array, min_value=None, reverse=True):
    """

    :param array:
    :param min_value:
    :param reverse:
    :return:
    """

    arr_copy = copy.deepcopy(array)
    if min_value is not None:
        arr_copy[arr_copy <= min_value] = np.nan
    no_nan_ind = ~np.isnan(arr_copy)
    no_nan_arr = arr_copy[no_nan_ind]

    sorted_arr = np.array(sorted(no_nan_arr, reverse=reverse))

    return sorted_arr


def load_hx_rates(fpath):
    """

    :param fpath:
    :return:
    """

    df = pd.read_csv(fpath)
    hxrate = HXRate(mean=df['rate_mean'].values,
                    median=df['rate_median'].values,
                    std=df['rate_std'].values,
                    ci_5=df['rate_5%'].values,
                    ci_95=df['rate_95%'].values)
    return hxrate


def get_proline_res_ind(sequence):
    """
    get proline res ind
    :param sequence:
    :return:
    """
    aa_list = [str(x) for x in sequence]
    if 'P' in aa_list:
        proline_res_ind = []
        for ind, aa in enumerate(aa_list):
            if aa == 'P':
                proline_res_ind.append(ind)
    else:
        proline_res_ind = None

    return proline_res_ind


def dg_calc(sequence: str,
            measured_hx_rates: np.ndarray,
            temp: float,
            ph: float,
            netcharge_corr: bool = True,
            min_fe_val: float = -2.0) -> object:
    """
    :param sequence: protein sequence
    :param measured_hx_rates: measured hx rates
    :param temp: temperature
    :param ph: ph
    :param netcharge_corr: for correcting fe using netcharge of the protein
    :param min_fe_val: minimum free energy value
    :return: dg_calc object
    """

    # init dg output

    dgoutput = DGOutput()

    intrinsic_rate = IntrinsicRate(intrinsic_rates=calculate_intrinsic_exchange_rates_suggie(sequence_str=sequence,
                                                                                             Temperature=temp,
                                                                                             pH=ph))
    dgoutput.intrinsic_rates = intrinsic_rate.intrinsic_rates

    # use median to calculate dg values
    dgoutput.intrinsic_rates_median = intrinsic_rate.median()

    dgoutput.free_energy = calc_free_energy_from_hx_rates(hx_instrinsic_rate=dgoutput.intrinsic_rates_median,
                                                          hx_meas_rate=np.exp(measured_hx_rates),
                                                          temperature=temp)

    if netcharge_corr:
        dgoutput.netcharge = calc_net_charge_at_ph(protein_sequence=sequence,
                                                   pH=ph)

        dgoutput.free_energy = corr_fe_with_net_charge(fe=dgoutput.free_energy,
                                                       net_charge=dgoutput.netcharge)

    dgoutput.sorted_free_energy = gen_sorted_array(array=dgoutput.free_energy,
                                                   min_value=min_fe_val,
                                                   reverse=True)

    return dgoutput


def dg_calc_from_file(hxrate_pickle_fpath: str,
                      temp: float,
                      ph: float,
                      netcharge_corr: bool = True,
                      min_fe_val: float = -2.0,
                      output_picklepath: str = None,
                      dg_csv_fpath: str = None,
                      dg_plot_fpath: str = None,
                      retun_flag: bool = False):
    """

    :param min_fe_val:
    :param merge_factor_fpath:
    :param hxrate_pickle_fpath:
    :param temp:
    :param ph:
    :param netcharge_corr:
    :param sort_min_val:
    :param output_picklepath:
    :param dg_csv_fpath:
    :param dg_plot_fpath:
    :param retun_flag:
    :return:
    """

    hxrate_obj_ = load_pickle_object(hxrate_pickle_fpath)

    dg_output = dg_calc(sequence=hxrate_obj_['sequence'],
                        measured_hx_rates=hxrate_obj_['bayes_sample']['rate']['mean'],
                        temp=temp,
                        ph=ph,
                        netcharge_corr=netcharge_corr,
                        min_fe_val=min_fe_val)

    # save attributes here

    dg_output.hxrate_output = hxrate_obj_

    if output_picklepath is not None:
        dg_output.to_pickle(filepath=output_picklepath)

    if dg_csv_fpath is not None:
        dg_output.to_csv(filepath=dg_csv_fpath)

    if dg_plot_fpath is not None:
        dg_output.plot_dg(filepath=dg_plot_fpath)

    if retun_flag:
        return dg_output


def gen_parser_args():

    import argparse

    parser_ = argparse.ArgumentParser(prog='DG Calculation')
    parser_.add_argument('-i', '--input_', type=str, help='HX rate fit .pickle file path')
    parser_.add_argument('-t', '--temp', type=float, default=295, help='temperature in K')
    parser_.add_argument('-p', '--ph', type=float, default=6.0, help='ph')
    parser_.add_argument('-m', '--minfe', type=float, default=-2.0, help='min fe value')
    parser_.add_argument('-n', '--netcharge', default=True, action=argparse.BooleanOptionalAction)
    parser_.add_argument('-opk', '--output_pickle', type=str, help='dg output .pickle file path')
    parser_.add_argument('-oc', '--output_csv', type=str, help='dg output .csv file path')
    parser_.add_argument('-opd', '--output_pdf', type=str, help='dg output plot .pdf file path')

    return parser_


def run_from_parser():

    parser_ = gen_parser_args()

    options = parser_.parse_args()

    dg_calc_from_file(hxrate_pickle_fpath=options.input_,
                      temp=options.temp,
                      ph=options.ph,
                      netcharge_corr=options.netcharge,
                      min_fe_val=options.minfe,
                      output_picklepath=options.output_pickle,
                      dg_csv_fpath=options.output_csv,
                      dg_plot_fpath=options.output_pdf,
                      retun_flag=False)


if __name__ == '__main__':

    run_from_parser()

    # hxrate_pkfpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/EEHEE_rd4_0642.pdb_15.15_EEHEE_rd4_0642.pdb_15.15/EEHEE_rd4_0642.pdb_15.15_EEHEE_rd4_0642.pdb_15.15_hx_rate_fit.pickle'
    # dgoutput_pkfpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/EEHEE_rd4_0642.pdb_15.15_EEHEE_rd4_0642.pdb_15.15/EEHEE_rd4_0642.pdb_15.15_EEHEE_rd4_0642.pdb_15.15_hx_rate_fit.pickle_dg_output.pickle'
    # hxrate_pkobj = load_pickle_object(hxrate_pkfpath)
    #
    # temp = 295.0
    # ph = 6.0
    # #
    # dg_calc_from_file(hxrate_pickle_fpath=hxrate_pkfpath,
    #                   temp=temp,
    #                   ph=ph,
    #                   netcharge_corr=True,
    #                   output_picklepath=hxrate_pkfpath + '_dg_output.pickle',
    #                   dg_csv_fpath=hxrate_pkfpath + '_dg_output.csv',
    #                   dg_plot_fpath=hxrate_pkfpath + '_dg_output.pdf')

