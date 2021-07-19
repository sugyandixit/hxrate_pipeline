import matplotlib.pyplot as plt
import numpy as np
import molmass
import matplotlib.gridspec as gridspec
from scipy.optimize import basinhopping
from sklearn.metrics import mean_squared_error
from scipy.ndimage import center_of_mass
from dataclasses import dataclass
# from numba import jit

# global variables
r_constant = 0.0019872036


@dataclass
class GaussFit(object):
    """
    class container to fit and store gauss fit data
    """
    x_dist: np.ndarray
    gauss_fit_dist: np.ndarray
    y_baseline: float
    amplitude: float
    centroid: float
    width: float
    r_sq: float
    rmse: float


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


# def gauss_func(x, )


# def fit_gaussian_to_isotope_dist(isotope_dist):
#     """
#     fit gaussian to isotope dist
#     :param isotope_dist:
#     :return:
#     """



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
    com = center_of_mass(data_array)
    return com


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
                                  squared: bool = False):
    """
    compute the mean squared error between exp and thr isotope distribution only with values of exp_dist that are higher
    than 0
    :param exp_isotope_dist:
    :param thr_isotope_dist:
    :return:
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
                         'tol': 0.0001,
                         'options': {'maxiter': 1,
                                     'disp': True}}

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
                        disp=True)

    if return_tuple:
        return opt_, init_rate_guess
    else:
        return opt_


def plot_hx_rate_fitting_(hx_rates: np.ndarray,
                          exp_isotope_dist: np.ndarray,
                          thr_isotope_dist: np.ndarray,
                          timepoints: np.ndarray,
                          fit_rmse_timepoints: np.ndarray,
                          fit_rmse_total: float,
                          backexchange: float,
                          output_path: str):
    """
    generate several plots for visualizing the hx rate fitting output
    :param hx_rates: in ln scale
    :param exp_isotope_dist: exp isotope dist array
    :param thr_isotope_dist: thr isotope dist array from hx rates
    :param timepoints: time points
    :param fit_rmse_timepoints: fit rmse for each timepoint
    :param fit_rmse_total: total fit rmse
    :param backexchange: backexchange value
    :param output_path: plot saveing output path
    :return:
    """

    # define figure size
    num_columns = 2
    num_rows = len(timepoints)
    fig_size = (20, 25)

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
    for num, (timepoint, exp_dist, thr_dist) in enumerate(zip(timepoints, exp_isotope_dist, thr_isotope_dist)):

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

        # put the rmse on the right side of the plot
        plt.text(1.0, 1.0, "fit rmse %.4f" % rmse_tp,
                 horizontalalignment="right",
                 verticalalignment="top",
                 transform=ax.transAxes)

        # put timepoint on  the left side of the plot
        plt.text(0.01, 1.2, '%s t %i' % (num, timepoint),
                 horizontalalignment="left",
                 verticalalignment="top",
                 transform=ax.transAxes)
    #######################################################
    #######################################################

    # 4 plots on the second row
    num_plots_second_row = 4
    second_plot_row_thickness = int(len(timepoints)/num_plots_second_row)
    second_plot_indices = [(num*second_plot_row_thickness) for num in range(num_plots_second_row)]

    #######################################################
    #######################################################
    # plot center of mass exp and thr

    exp_com_ = np.array([center_of_mass_(arr) for arr in exp_isotope_dist])
    thr_com_ = np.array([center_of_mass_(arr) for arr in thr_isotope_dist])

    timepoints_v2 = np.array([x for x in timepoints])
    timepoints_v2[0] = timepoints_v2[2] - timepoints_v2[1]

    ax3 = fig.add_subplot(gs[second_plot_indices[0]: second_plot_indices[1], 1])
    ax3.plot(timepoints_v2, exp_com_, marker='o', ls='-', color='blue')
    ax3.plot(timepoints_v2, thr_com_, marker='o', ls='-', color='red')
    ax3.set_xscale('log')
    ax3.set_xticks(timepoints_v2)
    ax3.set_xticklabels([])
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    plt.grid(axis='x', alpha=0.25)
    plt.xlabel('log(timepoint)')
    plt.ylabel('Center of Mass')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the error in center of mass between exp and thr distributions

    com_difference = np.subtract(thr_com_, exp_com_)

    ax4 = fig.add_subplot(gs[second_plot_indices[1]: second_plot_indices[2], 1])
    ax4.scatter(np.arange(len(timepoints)), com_difference, color='red')
    plt.axhline(y=0, ls='--', color='black')
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax4.set_xticklabels(range(0, len(timepoints) + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Center of mass difference (THR - EXP)')
    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot fit rmse

    if max(fit_rmse_tp) <= 0.15:
        y_ticks = np.round(np.linspace(0, 0.15, num=16), 2)
    else:
        y_ticks = np.round(np.linspace(0, max(fit_rmse_tp)+0.05, num=16), 2)

    ax2 = fig.add_subplot(gs[second_plot_indices[2]: second_plot_indices[3], 1])
    plt.scatter(np.arange(len(timepoints)), fit_rmse_tp, color='red')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.xticks(range(0, len(timepoints) + 1, 1))
    ax2.set_xticklabels(range(0, len(timepoints) + 1, 1))
    plt.yticks(y_ticks)
    ax2.set_yticklabels(y_ticks)
    if max(fit_rmse_tp) > 0.15:
        plt.axhline(y=0.15, ls='--', color='black')
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)
    plt.xlabel('Timepoint index')
    plt.ylabel('Fit RMSE')
    plt.grid(axis='x', alpha=0.25)
    ax2.tick_params(length=3, pad=3)

    #######################################################
    #######################################################

    #######################################################
    #######################################################
    # plot the rates in log10 scale
    hx_rates_log10 = np.log10(np.exp(hx_rates))

    ax5 = fig.add_subplot(gs[second_plot_indices[3]:, 1])
    plt.plot(np.arange(len(hx_rates_log10)), np.sort(hx_rates_log10), marker='o', ls='-', color='black', markerfacecolor='red')
    plt.xticks(range(0, len(hx_rates_log10) + 2, 2))
    ax5.set_xticklabels(range(0, len(hx_rates_log10) + 2, 2))
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    plt.grid(axis='x', alpha=0.25)
    plt.xlabel('Residues (Ranked from slowest to fastest exchanging)')
    plt.ylabel('Rate: log k (1/s)')
    #######################################################
    #######################################################

    # adjust some plot properties and add title
    plt.subplots_adjust(hspace=1.2, wspace=0.1, top=0.96)

    plot_title = 'EXP vs THR Isotope Distribution (Fit RMSE: %.4f | BACKEXCHANGE: %.2f)' % (fit_rmse_total, backexchange*100)
    plt.suptitle(plot_title)

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
