from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_grid(fig_obj,
              sample,
              sample_mean,
              sample_std,
              sample_5percent,
              sample_95percent,
              sample_rhat,
              sample_label,
              gridspec_obj,
              gridspec_index):
    """

    :param fig_obj:
    :param sample:
    :param sample_mean:
    :param sample_std:
    :param sample_5percent:
    :param sample_95percent:
    :param sample_rhat:
    :param sample_label:
    :param gridspec_obj:
    :param gridspec_index:
    :return:
    """

    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gridspec_obj[gridspec_index])

    ax00 = fig_obj.add_subplot(gs00[0, :-1])
    ax00.plot(sample, color='indianred', linewidth=0.5)
    ax00.hlines(y=sample_mean, xmin=0, xmax=len(sample), ls='--', colors='black', linewidth=0.5)
    ax00.hlines(y=sample_5percent, xmin=0, xmax=len(sample), ls='--', colors='black', linewidth=0.5)
    ax00.hlines(y=sample_95percent, xmin=0, xmax=len(sample), ls='--', colors='black', linewidth=0.5)
    ax00.spines['right'].set_visible(False)
    ax00.spines['top'].set_visible(False)
    plt.xticks(range(0, len(sample) + 100, 100))
    plt.grid(alpha=0.25)
    ax00.tick_params(length=3, pad=3)

    # put stats on the right side
    plt.text(0.95,
             1.2,
             "mean = %.4f\nstd = %.4f" % (sample_mean, sample_std),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax00.transAxes)

    # put the label on the left side

    plt.text(0.01, 1.2, sample_label,
             horizontalalignment="left",
             verticalalignment="top",
             transform=ax00.transAxes)

    ax01 = fig_obj.add_subplot(gs00[0, -1])
    ax01.scatter(0, sample_rhat, color='black')
    ax01.hlines(y=1+0.05, xmin=-1, xmax=1, ls='--', colors='black', linewidth=1)
    ax01.hlines(y=1-0.05, xmin=-1, xmax=1, ls='--', colors='black', linewidth=1)
    ax01.spines['right'].set_visible(False)
    ax01.spines['top'].set_visible(False)
    ax01.spines['bottom'].set_visible(False)
    plt.xticks([])
    ax01.tick_params(length=3, pad=3)

    # put rhat on the right side
    plt.text(0.95,
             1.2,
             "rhat = %.4f" % sample_rhat,
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax01.transAxes)


def plot_grid_v2(fig_obj,
                 sample,
                 sample_mean,
                 sample_std,
                 sample_5percent,
                 sample_95percent,
                 sample_rhat,
                 sample_label,
                 gridspec_obj,
                 gridspec_index):
    """

    :param fig_obj:
    :param sample:
    :param sample_mean:
    :param sample_std:
    :param sample_5percent:
    :param sample_95percent:
    :param sample_rhat:
    :param sample_label:
    :param gridspec_obj:
    :param gridspec_index:
    :return:
    """

    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gridspec_obj[gridspec_index])

    ax00 = fig_obj.add_subplot(gs00[0, :-1])

    chain_colors = cycle(['indianred', 'mediumaquamarine', 'deepskyblue', 'mediumpurple', 'palevioletred'])

    ind_arr = np.arange(0, len(sample[0]))
    for ind, (sample_per_chain, color_) in enumerate(zip(sample, chain_colors)):
        if ind == 0:
            ax00.plot(ind_arr, sample_per_chain, color=color_, linewidth=0.5)
        else:
            last_ind = ind_arr[-1]
            ind_arr = np.arange(last_ind+1, last_ind + 1 + len(sample_per_chain))
            ax00.plot(ind_arr, sample_per_chain, color=color_, linewidth=0.5)

    ax00.hlines(y=sample_mean, xmin=0, xmax=ind_arr[-1], ls='--', colors='red', linewidth=0.5)
    ax00.hlines(y=sample_5percent, xmin=0, xmax=ind_arr[-1], ls='--', colors='black', linewidth=0.5)
    ax00.hlines(y=sample_95percent, xmin=0, xmax=ind_arr[-1], ls='--', colors='black', linewidth=0.5)
    ax00.spines['right'].set_visible(False)
    ax00.spines['top'].set_visible(False)
    plt.grid(alpha=0.25)
    ax00.tick_params(length=3, pad=3)

    # put stats on the right side
    plt.text(0.95,
             1.2,
             "mean = %.4f\nstd = %.4f" % (sample_mean, sample_std),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax00.transAxes)

    # put the label on the left side
    plt.text(0.01, 1.2, sample_label,
             horizontalalignment="left",
             verticalalignment="top",
             transform=ax00.transAxes)

    ax01 = fig_obj.add_subplot(gs00[0, -1])
    ax01.scatter(0, sample_rhat, color='black')
    ax01.hlines(y=1+0.05, xmin=-1, xmax=1, ls='--', colors='black', linewidth=1)
    ax01.hlines(y=1-0.05, xmin=-1, xmax=1, ls='--', colors='black', linewidth=1)
    ax01.spines['right'].set_visible(False)
    ax01.spines['top'].set_visible(False)
    ax01.spines['bottom'].set_visible(False)
    plt.xticks([])
    ax01.tick_params(length=3, pad=3)

    # put rhat on the right side
    plt.text(0.95,
             1.2,
             "rhat = %.4f" % sample_rhat,
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax01.transAxes)


def reshape_posterior_rates_samples(posterior_rates):

    posterior_rates_shape = posterior_rates.shape

    rate_reshape_array = np.zeros((posterior_rates_shape[-1], posterior_rates_shape[0], posterior_rates_shape[1]))

    for chain_ind, chain_arrs in enumerate(posterior_rates):
        chain_arr_transpose = chain_arrs.T
        for rate_ind, rate_arr in enumerate(chain_arr_transpose):
            for sample_ind, sample_value in enumerate(rate_arr):
                rate_reshape_array[rate_ind][chain_ind][sample_ind] = sample_value

    return rate_reshape_array


def plot_posteriors(bayesfit_output, output_path=None):

    backexchange_present = False
    num_backexchange = 0
    if 'backexchange' in list(bayesfit_output['posterior_samples'].keys()):
        backexchange_present = True
        num_backexchange = len(bayesfit_output['backexchange']['mean'])

    posterior_rates = bayesfit_output['posterior_samples']['rate']

    rate_mean = bayesfit_output['rate']['mean']
    rate_std = bayesfit_output['rate']['std']
    rates_5percent = bayesfit_output['rate']['5percent']
    rates_95percent = bayesfit_output['rate']['95percent']
    rate_rhat = bayesfit_output['rate']['r_hat']

    # number of columns 2
    num_columns = 2

    num_of_subplots = 1 + len(rate_mean) + num_backexchange

    num_rows = num_of_subplots/num_columns

    if num_of_subplots % 2 != 0:
        num_rows = divmod(num_of_subplots, num_columns)[0] + 1

    num_rows = int(num_rows)

    font_size = 8

    fig_size = (20, 2.5 * num_rows)

    sample_sigma = bayesfit_output['posterior_samples']['sigma']
    ind_arr = np.arange(0, len(sample_sigma[0]))
    for ind, sample_per_chain in enumerate(sample_sigma):
        if ind == 0:
            ind_arr = ind_arr
        else:
            last_ind = ind_arr[-1]
            ind_arr = np.arange(last_ind+1, last_ind + 1 + len(sample_per_chain))

    rate_reshape_array = reshape_posterior_rates_samples(posterior_rates=posterior_rates)

    fig = plt.figure(figsize=fig_size)

    plt.rcParams.update({'font.size': font_size})

    gs0 = gridspec.GridSpec(nrows=num_rows, ncols=num_columns)

    # reshape sigma array

    # plot sigma
    plot_grid_v2(fig_obj=fig,
                 sample=bayesfit_output['posterior_samples']['sigma'],
                 sample_mean=bayesfit_output['sigma']['mean'],
                 sample_std=bayesfit_output['sigma']['std'],
                 sample_5percent=bayesfit_output['sigma']['5percent'],
                 sample_95percent=bayesfit_output['sigma']['95percent'],
                 sample_rhat=bayesfit_output['sigma']['r_hat'],
                 sample_label='sigma',
                 gridspec_obj=gs0,
                 gridspec_index=0)

    for num in range(len(rate_mean)):
        print('rate_'+str(num))
        plot_grid_v2(fig_obj=fig,
                     sample=rate_reshape_array[num],
                     sample_mean=rate_mean[num],
                     sample_std=rate_std[num],
                     sample_5percent=rates_5percent[num],
                     sample_95percent=rates_95percent[num],
                     sample_rhat=rate_rhat[num],
                     sample_label='rate_'+str(num),
                     gridspec_obj=gs0,
                     gridspec_index=num+1)

    if backexchange_present:
        bkexchange_reshape_array = reshape_posterior_rates_samples(posterior_rates=bayesfit_output['posterior_samples']['backexchange'])
        for num in range(len(bayesfit_output['backexchange']['mean'])):
            print('backexhange_'+str(num))
            plot_grid_v2(fig_obj=fig,
                         sample=bkexchange_reshape_array[num],
                         sample_mean=bayesfit_output['backexchange']['mean'][num],
                         sample_std=bayesfit_output['backexchange']['std'][num],
                         sample_5percent=bayesfit_output['backexchange']['5percent'][num],
                         sample_95percent=bayesfit_output['backexchange']['95percent'][num],
                         sample_rhat=bayesfit_output['backexchange']['r_hat'][num],
                         sample_label='backexchange_'+str(num),
                         gridspec_obj=gs0,
                         gridspec_index=num+len(rate_mean)+1)

    plt.subplots_adjust(hspace=1.2, wspace=0.12, top=0.95)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':

    pass
