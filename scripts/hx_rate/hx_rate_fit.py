# hx_rate_fitting code
# authors: Gabe Rocklin and Suggie

import argparse
import numpy as np
from dataclasses import dataclass
import pandas as pd
from bayesopt import BayesRateFit
from methods import normalize_mass_distribution_array, gauss_fit_to_isotope_dist_array,  \
    convert_hxrate_object_to_dict, plot_hx_rate_fitting_bayes
from backexchange import calc_back_exchange
from hxdata import load_data_from_hdx_ms_dist_, write_pickle_object, write_hx_rate_output_bayes, \
    write_isotope_dist_timepoints, load_tp_dependent_dict
from plot_posterior_samples import plot_posteriors


@dataclass
class MergeData(object):
    merge: bool = False
    factor: float = None
    mse: float = None


@dataclass
class ExpData(object):
    """
    class container to store exp data
    """
    protein_name: str = None
    protein_sequence: str = None
    timepoints: np.ndarray = None
    exp_isotope_dist_array: np.ndarray = None
    gauss_fit: list = None
    d2o_frac: float = None
    d2o_pur: float = None
    temp: float = None
    ph: float = None


@dataclass
class HXRate(object):
    """
    class container to store hx rate fitting data
    """
    exp_data: object = None
    back_exchange: object = None
    merge_data: object = None
    back_exchange_res_subtract: int = None
    optimization_cost: float = None
    optimization_func_evals: int = None
    optimization_init_rate_guess: np.ndarray = None
    hx_rates: np.ndarray = None
    bayesfit_output: dict = None


def fit_rate_bayes_(prot_name: str,
                    sequence: str,
                    time_points: np.ndarray,
                    norm_mass_distribution_array: np.ndarray,
                    d2o_fraction: float,
                    d2o_purity: float,
                    num_chains: int,
                    num_warmups: int,
                    num_samples: int,
                    ph: float = None,
                    temp: float = None,
                    sample_backexchange: bool = False,
                    return_posterior_distribution: bool = False,
                    adj_backexchange: bool = True,
                    backexchange_value: float = None,
                    backexchange_correction_dict: dict = None,
                    backexchange_array: np.ndarray = None,
                    max_res_subtract_for_backexchange: int = 3,
                    slow_rates_max_diff: float = 1.6) -> object:

    # todo: add param descritpions

    # initialize hxrate data object
    hxrate = HXRate()

    # store exp data in the object
    expdata = ExpData(protein_name=prot_name,
                      protein_sequence=sequence,
                      timepoints=time_points,
                      exp_isotope_dist_array=norm_mass_distribution_array,
                      d2o_frac=d2o_fraction,
                      d2o_pur=d2o_purity,
                      ph=ph,
                      temp=temp)

    # fit gaussian to exp data
    expdata.gauss_fit = gauss_fit_to_isotope_dist_array(norm_mass_distribution_array)

    # store expdata object in hxrate data object
    hxrate.exp_data = expdata

    back_exchange = calc_back_exchange(sequence=sequence,
                                       experimental_isotope_dist=norm_mass_distribution_array[-1],
                                       timepoints_array=time_points,
                                       d2o_fraction=d2o_fraction,
                                       d2o_purity=d2o_purity,
                                       usr_backexchange=backexchange_value,
                                       backexchange_array=backexchange_array,
                                       backexchange_corr_dict=backexchange_correction_dict)

    hxrate.back_exchange = back_exchange

    if adj_backexchange:

        # set a flag for backexchange adjusting
        backexchange_adjust = False
        hxrate.back_exchange_res_subtract = 0

        # set the original backexchange for reference
        original_backexchange = back_exchange.backexchange_value

        while backexchange_adjust is False:

            # store backexchange object in hxrate object
            hxrate.back_exchange = back_exchange

            print('\nHX RATE FITTING ... ')

            ratefit = BayesRateFit(num_chains=num_chains,
                                   num_warmups=num_warmups,
                                   num_samples=num_samples,
                                   sample_backexchange=sample_backexchange,
                                   return_posterior_distributions=return_posterior_distribution)

            ratefit.fit_rate(sequence=sequence,
                             timepoints=time_points,
                             exp_distribution=norm_mass_distribution_array,
                             back_exchange_array=back_exchange.backexchange_array,
                             d2o_fraction=d2o_fraction,
                             d2o_purity=d2o_purity)

            hxrate.bayesfit_output = ratefit.output

            # get the difference of the slowest rate and 3rd slowest.
            rate_diff = ratefit.output['rate']['mean'][2] - ratefit.output['rate']['mean'][0]
            # rate_diff = 2.0  # for debugging purpose

            # if the rate difference is smaller than 1.6, hxrate optimization ends
            if rate_diff < slow_rates_max_diff:
                backexchange_adjust = True

            # else, backexchange is adjusted by reducing its value in proportion to having less residues.
            # then redo the hx rate fitting
            else:
                hxrate.back_exchange_res_subtract += 1
                # if the backexchange res subtract exceeds the max res subtract, terminate the fitting routine
                if hxrate.back_exchange_res_subtract > max_res_subtract_for_backexchange:
                    backexchange_adjust = True
                else:
                    print('adjusting backexchange value ... \n')
                    backexchange_value = original_backexchange * ((len(ratefit.output['rate']['mean']) + hxrate.back_exchange_res_subtract)/len(ratefit.output['rate']['mean']))

                    back_exchange = calc_back_exchange(sequence=sequence,
                                                       experimental_isotope_dist=norm_mass_distribution_array[-1],
                                                       timepoints_array=time_points,
                                                       d2o_fraction=d2o_fraction,
                                                       d2o_purity=d2o_purity,
                                                       usr_backexchange=backexchange_value,
                                                       backexchange_array=None,
                                                       backexchange_corr_dict=hxrate.back_exchange.backexchange_correction_dict)

    else:

        print('\nHX RATE FITTING ... ')

        ratefit = BayesRateFit(num_chains=num_chains,
                               num_warmups=num_warmups,
                               num_samples=num_samples,
                               sample_backexchange=sample_backexchange,
                               return_posterior_distributions=return_posterior_distribution)

        ratefit.fit_rate(sequence=sequence,
                         timepoints=time_points,
                         exp_distribution=norm_mass_distribution_array,
                         back_exchange_array=back_exchange.backexchange_array,
                         d2o_fraction=d2o_fraction,
                         d2o_purity=d2o_purity)

        hxrate.bayesfit_output = ratefit.output

    return hxrate


def fit_rate_from_to_file(prot_name: str,
                          sequence: str,
                          hx_ms_dist_fpath: str,
                          d2o_fraction: float,
                          d2o_purity: float,
                          ph: float = None,
                          temp: float = None,
                          merge_stat_csv_path: str = None,
                          usr_backexchange: float = None,
                          backexchange_corr_fpath: str = None,
                          backexchange_array_fpath: str = None,
                          adjust_backexchange: bool = True,
                          num_chains: int = 4,
                          num_warmups: int = 100,
                          num_samples: int = 500,
                          sample_backexchange: bool = False,
                          return_posterior_distribution: bool = True,
                          hx_rate_output_path: str = None,
                          hx_rate_csv_output_path: str = None,
                          hx_isotope_dist_output_path: str = None,
                          hx_rate_plot_path: str = None,
                          posterior_plot_path: str = None,
                          return_flag: bool = False) -> object:
    # todo: add param descriptions for the function here

    # get the mass distribution data at each hdx timepoint
    timepoints, mass_dist = load_data_from_hdx_ms_dist_(hx_ms_dist_fpath)

    # normalize the mass distribution
    norm_dist = normalize_mass_distribution_array(mass_dist_array=mass_dist)

    # declare backexchange correction
    bkexch_corr_dict = None
    if backexchange_corr_fpath is not None:
        bkexch_corr_dict = load_tp_dependent_dict(filepath=backexchange_corr_fpath)

    # declare backexchange array
    backexchange_array = None
    if backexchange_array_fpath is not None:
        backexchange_array_dict = load_tp_dependent_dict(filepath=backexchange_array_fpath)
        backexchange_array = np.zeros(len(timepoints))
        for ind, tp in enumerate(timepoints):
            backexchange_array[ind] = backexchange_array_dict[tp]

    # fit rate
    hxrate_object = fit_rate_bayes_(prot_name=prot_name,
                                    sequence=sequence,
                                    time_points=timepoints,
                                    norm_mass_distribution_array=norm_dist,
                                    d2o_fraction=d2o_fraction,
                                    d2o_purity=d2o_purity,
                                    num_chains=num_chains,
                                    num_samples=num_samples,
                                    num_warmups=num_warmups,
                                    ph=ph,
                                    temp=temp,
                                    sample_backexchange=sample_backexchange,
                                    return_posterior_distribution=return_posterior_distribution,
                                    adj_backexchange=adjust_backexchange,
                                    backexchange_value=usr_backexchange,
                                    backexchange_correction_dict=bkexch_corr_dict,
                                    backexchange_array=backexchange_array)

    # fit gaussian to thr pred distribution
    hxrate_object.bayesfit_output['pred_dist_gauss_fit'] = gauss_fit_to_isotope_dist_array(isotope_dist=hxrate_object.bayesfit_output['pred_distribution'])

    # gen merge_data object
    merge_obj = MergeData()
    if merge_stat_csv_path is not None:
        df = pd.read_csv(merge_stat_csv_path)

        merge_obj = MergeData(merge=True,
                              factor=df['factor'].values[0],
                              mse=df['mse'].values[0])

    hxrate_object.merge_data = merge_obj

    # write hxrate as a csv file
    if hx_rate_csv_output_path is not None:
        write_hx_rate_output_bayes(hxrate_mean_array=hxrate_object.bayesfit_output['rate']['mean'],
                                   hxrate_median_array=hxrate_object.bayesfit_output['rate']['median'],
                                   hxrate_std_array=hxrate_object.bayesfit_output['rate']['std'],
                                   hxrate_5percent_array=hxrate_object.bayesfit_output['rate']['5percent'],
                                   hxrate_95percent_array=hxrate_object.bayesfit_output['rate']['95percent'],
                                   neff_array=hxrate_object.bayesfit_output['rate']['n_eff'],
                                   r_hat_array=hxrate_object.bayesfit_output['rate']['r_hat'],
                                   output_path=hx_rate_csv_output_path)

    # write hxrate distributions as a csv file
    if hx_isotope_dist_output_path is not None:
        write_isotope_dist_timepoints(timepoints=timepoints,
                                      isotope_dist_array=hxrate_object.bayesfit_output['pred_distribution'],
                                      output_path=hx_isotope_dist_output_path)

    # plot hxrate output as .pdf file
    if hx_rate_plot_path is not None:

        exp_centroid_arr = np.array([x.centroid for x in hxrate_object.exp_data.gauss_fit])
        exp_width_arr = np.array([x.width for x in hxrate_object.exp_data.gauss_fit])

        thr_centroid_arr = np.array([x.centroid for x in hxrate_object.bayesfit_output['pred_dist_gauss_fit']])
        thr_width_arr = np.array([x.width for x in hxrate_object.bayesfit_output['pred_dist_gauss_fit']])

        hxrate_error = np.zeros((2, len(hxrate_object.bayesfit_output['rate']['mean'])))
        hxrate_error[0] = np.subtract(hxrate_object.bayesfit_output['rate']['mean'], hxrate_object.bayesfit_output['rate']['5percent'])
        hxrate_error[1] = np.subtract(hxrate_object.bayesfit_output['rate']['95percent'], hxrate_object.bayesfit_output['rate']['mean'])

        plot_hx_rate_fitting_bayes(prot_name=prot_name,
                                   hx_rates=hxrate_object.bayesfit_output['rate']['mean'],
                                   hx_rates_error=hxrate_error,
                                   exp_isotope_dist=norm_dist,
                                   thr_isotope_dist=hxrate_object.bayesfit_output['pred_distribution'],
                                   exp_isotope_centroid_array=exp_centroid_arr,
                                   thr_isotope_centroid_array=thr_centroid_arr,
                                   exp_isotope_width_array=exp_width_arr,
                                   thr_isotope_width_array=thr_width_arr,
                                   timepoints=timepoints,
                                   fit_rmse_timepoints=hxrate_object.bayesfit_output['rmse']['per_timepoint'],
                                   fit_rmse_total=hxrate_object.bayesfit_output['rmse']['total'],
                                   backexchange=hxrate_object.back_exchange.backexchange_value,
                                   backexchange_array=hxrate_object.back_exchange.backexchange_array,
                                   d2o_fraction=d2o_fraction,
                                   d2o_purity=d2o_purity,
                                   output_path=hx_rate_plot_path)

    # plot posterior samples
    if posterior_plot_path is not None:
        plot_posteriors(bayesfit_output=hxrate_object.bayesfit_output,
                        output_path=posterior_plot_path)

    # save hxrate object into a pickle object
    if hx_rate_output_path is not None:
        hxrate_dict = convert_hxrate_object_to_dict(hxrate_object=hxrate_object)
        write_pickle_object(hxrate_dict, hx_rate_output_path)

    if return_flag:
        return hxrate_object


def gen_parser_arguments():
    """
    generate commandline arguements to run the hx rate fitting algorithm
    :return:parser
    """

    parser = argparse.ArgumentParser(prog='HX_RATE_FIT', description='Run HX rate fitting algorithm')
    parser.add_argument('-p', '--protname', help='protein name', default='PROTEIN')
    parser.add_argument('-s', '--sequence', help='protein sequence one letter amino acid', default='PROTEIN')
    parser.add_argument('-i', '--hxdist', help='hx mass distribution input file .csv')
    parser.add_argument('-df', '--d2o_frac', help='d2o fracion', default=0.95)
    parser.add_argument('-dp', '--d2o_pur', help='d2o purity', default=0.95)
    parser.add_argument('-ph', '--phval', help='ph value', type=float, default=6.0)
    parser.add_argument('-tm', '--temp', help='temp value K', type=float, default=295)
    parser.add_argument('-ub', '--user_bkexchange', help='user defined backexchange', default=None)
    parser.add_argument('-bcf', '--bkexchange_corr_fpath', help='backexchange correction filepath .csv')
    parser.add_argument('-baf', '--bkexchange_array_fpath', help='backexchange array filepath .csv')
    parser.add_argument('-mfp', '--merge_fact_path', help='merge_factor.csv filepath', default=None)
    parser.add_argument('--adjust_backexchange', help='adjust backexchange boolean', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('-nc', '--num_chains', help='number of independent markov chains for MCMC', default=4)
    parser.add_argument('-nw', '--num_warmups', help='number of warmups for MCMC', default=100)
    parser.add_argument('-ns', '--num_samples', help='number of samples for MCMC', default=500)
    parser.add_argument('--sample_backexchange', help='sample backexchange for MCMC', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--return_posterior', help='return posterior distribution boolean', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('-o', '--output_pickle_file', help='output pickle filepath', default=None)
    parser.add_argument('-or', '--output_rate_csv', help='output rates csv filepath', default=None)
    parser.add_argument('-op', '--output_rate_plot', help='output_rate_plot filepath', default=None)
    parser.add_argument('-opp', '--output_posterior_plot', help='output_posterior_plot filepath', default=None)
    parser.add_argument('-od', '--output_iso_dist', help='output isotope distribution filepath', default=None)

    return parser


def hx_rate_fitting_from_parser(parser):
    """
    from the parser arguments, generate essential arguments for hx rate fitting function and run the function
    :param parser: parser
    :return:
    """

    options = parser.parse_args()

    user_backexchange = None
    if options.user_bkexchange is not None:
        user_backexchange = float(options.user_bkexchange)

    fit_rate_from_to_file(prot_name=options.protname,
                          sequence=options.sequence,
                          hx_ms_dist_fpath=options.hxdist,
                          d2o_fraction=float(options.d2o_frac),
                          d2o_purity=float(options.d2o_pur),
                          ph=options.phval,
                          temp=options.temp,
                          usr_backexchange=user_backexchange,
                          backexchange_corr_fpath=options.bkexchange_corr_fpath,
                          backexchange_array_fpath=options.bkexchange_array_fpath,
                          merge_stat_csv_path=options.merge_fact_path,
                          adjust_backexchange=options.adjust_backexchange,
                          num_chains=int(options.num_chains),
                          num_warmups=int(options.num_warmups),
                          num_samples=int(options.num_samples),
                          sample_backexchange=options.sample_backexchange,
                          return_posterior_distribution=options.return_posterior,
                          hx_rate_output_path=options.output_pickle_file,
                          hx_rate_csv_output_path=options.output_rate_csv,
                          hx_isotope_dist_output_path=options.output_iso_dist,
                          hx_rate_plot_path=options.output_rate_plot,
                          posterior_plot_path=options.output_posterior_plot)


if __name__ == '__main__':
    pass

    parser_ = gen_parser_arguments()
    hx_rate_fitting_from_parser(parser_)

    # prot_name = 'HEEH_rd4_0097.pdb_16.2012'
    # prot_sequence = 'HMDVEEQIRRLEEVLKKNQPVTWNGTTYTDPNEIKKVIEELRKSM'
    # hx_dist_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/hx_rates_library/lib14/sample_bkexch/HEEH_rd4_0097.pdb_16.2012_winner_multibody.cpickle.zlib.csv'
    # # bkexch_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/hx_rates_library/lib15/20211225_ph6_re_20211223_ph9/merge_distribution/HEEH_rd4_0097.pdb_16.16448_HEEH_rd4_0097.pdb_16.27463/HEEH_rd4_0097.pdb_16.16448_HEEH_rd4_0097.pdb_16.27463_merge_backexchange.csv'
    # bk_corr_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/hx_rates_library/lib14/sample_bkexch/bkexch_corr.csv'
    # output_dir = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/hx_rates_library/lib14/sample_bkexch'
    #
    # fit_rate_from_to_file(prot_name=prot_name,
    #                       sequence=prot_sequence,
    #                       hx_ms_dist_fpath=hx_dist_fpath,
    #                       d2o_fraction=0.95,
    #                       d2o_purity=0.95,
    #                       backexchange_corr_fpath=bk_corr_fpath,
    #                       num_chains=4,
    #                       num_warmups=100,
    #                       num_samples=500,
    #                       sample_backexchange=True,
    #                       hx_rate_output_path=output_dir + '/_hxrate_rate_norm_prior_sigma2.5_bkexch_sample_True_v2.pickle',
    #                       hx_rate_csv_output_path=output_dir + '/_hxrate_rate_norm_prior_sigma2.5_bkexch_sample_True_v2.csv',
    #                       hx_isotope_dist_output_path=output_dir + '/_hxrate_rate_norm_prior_sigma2.5_bkexch_sample_True_v2.csv',
    #                       hx_rate_plot_path=output_dir + '/_hxrate_rate_norm_prior_sigma2.5_bkexch_sample_True_v2.pdf')
