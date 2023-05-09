# hx_rate_fitting code
# authors: Gabe Rocklin and Suggie

import argparse
import pandas as pd
import time
import numpy as np
from bayesopt import BayesRateFit, ExpDataRateFit
from methods import normalize_mass_distribution_array, gen_backexchange_correction_from_backexchange_array, gen_corr_backexchange
from backexchange import calc_back_exchange
from hxdata import load_data_from_hdx_ms_dist_, load_tp_dependent_dict


def fit_rate_bayes_(prot_name: str,
                    sequence: str,
                    time_points: np.ndarray or list,
                    norm_mass_distribution_array: np.ndarray or list,
                    d2o_fraction: float,
                    d2o_purity: float,
                    num_chains: int,
                    num_warmups: int,
                    num_samples: int,
                    prot_rt_name: str = 'PROTEIN_RT',
                    timepoint_label: list = None,
                    exp_label: str or list = None,
                    merge_exp: bool = False,
                    sample_backexchange: bool = False,
                    adj_backexchange: bool = True,
                    backexchange_value: float or list = None,
                    backexchange_correction_dict: dict or list = None,
                    backexchange_array: np.ndarray or list = None,
                    max_res_subtract_for_backexchange: int = 3,
                    slow_rates_max_diff: float = 1.6) -> object:

    # todo: add param descritpions

    # calc backexchange
    if merge_exp:

        list_backexch_obj = []

        if backexchange_value is None:
            backexchange_value = [None for _ in range(len(time_points))]
        if backexchange_array is None:
            backexchange_array = [None for _ in range(len(time_points))]
        if backexchange_correction_dict is None:
            backexchange_correction_dict = [None for _ in range(len(time_points))]
        if timepoint_label is None:
            timepoint_label = [None for _ in range(len(time_points))]

        for ind, (norm_mass_dist_arr, tp_arr, bkexch_val, bkexcharr, bkexch_corr) in enumerate(zip(norm_mass_distribution_array,
                                                                                                   time_points,
                                                                                                   backexchange_value,
                                                                                                   backexchange_array,
                                                                                                   backexchange_correction_dict)):

            bkexch_obj = calc_back_exchange(sequence=sequence,
                                            experimental_isotope_dist=norm_mass_dist_arr[-1],
                                            timepoints_array=tp_arr,
                                            d2o_purity=d2o_purity,
                                            d2o_fraction=d2o_fraction,
                                            usr_backexchange=bkexch_val,
                                            backexchange_array=bkexcharr,
                                            backexchange_corr_dict=bkexch_corr)

            list_backexch_obj.append(bkexch_obj)

        backexchange_list = [x.backexchange_array for x in list_backexch_obj]

        expdata_obj = ExpDataRateFit(sequence=sequence,
                                     prot_name=prot_name,
                                     prot_rt_name=prot_rt_name,
                                     timepoints=time_points,
                                     timepoint_index_list=timepoint_label,
                                     exp_label=exp_label,
                                     exp_distribution=norm_mass_distribution_array,
                                     backexchange=backexchange_list,
                                     merge_exp=merge_exp,
                                     d2o_purity=d2o_purity,
                                     d2o_fraction=d2o_fraction)

        bkexch_corr_arr = gen_backexchange_correction_from_backexchange_array(backexchange_array=expdata_obj.backexchange)

    else:
        backexchange_obj = calc_back_exchange(sequence=sequence,
                                              experimental_isotope_dist=norm_mass_distribution_array[-1],
                                              timepoints_array=time_points,
                                              d2o_purity=d2o_purity,
                                              d2o_fraction=d2o_fraction,
                                              usr_backexchange=backexchange_value,
                                              backexchange_array=backexchange_array,
                                              backexchange_corr_dict=backexchange_correction_dict)
        expdata_obj = ExpDataRateFit(sequence=sequence,
                                     prot_name=prot_name,
                                     prot_rt_name=prot_rt_name,
                                     timepoints=time_points,
                                     timepoint_index_list=timepoint_label,
                                     exp_label=exp_label,
                                     exp_distribution=norm_mass_distribution_array,
                                     backexchange=backexchange_obj.backexchange_array,
                                     merge_exp=merge_exp,
                                     d2o_purity=d2o_purity,
                                     d2o_fraction=d2o_fraction)

        bkexch_corr_arr = gen_backexchange_correction_from_backexchange_array(backexchange_array=expdata_obj.backexchange)

    # initialize hxrate data object
    elapsed_time = []
    cpu_time = []

    if adj_backexchange:
        # set a flag for backexchange adjusting
        backexchange_adjust = False
        back_exchange_res_subtract = 0

        # set the original backexchange for reference
        init_backexchange = expdata_obj.backexchange[-1]
        backexchange_val_list = [init_backexchange]

        ratefit = BayesRateFit(num_chains=num_chains,
                               num_warmups=num_warmups,
                               num_samples=num_samples,
                               sample_backexchange=sample_backexchange)

        while backexchange_adjust is False:

            print('\nHX RATE FITTING ... ')

            # start timer here
            init_time = time.time()
            init_cpu_time = time.process_time()

            ratefit.fit_rate(exp_data_object=expdata_obj)

            elapsed_time.append(time.time() - init_time)
            cpu_time.append(time.process_time() - init_cpu_time)

            # get the difference of the slowest rate and 3rd slowest.
            rate_diff = ratefit.output['bayes_sample']['rate']['mean'][2] - ratefit.output['bayes_sample']['rate']['mean'][0]

            # if the rate difference is smaller than 1.6, hxrate optimization ends
            if rate_diff < slow_rates_max_diff:
                backexchange_adjust = True

            # else, backexchange is adjusted by reducing its value in proportion to having less residues.
            # then redo the hx rate fitting
            else:
                back_exchange_res_subtract += 1
                # if the backexchange res subtract exceeds the max res subtract, terminate the fitting routine
                if back_exchange_res_subtract > max_res_subtract_for_backexchange:
                    backexchange_adjust = True
                else:
                    print('adjusting backexchange value ... \n')
                    backexchange_value_adj = init_backexchange * ((len(ratefit.output['bayes_sample']['rate']['mean']) + back_exchange_res_subtract)/len(ratefit.output['bayes_sample']['rate']['mean']))

                    backexchange_adj_arr = gen_corr_backexchange(mass_rate_array=bkexch_corr_arr,
                                                                 fix_backexchange_value=backexchange_value_adj)

                    # adjust backexchange value on the expdata ratefit
                    expdata_obj.backexchange = backexchange_adj_arr
                    backexchange_val_list.append(backexchange_value_adj)

        ratefit.output['elapsed_time'] = elapsed_time
        ratefit.output['cpu_time'] = cpu_time
        ratefit.output['back_exchange_res_subtract'] = back_exchange_res_subtract
        ratefit.output['backexchange_val_list'] = backexchange_val_list

    else:

        print('\nHX RATE FITTING ... ')

        ratefit = BayesRateFit(num_chains=num_chains,
                               num_warmups=num_warmups,
                               num_samples=num_samples,
                               sample_backexchange=sample_backexchange)

        # start timer here
        init_time = time.time()
        init_cpu_time = time.process_time()

        ratefit.fit_rate(exp_data_object=expdata_obj)

        elapsed_time.append(time.time() - init_time)
        cpu_time.append(time.process_time() - init_cpu_time)

        ratefit.output['elapsed_time'] = elapsed_time
        ratefit.output['cpu_time'] = cpu_time

    return ratefit


def fit_rate_from_to_file(prot_name: str,
                          sequence: str,
                          hx_ms_dist_fpath: str or list,
                          d2o_fraction: float,
                          d2o_purity: float,
                          prot_rt_name: str = 'PROTEIN_RT',
                          merge_exp: bool = False,
                          exp_label: str or list = None,
                          usr_backexchange: float or list = None,
                          low_high_backexchange_list_fpath: str = None,
                          backexchange_corr_fpath: str or list = None,
                          backexchange_array_fpath: str or list = None,
                          adjust_backexchange: bool = True,
                          num_chains: int = 4,
                          num_warmups: int = 100,
                          num_samples: int = 500,
                          sample_backexchange: bool = False,
                          save_posterior_samples: bool = True,
                          hx_rate_output_path: str = None,
                          hx_rate_csv_output_path: str = None,
                          hx_isotope_dist_output_path: str = None,
                          merge_hx_ms_dist_output_path: str = None,
                          hx_rate_plot_path: str = None,
                          posterior_plot_path: str = None,
                          return_flag: bool = False) -> object:
    # todo: add param descriptions for the function here

    # make it accept list

    if merge_exp:
        timepoints, mass_dist_list, timepoint_label = [], [], []
        for hxmsdist_fp in hx_ms_dist_fpath:
            data_dict = load_data_from_hdx_ms_dist_(fpath=hxmsdist_fp)
            timepoints.append(data_dict['tp'])
            mass_dist_list.append(data_dict['mass_dist'])
            if 'tp_ind' in list(data_dict.keys()):
                timepoint_label.append(data_dict['tp_ind'])
            else:
                timepoint_label = None

        norm_dist = [normalize_mass_distribution_array(mass_dist_array=x) for x in mass_dist_list]

        bkexch_corr_dict = None
        if backexchange_corr_fpath is not None:
            bkexch_corr_dict = []
            for bkexch_corr_fpath_ in backexchange_corr_fpath:
                bkexch_corr_dict.append(load_tp_dependent_dict(bkexch_corr_fpath_))

        backexchange_array = None
        if backexchange_array_fpath is not None:
            backexchange_array = []
            for ind, bkexch_arr_fpath_ in enumerate(backexchange_array_fpath):
                bkexch_arr_dict = load_tp_dependent_dict(filepath=bkexch_arr_fpath_)
                backexchange_array_tp = np.zeros(len(timepoints[ind]))
                for ind_, tp_ in enumerate(timepoints[ind]):
                    backexchange_array_tp[ind_] = bkexch_arr_dict[tp_]
                backexchange_array.append(backexchange_array_tp)

        # get backexchange value from user list or backexchange list file
        backexchange_value = None

        # if backexchange list filepath is present
        if low_high_backexchange_list_fpath is not None:
            backexchange_value = []
            low_high_bkexch_df = pd.read_csv(low_high_backexchange_list_fpath)
            # todo: delete the following low_high_name df assignment after making sure that is already present in
            #  the csv file
            low_high_bkexch_df['low_high_name'] = np.array([x+'_'+y for x, y in zip(low_high_bkexch_df['low_ph_protein_name'].values, low_high_bkexch_df['high_ph_protein_name'].values)])
            low_high_bkexch_prot_df = low_high_bkexch_df[low_high_bkexch_df['low_high_name'] == prot_rt_name]
            low_user_backexchange = low_high_bkexch_prot_df['low_ph_backexchange_new'].values
            high_user_backexchange = low_high_bkexch_prot_df['high_ph_backexchange'].values

            # low_high_name should be unique so we expect a single value on the backexchange array
            # put the backexchange values in a list
            if len(low_user_backexchange) == 1:
                backexchange_value.append(low_user_backexchange[0])
            else:
                backexchange_value.append(None)

            if len(high_user_backexchange) == 1:
                backexchange_value.append(high_user_backexchange[0])
            else:
                backexchange_value.append(None)

        # if backexchange list filepath is not present, see if user backexchange has values and if not get a
        # list of Nones
        else:
            if usr_backexchange is None:
                backexchange_value = [None for _ in range(len(hx_ms_dist_fpath))]
            else:
                backexchange_value = usr_backexchange
    else:

        if type(hx_ms_dist_fpath) == list:
            hx_ms_dist_fpath = hx_ms_dist_fpath[0]

        # timepoints, mass_dist = load_data_from_hdx_ms_dist_(fpath=hx_ms_dist_fpath)
        data_dict = load_data_from_hdx_ms_dist_(fpath=hx_ms_dist_fpath)
        timepoints = data_dict['tp']
        mass_dist = data_dict['mass_dist']
        if 'tp_ind' in list(data_dict.keys()):
            timepoint_label = data_dict['tp_ind']
        else:
            timepoint_label = None

        norm_dist = normalize_mass_distribution_array(mass_dist_array=mass_dist)

        # get backexchange correction
        bkexch_corr_dict = None
        if backexchange_corr_fpath is not None:
            if type(backexchange_corr_fpath) == list:
                backexchange_corr_fpath = backexchange_corr_fpath[0]
            bkexch_corr_dict = load_tp_dependent_dict(filepath=backexchange_corr_fpath)

        # get backexchange array
        backexchange_array = None
        if backexchange_array_fpath is not None:
            if type(backexchange_array_fpath) == list:
                backexchange_array_fpath = backexchange_array_fpath[0]
            backexchange_array_dict = load_tp_dependent_dict(filepath=backexchange_array_fpath)
            backexchange_array = np.zeros(len(timepoints))
            for ind, tp in enumerate(timepoints):
                backexchange_array[ind] = backexchange_array_dict[tp]

        if type(usr_backexchange) == list:
            usr_backexchange = float(usr_backexchange[0])
        elif type(usr_backexchange) == float:
            usr_backexchange = usr_backexchange
        else:
            usr_backexchange = None
        backexchange_value = usr_backexchange

    # make sure exp label gets passed as a single string if its only one exp
    if type(exp_label) == list:
        if len(exp_label) > 1:
            exp_label = exp_label
        else:
            exp_label = exp_label[0]

    # fit rate
    hxrate_out = fit_rate_bayes_(prot_name=prot_name,
                                 sequence=sequence,
                                 time_points=timepoints,
                                 timepoint_label=timepoint_label,
                                 exp_label=exp_label,
                                 norm_mass_distribution_array=norm_dist,
                                 d2o_fraction=d2o_fraction,
                                 d2o_purity=d2o_purity,
                                 num_chains=num_chains,
                                 num_warmups=num_warmups,
                                 num_samples=num_samples,
                                 prot_rt_name=prot_rt_name,
                                 merge_exp=merge_exp,
                                 sample_backexchange=sample_backexchange,
                                 adj_backexchange=adjust_backexchange,
                                 backexchange_value=backexchange_value,
                                 backexchange_correction_dict=bkexch_corr_dict,
                                 backexchange_array=backexchange_array)

    # write hxrate as csv file
    if hx_rate_csv_output_path is not None:
        hxrate_out.write_rates_to_csv(hx_rate_csv_output_path)

    # write hxrate pred distributions as a csv file
    if hx_isotope_dist_output_path is not None:
        hxrate_out.pred_dist_to_csv(hx_isotope_dist_output_path)

    # write merge distribution
    if merge_exp:
        if merge_hx_ms_dist_output_path is not None:
            hxrate_out.exp_merge_dist_to_csv(merge_hx_ms_dist_output_path)

    # save the ratefit output dictionary to pickle
    if hx_rate_output_path is not None:
        hxrate_out.output_to_pickle(hx_rate_output_path, save_posterior_samples=save_posterior_samples)

    # plot distribution
    if hx_rate_plot_path is not None:
        hxrate_out.plot_hxrate_output(hx_rate_plot_path)

    # plot posterior distribution
    if posterior_plot_path is not None:
        hxrate_out.plot_bayes_samples(posterior_plot_path)

    if return_flag:
        return hxrate_out.output


def gen_parser_arguments():
    """
    generate commandline arguements to run the hx rate fitting algorithm
    :return:parser
    """

    parser = argparse.ArgumentParser(prog='HX_RATE_FIT', description='Run HX rate fitting algorithm')
    parser.add_argument('-p', '--protname', help='protein name', default='PROTEIN')
    parser.add_argument('-pr', '--protrtname', help='protein rt name', default='PROTEIN_RT')
    parser.add_argument('-s', '--sequence', help='protein sequence one letter amino acid', default='PROTEIN')
    parser.add_argument('-i', '--hxdist', nargs='+', help='hx mass distribution input file .csv')
    parser.add_argument('-df', '--d2o_frac', help='d2o fracion', type=float, default=0.95)
    parser.add_argument('-dp', '--d2o_pur', help='d2o purity', type=float, default=0.95)
    parser.add_argument('-ub', '--user_bkexchange', help='user defined backexchange', nargs='+', type=float or list, default=None)
    parser.add_argument('-expl', '--exp_label', help='Exp label in str or [str, str,..]', nargs='+', type=str or list, default=None)
    parser.add_argument('-bcl', '--backexchange_low_high_list_fpath', type=str, help='backexchange low high ph list filepath .csv', default=None)
    parser.add_argument('-bcf', '--bkexchange_corr_fpath', nargs='+', help='backexchange correction filepath .csv')
    parser.add_argument('-baf', '--bkexchange_array_fpath', nargs='+', help='backexchange array filepath .csv')
    parser.add_argument('--adjust_backexchange', help='adjust backexchange boolean', default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-nc', '--num_chains', help='number of independent markov chains for MCMC', type=int, default=4)
    parser.add_argument('-nw', '--num_warmups', help='number of warmups for MCMC', type=int, default=100)
    parser.add_argument('-ns', '--num_samples', help='number of samples for MCMC', type=int, default=500)
    parser.add_argument('--merge', help='merge distributions', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--sample_backexchange', help='sample backexchange for MCMC', default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_posterior_samples', help='save posterior samples', default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-o', '--output_pickle_file', help='output pickle filepath', default=None)
    parser.add_argument('-or', '--output_rate_csv', help='output rates csv filepath', default=None)
    parser.add_argument('-op', '--output_rate_plot', help='output_rate_plot filepath', default=None)
    parser.add_argument('-opp', '--output_posterior_plot', help='output_posterior_plot filepath', default=None)
    parser.add_argument('-od', '--output_iso_dist', help='output isotope distribution filepath', default=None)
    parser.add_argument('-md', '--merge_dist_output', help='merge distribution output .csv filepath', default=None)

    return parser


def hx_rate_fitting_from_parser(parser):
    """
    from the parser arguments, generate essential arguments for hx rate fitting function and run the function
    :param parser: parser
    :return:
    """

    options = parser.parse_args()

    fit_rate_from_to_file(prot_name=options.protname,
                          prot_rt_name=options.protrtname,
                          sequence=options.sequence,
                          hx_ms_dist_fpath=options.hxdist,
                          d2o_fraction=options.d2o_frac,
                          d2o_purity=options.d2o_pur,
                          usr_backexchange=options.user_bkexchange,
                          merge_exp=options.merge,
                          exp_label=options.exp_label,
                          backexchange_corr_fpath=options.bkexchange_corr_fpath,
                          backexchange_array_fpath=options.bkexchange_array_fpath,
                          low_high_backexchange_list_fpath=options.backexchange_low_high_list_fpath,
                          adjust_backexchange=options.adjust_backexchange,
                          num_chains=options.num_chains,
                          num_warmups=options.num_warmups,
                          num_samples=options.num_samples,
                          sample_backexchange=options.sample_backexchange,
                          hx_rate_output_path=options.output_pickle_file,
                          hx_rate_csv_output_path=options.output_rate_csv,
                          hx_isotope_dist_output_path=options.output_iso_dist,
                          hx_rate_plot_path=options.output_rate_plot,
                          merge_hx_ms_dist_output_path=options.merge_dist_output,
                          posterior_plot_path=options.output_posterior_plot)


if __name__ == '__main__':
    pass

    parser_ = gen_parser_arguments()
    hx_rate_fitting_from_parser(parser_)

    # from hxdata import load_pickle_object

    # pkobj_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/bug_nans_isotopdedist/PDB5X1G_8.57_PDB5X1G_7.69_hx_rate_fit.pickle'
    # hxobj = load_pickle_object(pkobj_fpath)
    #
    # low_ph_ms_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/bug_nans_isotopdedist/PDB5X1G_17.16_ph6_winner_multibody.cpickle.zlib.csv'
    # high_ph_ms_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/bug_nans_isotopdedist/PDB5X1G_17.16_ph9_winner_multibody.cpickle.zlib.csv'
    #
    # low_ph_bkexch_corr = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/bug_nans_isotopdedist/backexchange/low_ph_bkexch_corr.csv'
    # high_ph_bkexch_corr = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/bug_nans_isotopdedist/backexchange/high_ph_bkexch_corr.csv'
    #
    # high_low_bkexch_list = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/bug_nans_isotopdedist/backexchange/high_low_backexchange_list.csv'
    #
    # fit_rate_from_to_file(prot_name=hxobj['protein_name'],
    #                       prot_rt_name=hxobj['protein_rt_name'],
    #                       sequence=hxobj['sequence'],
    #                       hx_ms_dist_fpath=[low_ph_ms_fpath, high_ph_ms_fpath],
    #                       d2o_purity=0.95,
    #                       d2o_fraction=0.90,
    #                       merge_exp=True,
    #                       backexchange_corr_fpath=[low_ph_bkexch_corr, high_ph_bkexch_corr],
    #                       low_high_backexchange_list_fpath=high_low_bkexch_list,
    #                       num_chains=4,
    #                       num_warmups=150,
    #                       num_samples=250,
    #                       save_posterior_samples=True,
    #                       merge_hx_ms_dist_output_path=low_ph_ms_fpath+'_merge_dist.csv',
    #                       hx_rate_output_path=low_ph_ms_fpath+"_hx_rate.pickle",
    #                       hx_rate_csv_output_path=low_ph_ms_fpath+'_hx_rate.csv',
    #                       hx_isotope_dist_output_path=low_ph_ms_fpath+'_pred_dist.csv',
    #                       hx_rate_plot_path=low_ph_ms_fpath+"_hx_rate.pdf",
    #                       posterior_plot_path=low_ph_ms_fpath+'_posterior.pdf')

    # print('heho')
    #
    # eehee_rd4_0642_sequence = 'HMKTVEVNGVKYDFDNPEQAREMAERIAKSLGLQVRLEGDTFKIE'
    # eehee_rd4_0642_lowph_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/EEHEE_rd4_0642/EEHEE_rd4_0642.pdb_15.13925_winner_multibody.cpickle.zlib_lowph_newformat.csv'
    # eehee_rd4_0642_highph_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/EEHEE_rd4_0642/EEHEE_rd4_0642.pdb_15.13928_winner_multibody.cpickle.zlib_highph_newformat.csv'
    # prot_name = 'EEHEE_rd4_0642'
    # low_ph_prot_rt_name = 'EEHEE_rd4_0642.pdb_15.13925'
    # high_ph_prot_rt_name = 'EEHEE_rd4_0642.pdb_15.13928'
    # prot_rt_name = low_ph_prot_rt_name + '_' + high_ph_prot_rt_name
    #
    # eehee_rd4_08742_sequence = 'HMTQVHVDGVTYTFSNPEEAKKFADEMAKRKGGTWEIKDGHIHVE'
    # eehee_rd4_0871_low_ph_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/EEHEE_rd4_0871/EEHEE_rd4_0871.pdb_8.57176_winner_multibody.cpickle.zlib.csv'
    # eehee_rd4_0871_high_ph_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/EEHEE_rd4_0871/EEHEE_rd4_0871.pdb_8.57163_winner_multibody.cpickle.zlib.csv'
    #
    #
    # prot_name = 'eehee_rd4_0871'
    # prot_rt_name_low_ph = 'EEHEE_rd4_0871.pdb_8.57176_EEHEE_rd4_0871.pdb_8.57163'
    #
    # low_ph_bkexch_corr_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/backexchange/low_ph_bkexch_corr.csv'
    # high_ph_bkexch_corr_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/backexchange/high_ph_bkexch_corr.csv'
    # #
    # bkexch_list_high_low_fpath = '/Users/smd4193/OneDrive - Northwestern University/hx_ratefit_gabe/hxratefit_new/bayes_opt/test/merge_dist_rate_fit_bayes/Lib15/backexchange/high_low_backexchange_list.csv'
    #
    # fit_rate_from_to_file(prot_name=prot_name,
    #                       sequence=eehee_rd4_0642_sequence,
    #                       hx_ms_dist_fpath=[eehee_rd4_0642_lowph_fpath, eehee_rd4_0642_highph_fpath],
    #                       d2o_purity=0.95,
    #                       d2o_fraction=0.95,
    #                       prot_rt_name=prot_rt_name,
    #                       merge_exp=True,
    #                       exp_label=['ph6', 'ph9'],
    #                       backexchange_corr_fpath=[low_ph_bkexch_corr_fpath, high_ph_bkexch_corr_fpath],
    #                       low_high_backexchange_list_fpath=bkexch_list_high_low_fpath,
    #                       num_chains=4,
    #                       num_warmups=100,
    #                       num_samples=250,
    #                       adjust_backexchange=False,
    #                       save_posterior_samples=True,
    #                       merge_hx_ms_dist_output_path=None,
    #                       hx_rate_output_path=eehee_rd4_0642_lowph_fpath + '_merge_rate_output.pickle',
    #                       hx_rate_csv_output_path=eehee_rd4_0642_lowph_fpath + '_merge_rate.csv',
    #                       hx_isotope_dist_output_path=eehee_rd4_0642_lowph_fpath + '_merge_pred_dist.csv',
    #                       hx_rate_plot_path=eehee_rd4_0642_lowph_fpath + '_merge_rates.pdf',
    #                       posterior_plot_path=eehee_rd4_0642_lowph_fpath + '_merge_posteriors.pdf')
