# command line operation of hx rate fitting

import os

import pandas as pd

from methods import make_new_dir
import argparse
from hx_rate_fit import fit_rate_from_to_file
import yaml
from datetime import datetime


def gen_parser_arguments():
    """
    generate commandline arguements to run the hx rate fitting algorithm
    :return:parser
    """
    parser = argparse.ArgumentParser(prog='HX_RATE_FIT', description='Run HX rate fitting algorithm')
    parser.add_argument('-i', '--i_hxdist', help='hx mass distribution input file .csv',
                        default='../../workfolder/input_hx_dist/HEEH_rd4_0097_hx_mass_dist.csv')
    parser.add_argument('-s', '--sequence', help='protein sequence one letter amino acid',
                        default='HMDVEEQIRRLEEVLKKNQPVTWNGTTYTDPNEIKKVIEELRKSM')
    parser.add_argument('-n', '--prot_name', help='protein name', default='HEEH_rd4_0097')
    parser.add_argument('-p', '--i_params', help='params YAML file .yml file',
                        default='../../params/params.yml')
    parser.add_argument('-o', '--output_dir', help='top output dir path -> output_dir/prot_name for output files',
                        default='../../workfolder/output_hxrate')
    return parser


def gen_parser_args_string(parser_options):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    new_line = '\n'
    out_string = '#####' + new_line
    out_string += '# ' + dt_string + new_line
    out_string += '# HX RATE FITTING' + new_line
    out_string += '# PROT NAME: ' + parser_options.prot_name + new_line
    out_string += '# PROT SEQUENCE: ' + parser_options.sequence + new_line
    out_string += '# HX PARAMS: ' + parser_options.i_params + new_line
    out_string += '# HX MASS DIST: ' + parser_options.i_hxdist + new_line
    out_string += '# HX OUTPUT DIR: ' + parser_options.output_dir + new_line
    out_string += '#####' + new_line + new_line
    return out_string


def hx_rate_fitting_from_parser(parser):
    """
    from the parser arguments, generate essential arguments for hx rate fitting function and run the function
    :param parser: parser
    :return:
    """
    options = parser.parse_args()
    hx_mass_dist_fpath = options.i_hxdist
    hx_rate_params_fpath = options.i_params
    prot_name = options.prot_name
    prot_sequence = options.sequence
    output_dirpath = options.output_dir

    params_dict = yaml.load(open(hx_rate_params_fpath, 'rb'), Loader=yaml.Loader)

    prot_output_dirpath = make_new_dir(os.path.join(output_dirpath, prot_name))

    end_file_string = '_hx_rate_'

    hx_rate_output_path_ = os.path.join(prot_output_dirpath, prot_name + end_file_string + '.pickle')
    hx_rate_csv_output_path_ = os.path.join(prot_output_dirpath, prot_name + end_file_string + '.csv')
    hx_isotope_dist_output_path_ = os.path.join(prot_output_dirpath, prot_name + end_file_string + '_isotope_dist.csv')
    hx_rate_plot_path_ = os.path.join(prot_output_dirpath, prot_name + end_file_string + '.pdf')

    print(gen_parser_args_string(parser_options=options))

    fit_rate_from_to_file(prot_name=prot_name,
                          sequence=prot_sequence,
                          hx_ms_dist_fpath=hx_mass_dist_fpath,
                          d2o_fraction=params_dict['d2o_fraction'],
                          d2o_purity=params_dict['d2o_purity'],
                          opt_temp=params_dict['opt_temp'],
                          opt_iter=params_dict['opt_iter'],
                          opt_step_size=params_dict['opt_step_size'],
                          usr_backexchange=params_dict['usr_backexchange'],
                          backexchange_corr_fpath=params_dict['backexchange_correction_fpath'],
                          multi_proc=params_dict['multi_proc'],
                          number_of_cores=params_dict['number_of_cores'],
                          free_energy_values=None,
                          temperature=None,
                          hx_rate_output_path=hx_rate_output_path_,
                          hx_rate_csv_output_path=hx_rate_csv_output_path_,
                          hx_isotope_dist_output_path=hx_isotope_dist_output_path_,
                          hx_rate_plot_path=hx_rate_plot_path_)

    # do another rate fitting if backexchange correction fpath is given

    if params_dict['backexchange_correction_fpath'] is not None:

        bkcorr_df = pd.read_csv(params_dict['backexchange_correction_fpath'])
        col_names = list(bkcorr_df.columns)

        if prot_name in col_names:

            end_file_string += 'bkcorr_orig_'

            hx_rate_output_path_ = os.path.join(prot_output_dirpath, prot_name + end_file_string + '.pickle')
            hx_rate_csv_output_path_ = os.path.join(prot_output_dirpath, prot_name + end_file_string + '.csv')
            hx_isotope_dist_output_path_ = os.path.join(prot_output_dirpath, prot_name + end_file_string + 'isotope_dist.csv')
            hx_rate_plot_path_ = os.path.join(prot_output_dirpath, prot_name + end_file_string + '.pdf')

            print('\n###### HX RATE FITTING BACKEXCHANGE ORIGINAL CORRECTION ##########\n')
            print(gen_parser_args_string(parser_options=options))

            fit_rate_from_to_file(prot_name=prot_name,
                                  sequence=prot_sequence,
                                  hx_ms_dist_fpath=hx_mass_dist_fpath,
                                  d2o_fraction=params_dict['d2o_fraction'],
                                  d2o_purity=params_dict['d2o_purity'],
                                  opt_temp=params_dict['opt_temp'],
                                  opt_iter=params_dict['opt_iter'],
                                  opt_step_size=params_dict['opt_step_size'],
                                  usr_backexchange=params_dict['usr_backexchange'],
                                  backexchange_corr_fpath=params_dict['backexchange_correction_fpath'],
                                  backexchange_corr_prot_name=prot_name,
                                  multi_proc=params_dict['multi_proc'],
                                  number_of_cores=params_dict['number_of_cores'],
                                  free_energy_values=None,
                                  temperature=None,
                                  hx_rate_output_path=hx_rate_output_path_,
                                  hx_rate_csv_output_path=hx_rate_csv_output_path_,
                                  hx_isotope_dist_output_path=hx_isotope_dist_output_path_,
                                  hx_rate_plot_path=hx_rate_plot_path_)


if __name__ == '__main__':

    parser = gen_parser_arguments()
    hx_rate_fitting_from_parser(parser)
