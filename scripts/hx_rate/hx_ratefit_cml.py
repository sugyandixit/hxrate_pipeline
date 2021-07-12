# command line operation of hx rate fitting

import os
import argparse
from hx_rate_fit import fit_rate_from_to_file


def gen_parser_arguments():
    """
    generate commandline arguements to run the hx rate fitting algorithm
    :return:parser
    """
    parser = argparse.ArgumentParser(prog='HX_RATE_FIT', description='Run HX rate fitting algorithm')
    parser.add_argument('-i', '--i_hxdist', help='hx mass distribution input file .csv', required=True)
    parser.add_argument('-n', '--prot_name', help='protein name', required=True)
    parser.add_argument('-p', '--i_params', help='params .csv file', required=False)
    parser.add_argument('-o', '--output_dir', help='top output dir path -> output_dir/prot_name for output files',
                        required=True)
    return parser


def make_new_dir(dirpath):
    """
    make a new directory if the directory doesn't already exists
    :param dirpath: directory path
    :return: directory path
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


def gen_params_from_file(param_fpath):
    """
    generate params from the param file
    :param param_fpath: param file csv
    :return: dictionary of params
    """
    file_dict = dict()
    with open(param_fpath, 'r') as inputfile:
        param_lines = inputfile.read().splitlines()
        for line in param_lines:
            chars = line.split(',')
            file_dict[chars[0]] = chars[1]
    return file_dict


def hx_rate_fitting_from_parser(parser):
    """
    from the parser arguments, generate essential arguments for hx rate fitting function and run the function
    :param parser: parser
    :return:
    """
    options, args = parser.parse_args()
    hx_mass_dist_fpath = options.i_hxdist
    hx_rate_params_fpath = options.i_params
    prot_name = options.prot_name
    output_dirpath = options.output_dir

    params_dict = gen_params_from_file(hx_rate_params_fpath)

    prot_output_dirpath = make_new_dir(os.path.join(output_dirpath, prot_name))

    if params_dict['d2o_fraction'] == '':
        d2o_fraction = 0.95
        print('setting the default d2o fraction to %.2f (Put a value in the params file to use your own)' % d2o_fraction)
    else:
        d2o_fraction = float(params_dict['d2o_fraction'])

    if params_dict['d2o_purity'] == '':
        d2o_purity = 0.95
        print('setting the default d2o purity to %.2f (Put a value in the params file to use your own)' % d2o_purity)
    else:
        d2o_purity = float(params_dict['d2o_purity'])

    if params_dict['opt_iter'] == '':
        opt_iter = 30
        print('setting the default opt_iter to %i (Put a value in the params file to use your own)' % opt_iter)
    else:
        opt_iter = int(params_dict['opt_iter'])

    if params_dict['opt_temp'] == '':
        opt_temp = 0.00003
        print('setting the default opt_temp to %.5f (Put a value in the params file to use your own)' % opt_temp)
    else:
        opt_temp = float(params_dict['opt_temp'])

    if params_dict['opt_step_size'] == '':
        opt_step_size = 0.02
        print('setting the default opt_step_size to %.2f (Put a value in the params file to use your own)' % opt_step_size)
    else:
        opt_step_size = float(params_dict['opt_step_size'])

    if params_dict['multi_proc'] == '':
        multi_proc = True
        print('setting multiprocessing to True')
    else:
        if params_dict['multi_proc'] == 'True':
            multi_proc = True
        elif params_dict['multi_proc'] == 'False':
            multi_proc = False
        else:
            multi_proc = True
            print('cannot recognize the multi proc param, setting it to False')

    if params_dict['number_of_cores'] == '':
        num_cores = 6
        print('setting the number of cores to 6')
    else:
        num_cores = int(params_dict['number_of_cores'])
