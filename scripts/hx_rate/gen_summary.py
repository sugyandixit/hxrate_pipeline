import argparse
from hxdata import write_merge_dist_summary, write_rate_fit_summary, write_dg_fit_summary


def write_summary(list_of_files, output_path, file_delim_str='', list_of_protein_names=None, mode='rate'):
    """

    :param list_of_files:
    :param output_path:
    :param file_delim_str:
    :param list_of_protein_names:
    :param mode:
    :return:
    """

    if mode == 'rate':

        write_rate_fit_summary(list_of_ratefit_pk_files=list_of_files,
                               output_fpath=output_path,
                               file_delim_string=file_delim_str)

    elif mode == 'merge':

        write_merge_dist_summary(list_of_csv_files=list_of_files,
                                 output_fpath=output_path,
                                 list_of_protein_names=list_of_protein_names,
                                 file_delim_string=file_delim_str)

    elif mode == 'dg':

        write_dg_fit_summary(list_of_dg_pk_files=list_of_files,
                             output_fpath=output_path,
                             file_delim_string=file_delim_str)

    else:

        print('Invalid writing mode. Use one of the following: rate, merge, dg')


def gen_parser_args():

    parser = argparse.ArgumentParser(prog='WRITE SUMMARY')
    parser.add_argument('-l', '--listfiles', nargs='+', help='list of files')
    parser.add_argument('-lp', '--listprotnames', nargs='+', default=None, help='list of protein names')
    parser.add_argument('-d', '--delim', type=str, default='', help='file delim string')
    parser.add_argument('-m', '--mode', type=str, choices=['rate', 'merge', 'dg'], help='summary writing mode')
    parser.add_argument('-o', '--output', type=str, help='output file path .csv')

    return parser


def run_from_parser():

    parse_ = gen_parser_args()
    options = parse_.parse_args()

    write_summary(list_of_files=options.listfiles,
                  output_path=options.output,
                  file_delim_str=options.delim,
                  list_of_protein_names=options.listprotnames,
                  mode=options.mode)


if __name__ == '__main__':

    run_from_parser()
