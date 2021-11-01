# command line operation of hx rate fitting
import argparse
from hx_rate_fit import fit_rate_from_to_file


def gen_parser_arguments():
    """
    generate commandline arguements to run the hx rate fitting algorithm
    :return:parser
    """
    parser = argparse.ArgumentParser(prog='HX_RATE_FIT', description='Run HX rate fitting algorithm')
    parser.add_argument('-i', '--hxdist', help='hx mass distribution input file .csv',
                        default='../../workfolder/input_hx_dist/HEEH_rd4_0097_hx_mass_dist.csv')
    parser.add_argument('-s', '--sequence', help='protein sequence one letter amino acid',
                        default='HMDVEEQIRRLEEVLKKNQPVTWNGTTYTDPNEIKKVIEELRKSM')
    parser.add_argument('-n', '--prot_name', help='protein name', default='HEEH_rd4_0097')
    parser.add_argument('-df', '--d2o_frac', help='d2o_fraction', default=0.95)
    parser.add_argument('-dp', '--d2o_purity', help='d2o_purity', default=0.95)
    parser.add_argument('-ot', '--opt_temp', help='optimization_temperature', default=0.0003)
    parser.add_argument('-os', '--opt_size', help='optimization_step_size', default=0.02)
    parser.add_argument('-oi', '--opt_iter', help='optimization_iteration_number', default=50)
    parser.add_argument('-ub', '--user_backexchange', help='user_backexchange', default=None)
    parser.add_argument('-bcf', '--backexchange_corr_fpath', help='backexchange_correction_fpath', default=None)
    parser.add_argument('-baf', '--backexchange_array_fpath', help='backexchange_array filepath', default=None)
    parser.add_argument('-mp', '--multi_proc', help='multi_processing_bool', default=True)
    parser.add_argument('-nc', '--num_cores', help='number_of_cores', default=6)
    parser.add_argument('-fe', '--free_energy_values', help='free_energy_values', default=None)
    parser.add_argument('-ft', '--free_energy_temp', help='free_energy_calc_temp', default=None)
    parser.add_argument('-o', '--output_pickle_file', help='output pickle filepath')
    parser.add_argument('-or', '--output_rate_csv', help='output rates csv filepath')
    parser.add_argument('-op', '--output_rate_plot', help='output_rate_plot filepath')
    parser.add_argument('-od', '--output_iso_dist', help='output isotope distribution filepath')
    return parser


def hx_rate_fitting_from_parser(parser):
    """
    from the parser arguments, generate essential arguments for hx rate fitting function and run the function
    :param parser: parser
    :return:
    """

    options = parser.parse_args()

    if options.user_backexchange is None:
        user_backexchange = None
    else:
        user_backexchange = float(options.user_backexchange)

    fit_rate_from_to_file(prot_name=options.prot_name,
                          sequence=options.sequence,
                          hx_ms_dist_fpath=options.hxdist,
                          d2o_fraction=float(options.d2o_frac),
                          d2o_purity=float(options.d2o_purity),
                          opt_temp=float(options.opt_temp),
                          opt_iter=int(options.opt_iter),
                          opt_step_size=float(options.opt_size),
                          usr_backexchange=user_backexchange,
                          backexchange_corr_fpath=options.backexchange_corr_fpath,
                          backexchange_array_fpath=options.backexchange_array_fpath,
                          multi_proc=options.multi_proc,
                          number_of_cores=int(options.num_cores),
                          hx_rate_output_path=options.output_pickle_file,
                          hx_rate_csv_output_path=options.output_rate_csv,
                          hx_isotope_dist_output_path=options.output_iso_dist,
                          hx_rate_plot_path=options.output_rate_plot)


if __name__ == '__main__':

    parser = gen_parser_arguments()
    hx_rate_fitting_from_parser(parser)
