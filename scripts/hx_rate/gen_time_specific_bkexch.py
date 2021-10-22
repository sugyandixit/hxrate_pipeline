from hxdata import load_data_from_hdx_ms_dist_
from methods import gauss_fit_to_isotope_dist_array, normalize_mass_distribution_array
import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class MassRate(object):
    """
    class container to store timepoint specific back exchange data
    """
    protein_name: str = 'PROTEIN'
    rate_tol: float = 0.05
    frac_threshold: float = 0.50
    frac_threshold_ind: float = 0.0
    max_mass_rate: float = 1.0
    accept: bool = False
    frac_cal: float = None
    frac_cal_ind: float = None
    frac_start_ind: int = None
    frac_end_ind: int = None
    mass_rate_arr: np.ndarray = None

    def check_mass_rate_pass(self):

        mass_rate_arr_comp = self.mass_rate_arr[1:]
        mass_rate_arr_tol = mass_rate_arr_comp[np.abs(mass_rate_arr_comp) < self.rate_tol]
        self.frac_cal = len(mass_rate_arr_tol)/len(mass_rate_arr_comp)

        start_index = 2
        if self.frac_start_ind is not None:
            start_index = self.frac_start_ind

        end_index = int(len(self.mass_rate_arr[start_index:])/4)
        if self.frac_end_ind is not None:
            end_index = self.frac_end_ind

        mass_rate_arr_ind = self.mass_rate_arr[start_index:end_index+1]
        mass_rate_arr_ind_tol = mass_rate_arr_ind[np.abs(mass_rate_arr_ind) < self.rate_tol]
        self.frac_cal_ind = len(mass_rate_arr_ind_tol)/len(mass_rate_arr_ind)

        if max(np.abs(self.mass_rate_arr[1:])) < self.max_mass_rate:
            if self.frac_cal >= self.frac_threshold:
                if self.frac_cal_ind >= self.frac_threshold_ind:
                    self.accept = True


def calc_mass_diff_rate(ref_mass, mass_array):

    # todo: add param description

    diff_rate = np.zeros(len(mass_array))
    for ind, mass_val in enumerate(mass_array):
        rate = (mass_val - ref_mass)/ref_mass
        diff_rate[ind] = rate
    return diff_rate


def gen_mass_rate_obj(protein_name: str,
                      isotope_distribution: np.ndarray,
                      mass_rate_tol: float,
                      frac_thres: float,
                      frac_thres_ind: float,
                      start_ind: int,
                      end_ind: int,
                      max_mass_rate: float):

    # todo: add param description

    # fit gaussian to the isotope distributions
    gauss_fit = gauss_fit_to_isotope_dist_array(isotope_dist=isotope_distribution)

    # generate a centroid array
    centroid_array = np.array([x.centroid for x in gauss_fit])

    mass_rate_arr_ = calc_mass_diff_rate(ref_mass=centroid_array[-1], mass_array=centroid_array)

    # initialize mass rate object
    mass_rate_object = MassRate(protein_name=protein_name,
                                rate_tol=mass_rate_tol,
                                frac_threshold=frac_thres,
                                frac_threshold_ind=frac_thres_ind,
                                frac_start_ind=start_ind,
                                frac_end_ind=end_ind,
                                max_mass_rate=max_mass_rate,
                                mass_rate_arr=mass_rate_arr_)

    # check if mass rate object passes the tolerance
    mass_rate_object.check_mass_rate_pass()

    return mass_rate_object


def gen_mass_rate_obj_from_file(protein_name: str,
                                hx_ms_dist_fpath: str,
                                rate_tol: float,
                                frac_threshold: float,
                                frac_threshold_ind: float = 0.0,
                                start_ind: int = None,
                                end_ind: int = None,
                                max_mass_rate: float = 1.0):

    # todo: add param description

    tp, iso_dist = load_data_from_hdx_ms_dist_(hx_ms_dist_fpath)

    norm_dist = normalize_mass_distribution_array(mass_dist_array=iso_dist)

    mass_rate_object = gen_mass_rate_obj(protein_name=protein_name,
                                         isotope_distribution=norm_dist,
                                         mass_rate_tol=rate_tol,
                                         frac_thres=frac_threshold,
                                         frac_thres_ind=frac_threshold_ind,
                                         start_ind=start_ind,
                                         end_ind=end_ind,
                                         max_mass_rate=max_mass_rate)

    return mass_rate_object


def gen_list_of_mass_rate_obj(sample_csv_fpath: str,
                              rate_tol: float,
                              frac_threshold: float,
                              frac_thres_ind: float = 0.0,
                              start_ind: int = None,
                              end_ind: int = None,
                              max_mass_rate: float = 1.0):

    # todo: add param description

    sample_df = pd.read_csv(sample_csv_fpath)

    prot_names = sample_df.iloc[:, 0].values
    hx_ms_dist_fpaths = sample_df.iloc[:, 2].values

    list_of_mass_rate_obj = []

    for ind, (prot_name, hx_ms_dist_fpath) in enumerate(zip(prot_names, hx_ms_dist_fpaths)):

        #todo: investigate the source of error and create better solution than a general try and except

        try:
            mass_rate_obj = gen_mass_rate_obj_from_file(protein_name=prot_name,
                                                        hx_ms_dist_fpath=hx_ms_dist_fpath,
                                                        rate_tol=rate_tol,
                                                        frac_threshold=frac_threshold,
                                                        frac_threshold_ind=frac_thres_ind,
                                                        start_ind=start_ind,
                                                        end_ind=end_ind,
                                                        max_mass_rate=max_mass_rate)

            list_of_mass_rate_obj.append(mass_rate_obj)
        except:
            print('skipping file %s' % hx_ms_dist_fpath)

    return list_of_mass_rate_obj


def filter_mass_rate_object(list_of_mass_rate_obj, min_number_accept=1, ch_frac_threshold=0.02, max_iter_number=50):

    # todo: add param description

    accept_list = [x for x in list_of_mass_rate_obj if x.accept is True]
    reject_list = [x for x in list_of_mass_rate_obj if x.accept is False]

    iter_num = 1
    while len(accept_list) < min_number_accept:
        for ind, tp_bkexc_obj in enumerate(reject_list):
            tp_bkexc_obj.frac_threshold -= ch_frac_threshold
            tp_bkexc_obj.check_mass_rate_pass()
            if tp_bkexc_obj.accept:
                accept_list.append(tp_bkexc_obj)
                del reject_list[ind]
        iter_num += 1
        if iter_num > max_iter_number:
            min_number_accept -= 1

    new_list = accept_list + reject_list

    return new_list


def write_backexchange_corr_to_file(protein_list, mass_rate_list, csv_out_path):
    """
    write the backexchange correction to a csv file with the correction rate
    :param bakexchange_corr_obj: backexchange correction object
    :param csv_out_path: csv output path
    :return: None
    """

    protein_list_string = ''
    header = ''
    data_string = ''

    if len(mass_rate_list) == 1:
        protein_list_string += protein_list[0]
        header += 'idx,avg_dm_rate,'+protein_list_string+'\n'
        mass_rate_arr = mass_rate_list[0]
        for num in range(len(mass_rate_arr)):
            data_string += '{},{},{}\n'.format(num, mass_rate_arr[num], mass_rate_arr[num])

    else:
        protein_list_string += ','.join([x for x in protein_list])
        header += 'idx,avg_dm_rate,'+protein_list_string+'\n'
        mass_rate_arr = np.array(mass_rate_list)
        average_mass_rate = np.average(mass_rate_arr, axis=0)
        mass_rate_arr_t = mass_rate_arr.T
        for num in range(len(average_mass_rate)):
            mass_rate_arr_string = ','.join([str(x) for x in mass_rate_arr_t[num]])
            data_string += '{},{},{}\n'.format(num, average_mass_rate[num], mass_rate_arr_string)

    with open(csv_out_path, 'w') as outfile:
        outfile.write(header + data_string)
        outfile.close()


def plot_mass_rate_all(list_of_tp_bkexchange_obj, output_path):

    # todo: add param description

    false_color = 'grey'
    true_color = 'red'

    accept_list = [x for x in list_of_tp_bkexchange_obj if x.accept is True]
    reject_list = [x for x in list_of_tp_bkexchange_obj if x.accept is False]

    max_len_tp = 0

    fig, ax = plt.subplots()

    for tp_obj in reject_list:

        if max(np.abs(tp_obj.mass_rate_arr)) < 1.0:

            rev_ind = np.arange(len(tp_obj.mass_rate_arr))
            plt.plot(rev_ind, tp_obj.mass_rate_arr, color=false_color, alpha=0.1)

            if len(rev_ind) > max_len_tp:
                max_len_tp = len(rev_ind)

    for tp_obj in accept_list:

        rev_ind = np.arange(len(tp_obj.mass_rate_arr))
        plt.plot(rev_ind, tp_obj.mass_rate_arr, color=true_color, ls='--', linewidth=1)

        if len(rev_ind) > max_len_tp:
            max_len_tp = len(rev_ind)

    if len(accept_list) == 1:
        average_rate_arr = accept_list[0].mass_rate_arr
    else:
        mass_rate_arr_ = np.array([x.mass_rate_arr for x in accept_list])
        average_rate_arr = np.average(mass_rate_arr_, axis=0)

    plt.plot(np.arange(len(average_rate_arr)), average_rate_arr, color='black', linewidth=1)

    plt.xticks(range(0, max_len_tp + 1, 1))
    ax.set_xticklabels(range(0, max_len_tp + 1, 1))
    plt.grid(axis='x', alpha=0.25)
    plt.grid(axis='y', alpha=0.25)

    plt.xlabel('Timepoint Index')
    plt.ylabel('dM rate')

    plt.savefig(output_path)
    plt.close()


def gen_backexchange_corr_obj_from_sample_list(sample_list_fpath: str,
                                               rate_tol: float,
                                               frac_threshold: float,
                                               frac_threshold_bound: float,
                                               start_bound: int,
                                               end_bound: int,
                                               max_rate: float,
                                               min_number_paths: int,
                                               change_frac_threshold: float,
                                               plot_rate_path: str,
                                               csv_out_path: str,
                                               return_flag: bool) -> list:
    """

    :param sample_list_fpath:
    :param rate_tol:
    :param frac_threshold:
    :param frac_threshold_bound:
    :param start_bound:
    :param end_bound:
    :param max_rate:
    :param min_number_paths:
    :param change_frac_threshold:
    :param plot_rate_path:
    :param pickle_out_path:
    :param csv_out_path:
    :param return_flag:
    :return:
    """

    # generate a list of mass rate object from the sample list

    list_of_mass_rate_obj = gen_list_of_mass_rate_obj(sample_csv_fpath=sample_list_fpath,
                                                      rate_tol=rate_tol,
                                                      frac_threshold=frac_threshold,
                                                      frac_thres_ind=frac_threshold_bound,
                                                      start_ind=start_bound,
                                                      end_ind=end_bound,
                                                      max_mass_rate=max_rate)

    # filter the mass rate object
    new_mass_rate_object_list = filter_mass_rate_object(list_of_mass_rate_obj=list_of_mass_rate_obj,
                                                        min_number_accept=min_number_paths,
                                                        ch_frac_threshold=change_frac_threshold)

    if plot_rate_path is not None:
        plot_mass_rate_all(list_of_tp_bkexchange_obj=new_mass_rate_object_list,
                           output_path=plot_rate_path)

    if csv_out_path is not None:
        mass_rate_array_list = [x.mass_rate_arr for x in new_mass_rate_object_list if x.accept is True]
        protein_list = [x.protein_name for x in new_mass_rate_object_list if x.accept is True]
        write_backexchange_corr_to_file(protein_list=protein_list,
                                        mass_rate_list=mass_rate_array_list,
                                        csv_out_path=csv_out_path)

    if return_flag:
        return new_mass_rate_object_list


def gen_parser_arguments():
    """
    generate commandline arguements to generate backexchange correction file
    :return:parser
    """
    parser = argparse.ArgumentParser(prog='Backexchange correction', description='Generate backexchange correction path')
    parser.add_argument('-s', '--sample_fpath', help='file path for sample.csv', default='../../workfolder/sample.csv')
    parser.add_argument('-r', '--rate_tol', help='rate tolerance', default=0.05)
    parser.add_argument('-f', '--frac_threshold', help='fraction threshold', default=0.50)
    parser.add_argument('-b', '--frac_threshold_bound', help='fraction threshold bound', default=0.8)
    parser.add_argument('-i', '--start_bound', help='start bound', default=1)
    parser.add_argument('-e', '--end_bound', help='end bound', default=4)
    parser.add_argument('-m', '--max_tolerance', help='max rate tolerance', default=0.5)
    parser.add_argument('-n', '--min_paths', help='minimum number of paths', default=1)
    parser.add_argument('-c', '--change_frac_threshold', help='change fraction threshold', default=0.01)
    parser.add_argument('-p', '--plot_path', help='plot file path', default='../../workfolder/bkexch_corr.pdf')
    parser.add_argument('-o', '--output_path', help='correction file path', default='../../workfolder/bkexch_corr.csv')
    parser.add_argument('-g', '--return_flag', help='return the value bool', default=False)

    return parser


def gen_bkexch_corr_obj_from_parser(parser):

    options = parser.parse_args()

    gen_backexchange_corr_obj_from_sample_list(sample_list_fpath=options.sample_fpath,
                                               rate_tol=float(options.rate_tol),
                                               frac_threshold=float(options.frac_threshold),
                                               frac_threshold_bound=float(options.frac_threshold_bound),
                                               start_bound=int(options.start_bound),
                                               end_bound=int(options.end_bound),
                                               max_rate=float(options.max_tolerance),
                                               min_number_paths=int(options.min_paths),
                                               change_frac_threshold=float(options.change_frac_threshold),
                                               plot_rate_path=options.plot_path,
                                               csv_out_path=options.output_path,
                                               return_flag=options.return_flag)


if __name__ == '__main__':

    parser_ = gen_parser_arguments()
    gen_bkexch_corr_obj_from_parser(parser_)

