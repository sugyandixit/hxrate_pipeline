import numpy as np
from sklearn.metrics import mean_squared_error
import numpyro
from jax import random
from numpyro.infer import NUTS, MCMC
import jax.numpy as jnp
import molmass

# global variables
r_constant = 0.0019872036


class BayesRateFit(object):
    """
    Bayes Rate Fit class
    """

    def __init__(self, num_chains=4, num_warmups=100, num_samples=500, return_posterior_distributions=True,
                 sample_backexchange=False):
        """
        initialize the class with mcmc key parameters
        :param num_chains: number of chains
        :param num_warmups: number of warmup or burn ins
        :param num_samples: number of samples for posterior
        :param return_posterior_distributions: bool. If True, returns all the posterior distributions
        :param sample_backexchange: If True, will sample backexchange for rate fitting
        """

        self.num_chains = num_chains
        self.num_warmups = num_warmups
        self.num_samples = num_samples
        self.return_posterior_distributions = return_posterior_distributions
        self.sample_backexchange = sample_backexchange
        self.output = None

    def fit_rate(self, sequence, timepoints, exp_distribution, back_exchange_array, d2o_fraction, d2o_purity):
        """
        fit rate using mcmc opt and generate key outputs in the self.output attribute
        :param sequence: protein sequence
        :param timepoints: timepoints array
        :param exp_distribution: experimental distribution array
        :param back_exchange_array: backexchange array
        :param d2o_fraction: d2o fraction
        :param d2o_purity: d2o purity
        :return: None
        """

        # set the number of cores to be the number of chains
        numpyro.set_host_device_count(n=self.num_chains)

        # generate a temporary rates to determine what residues can't exchange
        temp_rates = gen_temp_rates(sequence=sequence, rate_value=1)

        # get the indices of residues that don't exchange
        zero_indices = np.where(temp_rates == 0)[0]

        # calculate the number of residues that can exchange
        num_rates = len(temp_rates) - len(zero_indices)

        # number of bins in exp distribution
        num_bins = len(exp_distribution[0])

        # flatten experimental distribution and keep the non zero elements
        flat_exp_dist = np.concatenate(exp_distribution)
        non_zero_exp_dist_indices = np.nonzero(flat_exp_dist)[0]
        flat_exp_dist_non_zero = flat_exp_dist[non_zero_exp_dist_indices]

        # initialize the kernel for MCMC
        if self.sample_backexchange:
            nuts_kernel = NUTS(model=rate_fit_model_v2)
        else:
            nuts_kernel = NUTS(model=rate_fit_model)

        # initialize MCMC
        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmups, num_samples=self.num_samples, num_chains=self.num_chains,
                    chain_method='parallel')

        # gen random key
        rng_key = random.PRNGKey(0)

        # run mcmc
        mcmc.run(rng_key=rng_key,
                 num_rates=num_rates,
                 sequence=sequence,
                 timepoints=jnp.asarray(timepoints),
                 backexchange_array=jnp.asarray(back_exchange_array),
                 d2o_fraction=d2o_fraction,
                 d2o_purity=d2o_purity,
                 num_bins=num_bins,
                 obs_dist_nonzero_flat=jnp.asarray(flat_exp_dist_non_zero),
                 nonzero_indices=non_zero_exp_dist_indices,
                 extra_fields=('potential_energy',))

        # generate posterior samples but sort the rates in the posterior distribution
        posterior_samples_by_chain = sort_posterior_rates_in_samples(
            posterior_samples_by_chain=mcmc.get_samples(group_by_chain=True))

        # generate summary from the posterior samples
        summary_ = numpyro.diagnostics.summary(samples=posterior_samples_by_chain)

        # gen dictionary for output with key outputs
        self.output = dict()

        if self.return_posterior_distributions:
            self.output['posterior_samples'] = posterior_samples_by_chain
        else:
            self.output['posterior_samples'] = None

        self.output['rate'] = dict()
        self.output['rate']['mean'] = np.array(summary_['rate']['mean'])
        self.output['rate']['median'] = np.array(summary_['rate']['median'])
        self.output['rate']['std'] = np.array(summary_['rate']['std'])
        self.output['rate']['5percent'] = np.array(summary_['rate']['5.0%'])
        self.output['rate']['95percent'] = np.array(summary_['rate']['95.0%'])
        self.output['rate']['n_eff'] = np.array(summary_['rate']['n_eff'])
        self.output['rate']['r_hat'] = np.array(summary_['rate']['r_hat'])

        self.output['sigma'] = dict()
        self.output['sigma']['mean'] = summary_['sigma']['mean']
        self.output['sigma']['median'] = summary_['sigma']['median']
        self.output['sigma']['std'] = summary_['sigma']['std']
        self.output['sigma']['5percent'] = summary_['sigma']['5.0%']
        self.output['sigma']['95percent'] = summary_['sigma']['95.0%']
        self.output['sigma']['n_eff'] = summary_['sigma']['n_eff']
        self.output['sigma']['r_hat'] = summary_['sigma']['r_hat']

        if 'backexchange' in summary_.keys():
            self.output['backexchange'] = dict()
            self.output['backexchange']['mean'] = np.array(summary_['backexchange']['mean'])
            self.output['backexchange']['median'] = np.array(summary_['backexchange']['median'])
            self.output['backexchange']['std'] = np.array(summary_['backexchange']['std'])
            self.output['backexchange']['5percent'] = np.array(summary_['backexchange']['5.0%'])
            self.output['backexchange']['95percent'] = np.array(summary_['backexchange']['95.0%'])
            self.output['backexchange']['n_eff'] = np.array(summary_['backexchange']['n_eff'])
            self.output['backexchange']['r_hat'] = np.array(summary_['backexchange']['r_hat'])

            self.output['backexchange_sigma'] = dict()
            self.output['backexchange_sigma']['mean'] = summary_['backexchange_sigma']['mean']
            self.output['backexchange_sigma']['median'] = summary_['backexchange_sigma']['median']
            self.output['backexchange_sigma']['std'] = summary_['backexchange_sigma']['std']
            self.output['backexchange_sigma']['5percent'] = summary_['backexchange_sigma']['5.0%']
            self.output['backexchange_sigma']['95percent'] = summary_['backexchange_sigma']['95.0%']
            self.output['backexchange_sigma']['n_eff'] = summary_['backexchange_sigma']['n_eff']
            self.output['backexchange_sigma']['r_hat'] = summary_['backexchange_sigma']['r_hat']

            self.output['pred_distribution'] = np.array(
                gen_theoretical_isotope_dist_for_all_timepoints(sequence=sequence,
                                                                timepoints=jnp.asarray(timepoints),
                                                                rates=jnp.exp(summary_['rate']['mean']),
                                                                inv_backexchange_array=jnp.subtract(1, summary_[
                                                                    'backexchange']['mean']),
                                                                d2o_fraction=d2o_fraction,
                                                                d2o_purity=d2o_purity,
                                                                num_bins=num_bins))
        else:
            self.output['pred_distribution'] = np.array(
                gen_theoretical_isotope_dist_for_all_timepoints(sequence=sequence,
                                                                timepoints=jnp.asarray(timepoints),
                                                                rates=jnp.exp(summary_['rate']['mean']),
                                                                inv_backexchange_array=jnp.subtract(1, jnp.asarray(
                                                                    back_exchange_array)),
                                                                d2o_fraction=d2o_fraction,
                                                                d2o_purity=d2o_purity,
                                                                num_bins=num_bins))

        self.output['rmse'] = dict()
        self.output['rmse']['total'] = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=exp_distribution,
                                                                     thr_isotope_dist=self.output['pred_distribution'],
                                                                     squared=False)

        self.output['rmse']['per_timepoint'] = np.zeros(len(timepoints))
        for ind, (exp_dist, thr_dist) in enumerate(zip(exp_distribution, self.output['pred_distribution'])):
            self.output['rmse']['per_timepoint'][ind] = compute_rmse_exp_thr_iso_dist(exp_isotope_dist=exp_dist,
                                                                                      thr_isotope_dist=thr_dist,
                                                                                      squared=False)


def gen_temp_rates(sequence: str, rate_value: float = 1e2) -> np.ndarray:
    """
    generate template rates
    :param sequence: protein sequence
    :param rate_value: temporary rate value
    :return: an array of rates with first two residues and proline residues assigned to 0.0
    """
    rates = np.array([rate_value] * len(sequence), dtype=float)

    # set the rates for the first two residues as 0
    rates[:2] = 0

    # set the rate for proline to be 0
    if 'P' in sequence:
        amino_acid_list = [x for x in sequence]
        for ind, amino_acid in enumerate(amino_acid_list):
            if amino_acid == 'P':
                rates[ind] = 0

    return rates


def compute_rmse_exp_thr_iso_dist(exp_isotope_dist: np.ndarray,
                                  thr_isotope_dist: np.ndarray,
                                  squared: bool = False):
    """
    compute the mean squared error between exp and thr isotope distribution only with values of exp_dist that are higher
    than 0
    :param exp_isotope_dist:
    :param thr_isotope_dist:
    :param squared:
    :return:
    """
    exp_isotope_dist_comp = exp_isotope_dist[exp_isotope_dist > 0]
    thr_isotope_dist_comp = thr_isotope_dist[exp_isotope_dist > 0]
    rmse = mean_squared_error(exp_isotope_dist_comp, thr_isotope_dist_comp, squared=squared)
    return rmse


def PoiBin(success_probabilities):
    """
    poisson binomial probability distribution function
    :param success_probabilities:
    :return: probabiliy distribution
    """
    number_trials = success_probabilities.size

    omega = 2 * jnp.pi / (number_trials + 1)

    chi = jnp.empty(number_trials + 1, dtype=complex)
    chi = chi.at[0].set(1)
    # chi[0] = 1
    half_number_trials = int(
        number_trials / 2 + number_trials % 2)
    # set first half of chis:

    # idx_array = np.arange(1, half_number_trials + 1)
    exp_value = jnp.exp(omega * jnp.arange(1, half_number_trials + 1) * 1j)
    xy = 1 - success_probabilities + success_probabilities * exp_value[:, jnp.newaxis]
    # sum over the principal values of the arguments of z:
    argz_sum = jnp.arctan2(xy.imag, xy.real).sum(axis=1)
    # get d value:
    # exparg = np.log(np.abs(xy)).sum(axis=1)
    d_value = jnp.exp(jnp.log(jnp.abs(xy)).sum(axis=1))
    # get chi values:
    chi = chi.at[1:half_number_trials + 1].set(d_value * jnp.exp(argz_sum * 1j))
    # chi[1:half_number_trials + 1] = d_value * jnp.exp(argz_sum * 1j)

    # set second half of chis:

    chi = chi.at[half_number_trials + 1: number_trials + 1].set(jnp.conjugate(
        chi[1:number_trials - half_number_trials + 1][::-1]))

    # chi[half_number_trials + 1:number_trials + 1] = jnp.conjugate(
    #     chi[1:number_trials - half_number_trials + 1][::-1])
    chi /= number_trials + 1
    xi = jnp.fft.fft(chi)
    return xi.real


def theoretical_isotope_dist(sequence, num_isotopes=None):
    """
    calculate theoretical isotope distribtuion from a given one letter sequence of protein chain
    :param sequence: protein sequence in one letter code
    :param num_isotopes: number of isotopes to include. If none, includes all
    :return: isotope distribution
    """
    seq_formula = molmass.Formula(sequence)
    isotope_dist = jnp.array([x[1] for x in seq_formula.spectrum().values()])
    isotope_dist = isotope_dist / jnp.max(isotope_dist)
    if num_isotopes:
        if num_isotopes < len(isotope_dist):
            isotope_dist = isotope_dist[:num_isotopes]
        else:
            fill_arr = jnp.zeros(num_isotopes - len(isotope_dist))
            isotope_dist = jnp.append(isotope_dist, fill_arr)
    return isotope_dist


def calc_hx_prob(timepoint,
                 rate_constant,
                 inv_back_exchange,
                 d2o_purity,
                 d2o_fraction):
    """
    calculate the exchange probability for each residue at the timepoint given the rate constant, backexchange, d2o purity and fraction
    :param timepoint: timepoint in seconds
    :param rate_constant: array of rate_constant
    :param inv_back_exchange: 1 - backexchange value
    :param d2o_purity: d2o purity
    :param d2o_fraction: d2o fraction
    :return: array of probabilities
    """

    prob = (1.0 - jnp.exp(-rate_constant * timepoint)) * (d2o_fraction * d2o_purity * inv_back_exchange)
    return prob


def hx_rates_probability_distribution(timepoint,
                                      rates,
                                      inv_backexchange,
                                      d2o_fraction,
                                      d2o_purity,
                                      free_energy_values=None,
                                      temperature=None):
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

    fractions = jnp.array([1 for x in rates])
    if free_energy_values is not None:
        if temperature is None:
            raise ValueError('You need to specify temperature (K) in order to use free energy values')
        else:
            fractions = jnp.exp(-free_energy_values / (r_constant * temperature)) / (
                        1.0 + jnp.exp(-free_energy_values / (r_constant * temperature)))

    rate_constant_values = rates * fractions

    hx_probabs = calc_hx_prob(timepoint=timepoint,
                              rate_constant=rate_constant_values,
                              inv_back_exchange=inv_backexchange,
                              d2o_purity=d2o_purity,
                              d2o_fraction=d2o_fraction)

    return hx_probabs


def isotope_dist_from_PoiBin(sequence,
                             timepoint,
                             inv_backexchange,
                             rates: np.ndarray,
                             d2o_fraction,
                             d2o_purity,
                             num_bins,
                             free_energy_values=None,
                             temperature=None):
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

    isotope_dist_poibin = jnp.convolve(pmf_hx_probs, seq_isotope_dist)[:num_bins]
    isotope_dist_poibin_norm = isotope_dist_poibin / jnp.max(isotope_dist_poibin)

    return isotope_dist_poibin_norm


def gen_theoretical_isotope_dist_for_all_timepoints(sequence,
                                                    timepoints,
                                                    rates,
                                                    inv_backexchange_array,
                                                    d2o_fraction,
                                                    d2o_purity,
                                                    num_bins,
                                                    free_energy_values=None,
                                                    temperature=None):
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

    out_array = jnp.zeros((len(timepoints), num_bins))

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
        out_array = out_array.at[ind].set(isotope_dist)

    return out_array


def rate_fit_model(num_rates,
                   sequence,
                   timepoints,
                   backexchange_array,
                   d2o_fraction,
                   d2o_purity,
                   num_bins,
                   obs_dist_nonzero_flat,
                   nonzero_indices):
    """
    rate fit model for opt
    :param num_rates: number of rates
    :param sequence: protein sequence
    :param timepoints: timepoints array
    :param inv_backexchange_array:  1- backexchange array
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param num_bins: number of bins in integrated mz data
    :param obs_dist_nonzero_flat: observed experimental distribution flattened and with non zero values
    :param nonzero_indices: indices in exp distribution with non zero values
    :return: numpyro sample object
    """

    with numpyro.plate(name='rates', size=num_rates):
        rates_ = numpyro.sample(name='rate',
                                fn=numpyro.distributions.Uniform(low=-15, high=5))

    thr_dists = gen_theoretical_isotope_dist_for_all_timepoints(sequence=sequence,
                                                                timepoints=timepoints,
                                                                rates=jnp.exp(rates_),
                                                                inv_backexchange_array=jnp.subtract(1,
                                                                                                    backexchange_array),
                                                                d2o_fraction=d2o_fraction,
                                                                d2o_purity=d2o_purity,
                                                                num_bins=num_bins)

    flat_thr_dist = jnp.concatenate(thr_dists)
    flat_thr_dist_non_zero = flat_thr_dist[nonzero_indices]

    sigma = numpyro.sample(name='sigma',
                           fn=numpyro.distributions.Normal(loc=0.5, scale=0.5))

    with numpyro.plate(name='bins', size=len(flat_thr_dist_non_zero)):
        return numpyro.sample(name='bin_preds',
                              fn=numpyro.distributions.Normal(loc=flat_thr_dist_non_zero, scale=sigma),
                              obs=obs_dist_nonzero_flat)


def rate_fit_model_v2(num_rates,
                      sequence,
                      timepoints,
                      backexchange_array,
                      d2o_fraction,
                      d2o_purity,
                      num_bins,
                      obs_dist_nonzero_flat,
                      nonzero_indices):
    """
    rate fit model for opt
    :param num_rates: number of rates
    :param sequence: protein sequence
    :param timepoints: timepoints array
    :param inv_backexchange_array:  1- backexchange array
    :param d2o_fraction: d2o fraction
    :param d2o_purity: d2o purity
    :param num_bins: number of bins in integrated mz data
    :param obs_dist_nonzero_flat: observed experimental distribution flattened and with non zero values
    :param nonzero_indices: indices in exp distribution with non zero values
    :return: numpyro sample object
    """

    with numpyro.plate(name='rates', size=num_rates):
        rates_ = numpyro.sample(name='rate',
                                fn=numpyro.distributions.Uniform(low=-15, high=5))

    # todo: need to re evaluate the prior distribution params
    backexchange_sigma = numpyro.sample(name='backexchange_sigma',
                                        fn=numpyro.distributions.HalfNormal(scale=0.1))

    with numpyro.plate(name='backexchange_values', size=len(backexchange_array)):
        backexchange = numpyro.sample(name='backexchange',
                                      fn=numpyro.distributions.Normal(loc=backexchange_array,
                                                                      scale=backexchange_sigma))

    thr_dists = gen_theoretical_isotope_dist_for_all_timepoints(sequence=sequence,
                                                                timepoints=timepoints,
                                                                rates=jnp.exp(rates_),
                                                                inv_backexchange_array=jnp.subtract(1, backexchange),
                                                                d2o_fraction=d2o_fraction,
                                                                d2o_purity=d2o_purity,
                                                                num_bins=num_bins)

    flat_thr_dist = jnp.concatenate(thr_dists)
    flat_thr_dist_non_zero = flat_thr_dist[nonzero_indices]

    sigma = numpyro.sample(name='sigma',
                           fn=numpyro.distributions.Normal(loc=0.5, scale=0.5))

    with numpyro.plate(name='bins', size=len(flat_thr_dist_non_zero)):
        return numpyro.sample(name='bin_preds',
                              fn=numpyro.distributions.Normal(loc=flat_thr_dist_non_zero, scale=sigma),
                              obs=obs_dist_nonzero_flat)


def sort_posterior_rates_in_samples(posterior_samples_by_chain):
    """
    sort the posterior rates in each chain
    :param posterior_samples_by_chain: posterior samples by chain provided by mcmc
    :return: posterior samples by chain
    """

    for key, values in posterior_samples_by_chain.items():

        if key == 'rate':

            store_chain_values = []

            for ind, chain_values in enumerate(values):
                new_chain_values = jnp.sort(chain_values, axis=1)
                store_chain_values.append(new_chain_values)

            store_chain_array = jnp.array(store_chain_values)

            posterior_samples_by_chain['rate'] = store_chain_array

    return posterior_samples_by_chain


if __name__ == '__main__':
    pass
