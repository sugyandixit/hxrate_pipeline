
import numpy as np
import numpyro
from jax import random
from numpyro.infer import NUTS, MCMC
from methods import gen_temp_rates, compute_rmse_exp_thr_iso_dist
import jax.numpy as jnp
import molmass

# global variables
r_constant = 0.0019872036


class BayesRateFit(object):

    def __init__(self, num_chains=4, num_warmups=1000, num_samples=500, return_posterior_distributions=False):

        self.num_chains = num_chains
        self.num_warmups = num_warmups
        self.num_samples = num_samples
        self.return_posterior_distributions = return_posterior_distributions
        self.output = None

    def fit_rate(self, sequence, timepoints, exp_distribution, back_exchange_array, d2o_fraction, d2o_purity):

        inv_backexchange_array = np.subtract(1, back_exchange_array)
        num_rates = len(gen_temp_rates(sequence=sequence))

        num_bins = len(exp_distribution[0])

        flat_exp_dist = np.concatenate(exp_distribution)
        non_zero_exp_dist_indices = np.nonzero(flat_exp_dist)[0]
        flat_exp_dist_non_zero = flat_exp_dist[non_zero_exp_dist_indices]

        nuts_kernel = NUTS(model=rate_fit_model)
        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmups, num_samples=self.num_samples, num_chains=self.num_chains)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key=rng_key,
                 num_rates=num_rates,
                 sequence=sequence,
                 timepoints=jnp.asarray(timepoints),
                 inv_backexchange_array=jnp.asarray(inv_backexchange_array),
                 d2o_fraction=d2o_fraction,
                 d2o_purity=d2o_purity,
                 num_bins=num_bins,
                 obs_dist_nonzero_flat=jnp.asarray(flat_exp_dist_non_zero),
                 nonzero_indices=non_zero_exp_dist_indices,
                 extra_fields=('potential_energy',))

        posterior_samples_by_chain = sort_posterior_rates_in_samples(posterior_samples_by_chain=mcmc.get_samples(group_by_chain=True))

        summary_ = numpyro.diagnostics.summary(samples=posterior_samples_by_chain)

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

        self.output['pred_distribution'] = np.array(gen_theoretical_isotope_dist_for_all_timepoints(sequence=sequence,
                                                                                                    timepoints=jnp.asarray(timepoints),
                                                                                                    rates=jnp.exp(summary_['rate']['mean']),
                                                                                                    inv_backexchange_array=jnp.asarray(inv_backexchange_array),
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
    isotope_dist = isotope_dist/jnp.max(isotope_dist)
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
            fractions = jnp.exp(-free_energy_values / (r_constant * temperature)) / (1.0 + jnp.exp(-free_energy_values / (r_constant * temperature)))

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
    isotope_dist_poibin_norm = isotope_dist_poibin/jnp.max(isotope_dist_poibin)

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
                      inv_backexchange_array,
                      d2o_fraction,
                      d2o_purity,
                      num_bins,
                      obs_dist_nonzero_flat,
                      nonzero_indices):

    with numpyro.plate(name='rates', size=num_rates):

        rates_ = numpyro.sample(name='rate',
                                fn=numpyro.distributions.Uniform(low=-15, high=5))

    thr_dists = gen_theoretical_isotope_dist_for_all_timepoints(sequence=sequence,
                                                                timepoints=timepoints,
                                                                rates=jnp.exp(rates_),
                                                                inv_backexchange_array=inv_backexchange_array,
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


if __name__=='__main__':

    pass

