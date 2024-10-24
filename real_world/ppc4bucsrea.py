import os
import pathlib
from random import SystemRandom
import subprocess
import arviz as az

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp
from jax import jit
from jax.random import PRNGKey
from tensorflow_probability.substrates.jax import bijectors
from tensorflow_probability.substrates.jax.mcmc.transformed_kernel import \
    TransformedTransitionKernel

from hbv import create_joint_posterior
from utils import make_inverse_temperature_schedule
from jax import grad, vmap
from  ic import log_pw_pred_density, PWAIC_1, PWAIC_2, WAIC_1, WAIC_2, calculate_PD_1 , calculate_DIC_1, calculate_DIC_2


tf = tfp.tf2jax
tfd = tfp.distributions
tfb = tfp.bijectors

jax.config.update("jax_enable_x64", True)

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")


def run_analysis(params):
    output_dir = params["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Thermodynamic integration
    num_betas = 48

    # HMC Parameters
    num_results = 3000
    num_burnin_steps = 1000
    dual_adaptation_ratio = 0.8
    num_chains = 1
    step_size = 0.01
    num_leapfrog_steps = 50

    # Load data
    df = pd.read_pickle("../../data/megala_creek_australia.pkl.gz")

    # Slice out first five  months of 1980
    df = df[(df['date'] >= '1980-01-01') & (df['date'] <= '1980-05-31')]

    assert not np.any(np.isnan(df['precipitation']))
    assert not np.any(np.isnan(df['evapotranspiration']))

    print("Head of dataset")
    print(df.head())
    print("Tail of dataset")
    print(df.tail())

    t_start = 0.0  # days
    num_days = (df['date'].values[-1] - df['date'].values[0]
                ).astype('timedelta64[D]').astype(int) + 1

    # Times to observe solution
    T = jnp.float64(num_days)
    t_obs = jnp.arange(0, num_days) + 0.5
    precipitation = jnp.array(df['precipitation'], dtype=jnp.float64)
    evapotranspiration = jnp.array(df['evapotranspiration'], dtype=jnp.float64)
    discharge  = jnp.array(df['observed_discharge'], dtype=jnp.float64)
    # NOTE: Should discuss these parameters this week.
    # NOTE: These are not the same as the parameters in your scripts!
    
    model_prior_params = {
        "n":4,
        "k": {"loc": jnp.log(jnp.array([0.8, 0.2, 0.2, 0.2])),
              "scale": jnp.array([0.25, 0.25, 0.25, 0.25])},
        "k_int": {"loc": jnp.array([0.6, 0.6, 0.6]),
                  "scale": jnp.array([0.25, 0.25, 0.25])},
        "v_init": {"loc": tf.cast(0.0, dtype=jnp.float64),
                   "scale": tf.cast(0.25, dtype=jnp.float64)},
        "v_max": {"loc": tf.cast(0.0, dtype=jnp.float64),
                  "scale": tf.cast(0.25, dtype=jnp.float64)},
        "sigma": {"concentration": tf.cast(5.0, dtype=jnp.float64),
                  "scale": tf.cast(0.1, dtype=jnp.float64)},
        "t_obs": t_obs,
        "precipitation": precipitation,
        "evapotranspiration": evapotranspiration
    }


    dist = create_joint_posterior(model_prior_params)

    # TODO: Make truly random like Gaussian shells example
    # Random chain
    seed = SystemRandom().randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    key = random.PRNGKey(seed)

    key, subkey = jax.random.split(key)

    y_obs = discharge

    posterior = dist.experimental_pin(y=y_obs)
    
    def make_kernel_fn(target_log_prob_fn):
        kernel_hmc = tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size)
        kernel_dassa = tfp.mcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=kernel_hmc, num_adaptation_steps=int(0.8 * num_burnin_steps))
        return kernel_dassa

    # NOTE: I wonder if we could make this into a function in utils so
    # we do not need to repeat 1000 times
    inverse_temperatures = make_inverse_temperature_schedule(num_betas)

    def tempered_log_prob_fn(*samples):
        log_prob_parts = posterior.unnormalized_log_prob_parts(*samples)
        return log_prob_parts.pinned[0]

    def untempered_log_prob_fn(*samples):
        log_prob_parts = posterior.unnormalized_log_prob_parts(*samples)
        return sum(list(log_prob_parts.unpinned))

    remc = tfp.mcmc.ReplicaExchangeMC(
        target_log_prob_fn=None,
        untempered_log_prob_fn=untempered_log_prob_fn,
        tempered_log_prob_fn=tempered_log_prob_fn,
        inverse_temperatures=inverse_temperatures,
        make_kernel_fn=make_kernel_fn)

    def trace_swaps(unused_state, results):
        return (results.post_swap_replica_states)

    key, subkey = random.split(key)
    current_state = posterior.sample_unpinned(num_chains, seed=subkey)
    print(current_state)
    @jit
    def run_remc_chain(key):
        """Samples from tempered posterior with REMC
        """
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=current_state,
            kernel=remc,
            num_burnin_steps=num_burnin_steps,
            trace_fn=trace_swaps,
            seed=key)

    run_remc_chain_jit = jit(run_remc_chain)
    print("Running REMC...")
    key, subkey = random.split(subkey)
    posterior_samples, posterior_samples_betas = run_remc_chain_jit(subkey)
    print("REMC finished.")

    # posterior parameters beta=1
    parameter_names = posterior._flat_resolve_names()
    posterior_samp = {k: jnp.swapaxes( v, 0, 1) for k, v in zip(
            parameter_names, posterior_samples)}

    names = parameter_names
    
    for i in range(len(names)):
        names[i] = jnp.squeeze(posterior_samp.get(names[i]))
    post_save = pd.DataFrame(np.column_stack((names))) 
    post_save.to_csv('post4bucsrea.csv', index=False)

    def log_likelihood_fn(*samples):
        log_prob_parts = posterior.unnormalized_log_prob_parts(*samples)
        log_likelihood = log_prob_parts.pinned[0]
        return log_likelihood

    # posterior parameters beta=1
    ll = log_likelihood_fn(posterior_samples)
    az_trace = az.from_dict(posterior=posterior_samp,  observed_data={"observations": y_obs}, log_likelihood={'ll': ll})
    print(az.summary(az_trace))
    az.to_netcdf(az_trace, 'hbvresult4bucsrea')

    print("Calculating marginal likelihood.")
    mll = log_likelihood_fn(posterior_samples_betas)
    mll = jnp.mean(mll, axis=0)
    marginal_likelihood = -jnp.trapz(mll, inverse_temperatures, axis=0)

    print(f"Marginal likelihood: {marginal_likelihood}")

    df2 = pd.DataFrame(dict(mll=jnp.reshape(mll, num_betas), temp=inverse_temperatures))
    df2.to_csv('meanlogll41bucs.csv', index=False)

    # Input parameters
    params["num_burnin_steps"] = num_burnin_steps
    params["dual_adaptation_ratio"] = dual_adaptation_ratio
    params["num_chains"] = num_chains
    params["step_size"] = step_size
    params["num_leapfrog_steps"] = num_leapfrog_steps

    git_describe = subprocess.check_output(
        ["git", "describe", "--always", "--dirty"]).strip().decode()

    results = {"marginal_likelihood": marginal_likelihood,
               "mll": mll,
               "params": params,
               "git_describe": git_describe}

    print(f'Writing results to {params["output_dir"]/"results.npz"}...')
    np.savez(params["output_dir"] / "results.npz", **results)
    print("Finished.")
    
    keys = random.split(PRNGKey(2), num_results)
    posterior_predictive = jax.vmap(lambda key, sample: dist.sample(seed=key, value=sample))(keys, posterior_samples)

    # save the predicted dicharge values for post processing
    jnp.save('ppc4bucsrea', posterior_predictive[-1], allow_pickle=True)


    def least_squares_error(posterior_predictive, discharge):
        error = jnp.mean((posterior_predictive[-1] - discharge) ** 2)
        return error
    print(least_squares_error(posterior_predictive[-1], discharge))
    
    # For DIC
    # Calculate the deviance of the posterior mean
    def calculate_Dhat(m_bar):
        Dhat = -2 * log_likelihood_fn(m_bar)
        return Dhat

    def calculate_Dbar(Pos_draws):
        likelihood_samples = vmap(log_likelihood_fn)(Pos_draws)
        Dbar = -2 * jnp.mean(likelihood_samples)
        return Dbar

    # Calculate 1/2*(Deviance of the posterior variance) using method 2
    def calculate_PD_2(Pos_draws):
        likelihood_samples = vmap(log_likelihood_fn)(Pos_draws)
        PV = -0.5 * jnp.var(likelihood_samples)
        return PV

    k = jnp.mean(posterior_samples.k, axis=0)
    k_int = jnp.mean(posterior_samples.k_int, axis=0)
    v_init = jnp.mean(posterior_samples.v_init, axis=0)
    v_max = jnp.mean(posterior_samples.v_max, axis=0)
    sigma = jnp.mean(posterior_samples.sigma, axis=0)
    m_bar = [k, k_int, v_init, v_max, sigma ]
    Dbar = calculate_Dbar(posterior_samples)
    Dhat = calculate_Dhat(m_bar)
    PD_1 = calculate_PD_1(Dbar, Dhat)
    PD_2 = calculate_PD_2(posterior_samples)
    DIC_1 = calculate_DIC_1(Dhat, PD_1)
    DIC_2 = calculate_DIC_2(Dbar, PD_2)


    # BIC
    # Calculate the BIC
    bic = -2 * jnp.sum(log_likelihood_fn(m_bar)) + 13 * jnp.log(num_days)

    # For WAIC
    def pw_log_pred_density(posterior_samples):
        ppd = log_likelihood_fn(posterior_samples)
        return ppd

    plpd = pw_log_pred_density(posterior_samples)
    lppd = log_pw_pred_density(plpd)

    b = az.waic(az_trace, pointwise=True, scale='deviance')
    waic1 = WAIC_1(plpd)
    waic2 = WAIC_2(plpd)

    print("BIC:", bic)
    print("WAIC_1:", waic1)
    print("WAIC_2:", waic2)
    print("az_waic:", b)
    print("DIC_1:", DIC_1)
    print("DIC_2:", DIC_2)

if __name__ == "__main__":
    print("Started.")

    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate the marginal likelihood")
    parser.add_argument('--output_dir', type=pathlib.Path, help="Output directory",
                        default="output")

    args = parser.parse_args()
    run_analysis(vars(args))
