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
    num_betas = 15

    # HMC Parameters
    num_results = 3000
    num_burnin_steps = 1000
    dual_adaptation_ratio = 0.8
    num_chains = 1
    step_size = 0.01
    num_leapfrog_steps = 50

    # Load data
    df = pd.read_pickle("../../data/megala_creek_australia.pkl.gz")

    # Slice out first fivee months of 1980
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
        "n":2,
        "k": {"loc": jnp.log(jnp.array([0.8, 0.2])),
              "scale": jnp.array([0.25, 0.25])},
        "k_int": {"loc": jnp.array([0.6]),
                  "scale": jnp.array([0.25])},
        "v_init": {"loc": tf.cast(0.0, dtype=jnp.float64),
                   "scale": tf.cast(0.25, dtype=jnp.float64)},
        "v_max": {"loc": tf.cast(0.6, dtype=jnp.float64),
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
            #trace_fn=trace_swaps,
            seed=key)

    run_remc_chain_jit = jit(run_remc_chain)
    print("Running REMC...")
    key, subkey = random.split(subkey)
    posterior_samples = run_remc_chain_jit(subkey)
    print("REMC finished.")

    num_trues = tf.reduce_sum(tf.cast(posterior_samples.trace.is_swap_accepted_adjacent, tf.int32))

    print("Number of True values:", num_trues)
    num_trues = tf.reduce_sum(tf.cast(posterior_samples.trace.is_swap_accepted_adjacent, tf.float32))

    # Calculate the total number of elements
    total_elements = tf.size(posterior_samples.trace.is_swap_proposed_adjacent, out_type=tf.float32)

    # Calculate the proportion of True values
    proportion_trues = num_trues / total_elements

    print("Proportion of accepted:", proportion_trues)

    
    num_trues = tf.reduce_sum(tf.cast(posterior_samples.trace.is_swap_proposed_adjacent, tf.int32))

    print("Number of True values:", num_trues)
    num_trues = tf.reduce_sum(tf.cast(posterior_samples.trace.is_swap_proposed_adjacent, tf.float32))

    # Calculate the total number of elements
    total_elements = tf.size(posterior_samples.trace.is_swap_proposed_adjacent, out_type=tf.float32)

    # Calculate the proportion of True values
    proportion_trues = num_trues / total_elements

    print("Proportion of proposed:", proportion_trues)



    # Input parameters
    params["num_burnin_steps"] = num_burnin_steps
    params["dual_adaptation_ratio"] = dual_adaptation_ratio
    params["num_chains"] = num_chains
    params["step_size"] = step_size
    params["num_leapfrog_steps"] = num_leapfrog_steps

    git_describe = subprocess.check_output(
        ["git", "describe", "--always", "--dirty"]).strip().decode()

if __name__ == "__main__":
    print("Started.")

    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate the marginal likelihood")
    parser.add_argument('--output_dir', type=pathlib.Path, help="Output directory",
                        default="output")

    args = parser.parse_args()
    run_analysis(vars(args))
