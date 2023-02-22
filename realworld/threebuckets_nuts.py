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


    # NUTs parameters
    num_results = 100
    num_burnin_steps = 100
    dual_adaptation_ratio = 0.8
    num_chains = 1
    step_size = 0.1

    # Load data
    df = pd.read_pickle("data/megala_creek_australia.pkl.gz")

    # Slice out first three months of 1980
    df = df[(df['date'] >= '1980-01-01') & (df['date'] <= '1980-03-31')]

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
        "n": 3,
        "k": {"loc": jnp.array([0.0, 0.0, 0.0]),
            "scale": jnp.array([1.0, 1.0, 1.0])},
        "k_int": {"loc": jnp.array([0.0, 0.0]),
            "scale": jnp.array([1.0, 1.0])},
        "v_init": {"loc": tf.cast(0.0, dtype=jnp.float64),
            "scale": tf.cast(1.0, dtype=jnp.float64)},
        "v_max": {"loc": tf.cast(0.0, dtype=jnp.float64),
            "scale": tf.cast(1.0, dtype=jnp.float64)},
        "sigma": {"concentration": tf.cast(5.0, dtype=jnp.float64),
            "scale": tf.cast(0.1, dtype=jnp.float64)},
        "t_obs": t_obs,
        "precipitation": precipitation,
        "evapotranspiration": evapotranspiration,
        "discharge": discharge
        }


    dist = create_joint_posterior(model_prior_params)

    # TODO: Make truly random like Gaussian shells example
    # Random chain
    seed = SystemRandom().randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    key = random.PRNGKey(seed)

    key, subkey = jax.random.split(key)

    y_obs = discharge
    posterior = dist.experimental_pin(y=y_obs)

    #y_obs = jnp.load('./data_1980.npy', allow_pickle=True)
    #posterior = dist.experimental_pin(y=y_obs)

    bij = posterior.experimental_default_event_space_bijector()

    sampler = tfp.mcmc.TransformedTransitionKernel(
            tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=posterior.unnormalized_log_prob,
                step_size=0.1), bijector=bij)
    

    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=sampler,
            num_adaptation_steps=int(0.8 * num_burnin_steps),
            target_accept_prob=0.75)


    key, subkey = random.split(key)
    current_state = posterior.sample_unpinned(num_chains, seed=subkey)

    
    @jit
    def run_chain(key):
        """ Posterior samples
        """
        return tfp.mcmc.sample_chain(
                num_results=num_results,
                current_state=current_state,
                kernel=adaptive_sampler,
                num_burnin_steps=num_burnin_steps,
                #trace_fn=None,
                #num_steps_between_results=1,
                seed=key)


    run_nuts_jit = jit(run_chain)
    print("Running NUTS...")
    key, subkey = random.split(subkey)
    posterior_samples = run_nuts_jit(subkey)
    print("NUTS  finished.")

    print(posterior_samples)

    # posterior parameters beta=1
    #parameter_names = posterior._flat_resolve_names()
    #posterior_samp = {k: jnp.swapaxes( v, 0, 1) for k, v in zip(
    #        parameter_names, posterior_samples)}

    #az_trace = az.from_dict(posterior=posterior_samp)

    #print(az.summary(az_trace))
    
    #az.to_netcdf(az_trace, 'hbvresult4nuts')
    
    # Input parameters
    params["num_burnin_steps"] = num_burnin_steps
    params["dual_adaptation_ratio"] = dual_adaptation_ratio
    params["num_chains"] = num_chains
    params["step_size"] = step_size

    print("Finished.")


if __name__ == "__main__":
    print("Started.")

    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate the marginal likelihood")
    parser.add_argument('--output_dir', type=pathlib.Path, help="Output directory",
                        default="output")

    args = parser.parse_args()
    run_analysis(vars(args))
