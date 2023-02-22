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


tf = tfp.tf2jax
tfd = tfp.distributions
tfb = tfp.bijectors

jax.config.update("jax_enable_x64", True)

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")


def run_analysis( ):

    # Number of  Parameters
    num_results = 5000
    

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
        "sigma": {"concentration": tf.cast(1.0, dtype=jnp.float64),
                  "scale": tf.cast(1.0, dtype=jnp.float64)},
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


    *prior, prior_obs = dist.sample(num_results, seed=subkey)
    jnp.save('./prior_obs', prior_obs, allow_pickle=True)

    print("REMC finished.")
if __name__ == "__main__":

    run_analysis( )
