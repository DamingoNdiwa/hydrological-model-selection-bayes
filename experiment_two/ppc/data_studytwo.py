import os
import pathlib
import subprocess

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


def run_analysis():
    
    # Load data
    df = pd.read_pickle("../../../data/megala_creek_australia.pkl.gz")

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
        "k": {"loc": jnp.array([1.0, 0.6, 0.3]),
              "scale": jnp.array([0.25, 0.25, 0.25])},
        "k_int": {"loc": jnp.array([0.8, 0.4]),
                  "scale": jnp.array([0.25, 0.25])},
        "v_init": {"loc": tf.cast(0.0, dtype=jnp.float64),
                   "scale": tf.cast(1.0, dtype=jnp.float64)},
        "v_max": {"loc": tf.cast(1.0, dtype=jnp.float64),
                  "scale": tf.cast(0.25, dtype=jnp.float64)},
        "sigma": {"concentration": tf.cast(5.0, dtype=jnp.float64),
                  "scale": tf.cast(0.1, dtype=jnp.float64)},
        "t_obs": t_obs,
        "precipitation": precipitation,
        "evapotranspiration": evapotranspiration
    }

    dist = create_joint_posterior(model_prior_params)

    # TODO: Make truly random like Gaussian shells example
    key = PRNGKey(1569)
    key, subkey = jax.random.split(key)
    
    k, k_int, v_init, v_max, sigma,  y_obs = dist.sample(1,seed=subkey)
    
    print(f'the true value for is ks: {k}\n')
    print(f'the true value for k_int: {k_int}\n')
    print(f'the true value for v_init: {v_init}\n')
    print(f'the true value for v_max: {v_max}\n')
    print(f'the true value for sigma: {sigma}\n')

    return  y_obs 

y_obs = run_analysis()

jnp.save('./data_1980', y_obs, allow_pickle=True)



