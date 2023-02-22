import os
import pathlib
from random import SystemRandom
import subprocess

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
from jax import jit
from jax.random import PRNGKey
from jax.experimental.ode import odeint


jax.config.update("jax_enable_x64", True)

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

jax.config.update("jax_enable_x64", True)

# /Users/damian.ndiwago/Dropbox/Mac/Desktop/python/thesis-ideas
# /Users/damian.ndiwago/Dropbox/Mac/Desktop/python/thesis-ideas/tfponjax


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
# print('dta loaded successfully')
# Times to observe solution
T = jnp.float64(num_days)
t_obs = jnp.arange(0, num_days) + 0.5
precipitation = jnp.array(df['precipitation'], dtype=jnp.float64)
evapotranspiration = jnp.array(df['evapotranspiration'], dtype=jnp.float64)
discharge = jnp.array(df['observed_discharge'], dtype=jnp.float64)

