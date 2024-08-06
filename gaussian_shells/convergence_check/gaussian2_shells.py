import logging
import os
import pathlib
import pprint
from random import SystemRandom
import subprocess
import scienceplots

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import jit, random
from tensorflow_probability.substrates.jax.distributions import distribution
from tensorflow_probability.substrates.jax.internal import prefer_static as ps
from tensorflow_probability.substrates.jax.internal import parameter_properties
from tensorflow_probability.substrates.jax.internal import tensor_util

jax.config.update("jax_enable_x64", True)

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

tf = tfp.tf2jax

tfd = tfp.distributions

# Turn off typecheck warning
logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())


def run_analysis(params):
    output_dir = params["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Distribution
    dimension =  2.0

    # Thermodynamic integration
    num_betas = params["num_betas"]
    burn_in_ratio = 0.05
    # HMC Parameters
    num_results = 6000
    num_burnin_steps = 1000
    dual_adaptation_ratio = 0.8
    num_chains = 1
    step_size = 0.5
    num_leapfrog_steps = 20

    pprint.pp(locals())

    # Random chain
    seed = SystemRandom().randint(
        np.iinfo(
            np.uint32).min, np.iinfo(
            np.uint32).max)
    key = random.PRNGKey(seed)

    class GaussianShell(distribution.AutoCompositeTensorDistribution):

        def __init__(self,
                     theta,
                     validate_args=False,
                     allow_nan_stats=True,
                     name='GaussianShell'):

            parameters = dict(locals())
            with tf.name_scope(name) as name:
                dtype = theta.dtype
                self._theta = tensor_util.convert_nonref_to_tensor(
                    theta, dtype=dtype, name='theta')
                super(GaussianShell, self).__init__(
                    dtype=dtype,
                    reparameterization_type=tfd.FULLY_REPARAMETERIZED,
                    validate_args=validate_args,
                    allow_nan_stats=allow_nan_stats,
                    parameters=parameters,
                    name=name)

        @classmethod
        def _parameter_properties(cls, dtype, num_classes=None):
            # pylint: disable=g-long-lambda
            return dict(
                theta=parameter_properties.ParameterProperties())
            # pylint: enable=g-long-lambda

        @property
        def theta(self):
            """Distribution parameter for the mean."""
            return self._theta

        def _event_shape_tensor(self):
            return tf.constant([], dtype=tf.int32)

        def _event_shape(self):
            return tf.TensorShape([])

        def _sample_n(self, n, seed=None):
            theta = tf.convert_to_tensor(self.theta)

            shape = ps.concat(
                [[n], self._batch_shape_tensor(theta=theta)], axis=0)
            sampled = tf.zeros(shape=shape, dtype=self.dtype)

            return sampled

        def _log_prob(self, x):
            theta = tf.convert_to_tensor(self.theta)

            c_1 = tf.zeros(theta.shape[0])
            c_1 = c_1.at[0].set(-3.5)
            c_2 = tf.zeros(theta.shape[0])
            c_2 = c_2.at[0].set(+3.5)

            r = tf.constant([2.0])
            w = tf.constant([0.1])

            def log_gaussian_shell(theta, c):
                return -0.5 * (tf.linalg.norm(theta - c, axis=-1) - r)**2 / \
                    w**2 - tf.math.log(tf.sqrt(2 * np.pi * w**2))

            return tf.math.reduce_logsumexp(
                [log_gaussian_shell(theta, c_1), log_gaussian_shell(theta, c_2)])

    def model():
        theta = yield tfd.Sample(tfd.Uniform(tf.cast(-6.0, dtype='float64'), tf.cast(6.0, dtype='float64')), sample_shape=(dimension), name='theta')
        z = yield tfd.Sample(GaussianShell(theta), name='z')

    dist = tfd.JointDistributionCoroutineAutoBatched(model)
    posterior = dist.experimental_pin(z=1)

    inverse_temperatures = jnp.flip(jnp.asarray(
        a=[pow((i / (num_betas)), 5) for i in range(1, num_betas + 1)]))

    def make_kernel_fn(target_log_prob_fn):
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            num_leapfrog_steps=num_leapfrog_steps,
            step_size=step_size)
        return tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel, num_adaptation_steps=int(
                num_results * burn_in_ratio * dual_adaptation_ratio))

    key, subkey = random.split(key)

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

    def run_remc_chain(key):
        """Samples from tempered posterior with REMC
        """
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=current_state,
            kernel=remc,
            num_burnin_steps=num_results * burn_in_ratio,
            trace_fn=trace_swaps,
            seed=key)

    run_remc_chain_jit = jit(run_remc_chain)
    print("Running REMC...")
    posterior_samples, posterior_samples_betas = run_remc_chain_jit(subkey)
    print("REMC finished.")

    @jit
    def log_likelihood_fn(*samples):
        log_prob_parts = posterior.unnormalized_log_prob_parts(*samples)
        log_likelihood = log_prob_parts.pinned[0]
        return log_likelihood

    print("Calculating marginal likelihood.")
    mll = log_likelihood_fn(posterior_samples_betas)
    mll = jnp.mean(mll, axis=0)
    marginal_likelihood = -jnp.trapz(mll, inverse_temperatures, axis=0)

    print(f"Marginal likelihood: {marginal_likelihood}")

    # Input parameters
    params["burn_in_ratio"] = burn_in_ratio
    params["dual_adaptation_ratio"] = dual_adaptation_ratio
    params["num_chains"] = num_chains
    params["step_size"] = step_size
    params["num_leapfrog_steps"] = num_leapfrog_steps
    params["dimension"] = dimension

    git_describe = subprocess.check_output(
        ["git", "describe", "--always", "--dirty"]).strip().decode()

    results = {"marginal_likelihood": marginal_likelihood,
               "mll": mll,
               "params": params,
               "git_describe": git_describe}

    if params["output_posterior_samples_betas"]:
        results["posterior_samples_betas"] = posterior_samples_betas
    if params["output_posterior_samples"]:
        results["posterior_samples"] = posterior_samples

    print(f'Writing results to {params["output_dir"]/"results.npz"}...')
    np.savez(params["output_dir"] / "results.npz", **results)
    print("Finished.")


if __name__ == "__main__":
    print("Started.")
    
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate the marginal likelihood for a given number of temperature values")
    parser.add_argument('--num_betas', type=int,help="Number of temperatures", default=10)
    parser.add_argument('--output_posterior_samples_betas',
                        action='store_true', default=False)
    parser.add_argument('--output_posterior_samples',
                        action='store_true', default=True)
    parser.add_argument(
        '--output_dir',
        type=pathlib.Path,
        help="Output directory",
        default="output")

    args = parser.parse_args()

    run_analysis(vars(args))
