import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax.experimental.ode import odeint


tf = tfp.tf2jax
tfd = tfp.distributions
tfb = tfp.bijectors


def v(k, k_int, v_init, v_max, n, t_obs, precipitation, evapotranspiration):
    def extract_data(t: float, data) -> float:
        """Extracts the correct data from a daily timeseries at time t.

        Args:
            t: time
            data: data

        Returns:
            data_t [float]: The value of the data at t.
        """
        pos = jnp.floor(t).astype(jnp.int64)
        data_t = data[pos]
        return data_t

    def system_n(v, t, k, k_int, v_max, n):
        dv = jnp.zeros_like(v)

        dv = dv.at[0].set(extract_data(t, precipitation) - ((v[0] / v_max)
                          * extract_data(t, evapotranspiration)) - k[0] * v[0])

        for i in range(1, n):
            dv = dv.at[i].set(k_int[i] * v[i - 1] - k[i] * v[i])

        return dv

    system = jax.tree_util.Partial(system_n, n=n)

    vs = odeint(system, v_init, t_obs, k, k_int, v_max)
    return vs

def Q(k, k_int, v_init, v_max, n, t_obs, precipitation, evapotranspiration):
    vs = v(k, k_int, v_init, v_max, n, t_obs,
           precipitation, evapotranspiration)
    Qs = k*vs
    return jnp.sum(Qs, axis=1)


def create_joint_posterior(model_prior_params: dict):
    p = model_prior_params

    def model():
        k = yield tfd.Independent(tfd.LogNormal(loc=p["k"]["loc"],
                                                scale=p["k"]["scale"]), name='k', reinterpreted_batch_ndims=1)
        k_int = yield tfd.Independent(tfd.LogNormal(loc=p["k_int"]["loc"],
                                                    scale=p["k_int"]["scale"]), name='k_int', reinterpreted_batch_ndims=1)
        v_init = yield tfd.Sample(tfd.LogNormal(loc=p["v_init"]["loc"],
                                                scale=p["v_init"]["scale"]),
                                  sample_shape=(p["n"]), name='v_init')
        v_max = yield tfd.LogNormal(loc=p["v_max"]["loc"],
                                    scale=p["v_max"]["scale"],
                                    name='v_max')
#
        sigma = yield tfd.InverseGamma(concentration=p["sigma"]["concentration"], scale=p["sigma"]["scale"], name='sigma')

        y = yield tfd.MultivariateNormalDiag(loc=Q(k, k_int, v_init, v_max, p["n"], p["t_obs"], p["precipitation"], p["evapotranspiration"]), scale_identity_multiplier=sigma, name='y')

    return tfd.JointDistributionCoroutineAutoBatched(model)
