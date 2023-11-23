import jax.numpy as jnp


def make_inverse_temperature_schedule(num, power=5):
    return jnp.flip(jnp.power(jnp.arange(1, num + 1)/num, power))
