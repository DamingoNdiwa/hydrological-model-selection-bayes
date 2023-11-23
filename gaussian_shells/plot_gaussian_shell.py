import jax.numpy as jnp
from jax.numpy.linalg import norm
from jax.numpy import log, sqrt, pi
from matplotlib import pyplot as plt


def loglikelihood(theta, c1=jnp.array([-3.5, 0.]), c2=jnp.array([3.5, 0.])):
    """gives the likelihodd for gaussian shells"""
    def logcirc(theta, c, r=jnp.array(2.0), w=jnp.array(0.1)):
        logll = -0.5 * (norm(theta - c, axis=-1) - r)**2 / \
            w**2 - log(sqrt(2 * pi * w**2))
        return logll
    return jnp.logaddexp(logcirc(theta, c1), logcirc(theta, c2))


# likelihood  in 2-d
xx, yy = jnp.meshgrid(jnp.linspace(-6., 6., 200),
                      jnp.linspace(-6., 6., 200))
z = jnp.exp(loglikelihood(jnp.dstack((xx, yy))))


fig = plt.figure(111, figsize=(8., 8.))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xx, yy, z, rstride=1, cstride=1, linewidth=0)
ax.set_xlim(-6., 6.)
ax.set_ylim(-6., 6.)
ax.set_zlim(0., 4.)
ax.set_zlabel(r'$L$', rotation='horizontal')
ax.grid(False)
ax.azim = 250
ax.set_title('Likelihood evaluations')
fig.savefig("./rings_cylinder.pdf", dpi=150)


# Plot PDF contours.
x = jnp.linspace(-6., 6., int(1e4), dtype=jnp.float32)


def meshgrid(x, y=x):
    [gx, gy] = jnp.meshgrid(x, y, indexing='ij')
    gx, gy = jnp.float32(gx), jnp.float32(gy)
    grid = jnp.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)


grid = meshgrid(jnp.linspace(-6, 6, 100, dtype=jnp.float32))
fig, ax = plt.subplots(figsize=(8., 5.))
ab = ax.contourf(grid[..., 0], grid[..., 1], loglikelihood(grid))
fig.colorbar(ab, shrink=0.9)
ax.set_xlabel(r'$\theta_2$', fontsize=18)
ax.set_ylabel(r'$\theta_1$', fontsize=18, rotation='horizontal')
fig.savefig("./gauss_contours.pdf", dpi=150)
