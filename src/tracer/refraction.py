"""Vector form of Snell's law and ray propagation.

All functions are JAX-compatible (no Python-level control flow) so they
work under jax.jit, jax.vmap, and jax.grad.
"""

import jax.numpy as jnp


def snell_refraction(
    incident_dir: jnp.ndarray,
    normal: jnp.ndarray,
    n1: float,
    n2: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the refracted ray direction using the vector form of Snell's law.

    Parameters
    ----------
    incident_dir : (3,) array
        Unit direction vector of the incoming ray.
    normal : (3,) array
        Unit surface normal pointing *into* the medium the ray is entering
        (i.e. in the direction of propagation, from n1 side to n2 side).
    n1 : float
        Refractive index of the medium the ray is leaving.
    n2 : float
        Refractive index of the medium the ray is entering.

    Returns
    -------
    refracted_dir : (3,) array
        Unit direction vector of the refracted ray.  When total internal
        reflection (TIR) occurs the returned direction is still finite
        (a smoothly degraded value) so that gradients remain well-defined.
    valid : scalar bool array
        True when refraction is physically valid (no TIR).  Downstream code
        should accumulate this into an overall validity mask.
    """
    eta = n1 / n2  # ratio of refractive indices

    cos_i = -jnp.dot(incident_dir, normal)
    # Ensure cos_i is positive (normal should face the incoming ray).
    # If it's negative the normal is flipped relative to our convention;
    # we handle this with an abs to keep the function branchless.
    cos_i = jnp.abs(cos_i)

    sin2_t = eta ** 2 * (1.0 - cos_i ** 2)

    # TIR occurs when sin2_t > 1.  We clamp so sqrt stays real and
    # gradients stay finite, then flag validity separately.
    valid = sin2_t <= 1.0
    sin2_t_safe = jnp.clip(sin2_t, 0.0, 1.0)

    cos_t = jnp.sqrt(1.0 - sin2_t_safe)

    refracted = eta * incident_dir + (eta * cos_i - cos_t) * normal

    # Normalize to guard against accumulated floating-point drift.
    refracted = refracted / (jnp.linalg.norm(refracted) + 1e-12)

    return refracted, valid


def propagate(
    origin: jnp.ndarray,
    direction: jnp.ndarray,
    distance: float,
) -> jnp.ndarray:
    """Advance a ray along its direction by *distance*.

    Parameters
    ----------
    origin : (3,) array
        Current position of the ray.
    direction : (3,) array
        Unit direction vector.
    distance : float or scalar array
        Distance to travel along the direction.

    Returns
    -------
    new_origin : (3,) array
        ``origin + distance * direction``.
    """
    return origin + distance * direction
