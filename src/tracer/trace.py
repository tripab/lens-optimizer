"""Sequential multi-surface ray tracing and spot size computation.

Uses ``jax.lax.scan`` to trace a ray through a stack of surfaces and
``jax.vmap`` to vectorise over entire ray fans.  All functions are
JAX-differentiable.
"""

import jax
import jax.numpy as jnp

from .datatypes import CURVATURE, THICKNESS, N_BEFORE, N_AFTER, APERTURE_RADIUS
from .intersection import intersect_sphere, surface_normal
from .refraction import snell_refraction, propagate


def _trace_single_surface(carry, surface_params):
    """Body function for ``jax.lax.scan`` over surfaces.

    Parameters
    ----------
    carry : tuple (origin, direction, z_offset, valid)
        Current ray state and accumulated z-position.
    surface_params : (NUM_SURFACE_PARAMS,) array
        One row of the stacked surfaces array.

    Returns
    -------
    new_carry : same structure as *carry*.
    None      : no per-surface output collected.
    """
    origin, direction, z_offset, valid = carry

    c = surface_params[CURVATURE]
    thickness = surface_params[THICKNESS]
    n1 = surface_params[N_BEFORE]
    n2 = surface_params[N_AFTER]
    aperture = surface_params[APERTURE_RADIUS]

    # 1. Find intersection with this surface
    _t, hit = intersect_sphere(origin, direction, c, z_offset)

    # 2. Vignetting check: is the hit point inside the clear aperture?
    r_hit = jnp.sqrt(hit[0] ** 2 + hit[1] ** 2)
    not_vignetted = r_hit <= aperture

    # 3. Surface normal at the hit point
    normal = surface_normal(hit, c, z_offset)

    # 4. Refract
    new_dir, refraction_valid = snell_refraction(direction, normal, n1, n2)

    # 5. Update accumulated z-offset for the next surface
    new_z_offset = z_offset + thickness

    # 6. Propagate validity
    new_valid = valid & not_vignetted & refraction_valid

    return (hit, new_dir, new_z_offset, new_valid), None


def trace_ray_through_system(ray_origin, ray_direction, surfaces):
    """Trace a single ray through all surfaces in the lens system.

    Parameters
    ----------
    ray_origin : (3,) array
        Starting position of the ray.
    ray_direction : (3,) array
        Unit direction vector of the ray.
    surfaces : (num_surfaces, NUM_SURFACE_PARAMS) array
        Stacked surface parameter array.

    Returns
    -------
    final_origin : (3,) array
        Position of the ray after the last surface.
    final_direction : (3,) array
        Direction of the ray after the last refraction.
    valid : scalar bool
        False if the ray was vignetted or hit total internal reflection.
    """
    z_offset = jnp.float32(0.0) if surfaces.dtype == jnp.float32 else 0.0
    init_carry = (ray_origin, ray_direction, jnp.array(0.0), jnp.array(True))

    (final_origin, final_direction, _z, valid), _ = jax.lax.scan(
        _trace_single_surface, init_carry, surfaces
    )

    return final_origin, final_direction, valid


def trace_to_sensor(ray_origin, ray_direction, surfaces):
    """Trace a ray through all surfaces and propagate to the image plane.

    The image plane is located at a distance equal to the last surface's
    thickness beyond the last surface vertex.  This is the standard
    sequential-trace convention where the final thickness represents the
    back focal distance to the sensor.

    Returns
    -------
    image_point : (3,) array
        Ray position on the image plane.
    final_direction : (3,) array
        Ray direction after the last refraction.
    valid : scalar bool
    """
    last_origin, last_dir, valid = trace_ray_through_system(
        ray_origin, ray_direction, surfaces
    )
    # Propagate from the last surface hit to the image plane.
    # The last surface's thickness is the distance to the sensor.
    last_thickness = surfaces[-1, THICKNESS]
    # Distance along the ray to reach the image plane z-coordinate.
    # The image plane is at z = (sum of all thicknesses from surfaces before
    # the last) + last_thickness.  But since last_origin is already on the
    # last surface, we just need to travel until we advance by last_thickness
    # in z.  For a nearly-paraxial ray this is ≈ last_thickness / dir_z.
    t_to_sensor = last_thickness / (jnp.abs(last_dir[2]) + 1e-12)
    image_point = propagate(last_origin, last_dir, t_to_sensor)
    return image_point, last_dir, valid


def trace_all_to_sensor(ray_origins, ray_directions, surfaces):
    """Trace a batch of rays to the image plane using ``jax.vmap``.

    Returns
    -------
    image_points : (N, 3) array
    final_directions : (N, 3) array
    valid_mask : (N,) bool array
    """
    batched = jax.vmap(trace_to_sensor, in_axes=(0, 0, None))
    return batched(ray_origins, ray_directions, surfaces)


def trace_all_rays(ray_origins, ray_directions, surfaces):
    """Trace a batch of rays through the lens system using ``jax.vmap``.

    Parameters
    ----------
    ray_origins : (N, 3) array
        Starting positions for each ray.
    ray_directions : (N, 3) array
        Unit direction vectors for each ray.
    surfaces : (num_surfaces, NUM_SURFACE_PARAMS) array
        Stacked surface parameter array (shared across all rays).

    Returns
    -------
    final_origins : (N, 3) array
        Final positions of each ray.
    final_directions : (N, 3) array
        Final directions of each ray.
    valid_mask : (N,) bool array
        Per-ray validity flags.
    """
    batched = jax.vmap(trace_ray_through_system, in_axes=(0, 0, None))
    return batched(ray_origins, ray_directions, surfaces)


def compute_rms_spot_size(image_points, valid_mask, chief_ray_point):
    """RMS spot radius of ray intersections relative to the chief ray.

    Parameters
    ----------
    image_points : (N, 3) array
        Final ray positions on the image plane.
    valid_mask : (N,) bool array
        Which rays are valid (not vignetted / TIR).
    chief_ray_point : (3,) array
        Position of the chief ray on the image plane (centroid reference).

    Returns
    -------
    rms : scalar
        RMS distance of valid rays from the chief ray in the x-y plane.
        Returns 0.0 if no rays are valid.
    """
    dx = image_points[:, 0] - chief_ray_point[0]
    dy = image_points[:, 1] - chief_ray_point[1]
    r_squared = dx ** 2 + dy ** 2

    # Mask out invalid rays
    weights = valid_mask.astype(image_points.dtype)
    n_valid = jnp.sum(weights)

    # Avoid division by zero when no rays are valid
    mean_r2 = jnp.sum(r_squared * weights) / jnp.maximum(n_valid, 1.0)
    return jnp.sqrt(mean_r2)
