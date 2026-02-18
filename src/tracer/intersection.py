"""Ray-surface intersection and surface normals for spherical surfaces.

All functions are JAX-traceable: no Python-level control flow, only
``jnp.where`` / ``jax.lax.cond`` so that ``jax.grad`` and ``jax.jit``
work through the entire computation.

Coordinate convention
---------------------
* Optical axis is the z-axis; rays propagate in the +z direction.
* A surface vertex sits at ``z = z_offset`` on the optical axis.
* For a spherical surface with curvature *c = 1/R*, the center of
  curvature is at ``(0, 0, z_offset + R)``.
* Curvature sign: ``c > 0`` → center of curvature is to the *right*
  of the vertex (convex toward incoming light); ``c < 0`` → concave.
"""

import jax.numpy as jnp

# Small epsilon to guard against division by zero in JAX traces.
_EPS = 1e-12


def intersect_sphere(
    ray_origin: jnp.ndarray,
    ray_dir: jnp.ndarray,
    curvature: float,
    z_offset: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find where a ray hits a spherical (or planar) surface.

    Parameters
    ----------
    ray_origin : (3,) array – ray starting point [x, y, z].
    ray_dir    : (3,) array – unit direction vector of the ray.
    curvature  : scalar – 1/R of the surface (0 for a flat surface).
    z_offset   : scalar – z-position of the surface vertex.

    Returns
    -------
    t         : scalar – parametric distance along the ray to the hit point.
    hit_point : (3,) array – intersection coordinates.

    Notes
    -----
    For a planar surface (curvature ≈ 0) the intersection is simply the
    z = z_offset plane.  For a sphere the standard quadratic is solved and
    the root closest to the vertex (smallest positive *t*) is selected.
    Both paths are evaluated and blended with ``jnp.where`` so that the
    function remains differentiable everywhere.
    """

    # --- planar intersection (always computed) ---
    # t_plane such that origin_z + t * dir_z = z_offset
    t_plane = (z_offset - ray_origin[2]) / (ray_dir[2] + _EPS)

    # --- spherical intersection (always computed) ---
    # Sphere center: (0, 0, z_offset + R) where R = 1/c.
    # To avoid 1/c blowing up when c→0, multiply through by c later.
    #
    # Sphere equation: |P - C|^2 = R^2
    # With P = O + tD, let A = O - C:
    #   t^2 |D|^2 + 2t (A·D) + |A|^2 - R^2 = 0
    #
    # R = 1/c  →  C = (0, 0, z_offset + 1/c)
    # Use safe_inv to keep the computation stable for small c.
    safe_c = jnp.where(jnp.abs(curvature) < _EPS, _EPS, curvature)
    R = 1.0 / safe_c
    center = jnp.array([0.0, 0.0, z_offset + R])

    A = ray_origin - center
    a_coeff = jnp.dot(ray_dir, ray_dir)  # 1.0 if unit dir
    b_coeff = 2.0 * jnp.dot(A, ray_dir)
    c_coeff = jnp.dot(A, A) - R * R

    discriminant = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
    safe_disc = jnp.maximum(discriminant, 0.0)
    sqrt_disc = jnp.sqrt(safe_disc)

    # Two candidate roots
    t1 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff + _EPS)
    t2 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff + _EPS)

    # Pick the root whose hit point is closest to the vertex in z.
    # For a well-behaved sequential trace the first positive root that
    # lands near z_offset is the one we want.  We pick the smaller
    # positive t; if both are negative we fall back to t2.
    t_sphere = jnp.where((t1 > 0), t1, t2)

    # --- blend planar / spherical based on |curvature| ---
    is_flat = jnp.abs(curvature) < _EPS
    t = jnp.where(is_flat, t_plane, t_sphere)

    hit_point = ray_origin + t * ray_dir
    return t, hit_point


def surface_normal(
    hit_point: jnp.ndarray,
    curvature: float,
    z_offset: float,
) -> jnp.ndarray:
    """Outward-facing unit normal at *hit_point* on a spherical surface.

    The returned normal is oriented to oppose the incoming ray direction
    (i.e. it points back toward the medium the ray came from).  For a
    planar surface the normal is simply ``[0, 0, -1]`` (opposing +z
    propagation).

    Parameters
    ----------
    hit_point : (3,) array
    curvature : scalar – 1/R
    z_offset  : scalar – vertex z-position

    Returns
    -------
    normal : (3,) unit vector
    """
    # Planar normal (always computed)
    plane_normal = jnp.array([0.0, 0.0, -1.0])

    # Spherical normal: n = (hit_point - center) / |...|
    safe_c = jnp.where(jnp.abs(curvature) < _EPS, _EPS, curvature)
    R = 1.0 / safe_c
    center = jnp.array([0.0, 0.0, z_offset + R])

    diff = hit_point - center
    sphere_normal = diff / (jnp.linalg.norm(diff) + _EPS)

    # For sequential tracing (+z propagation) the normal must oppose the
    # incoming ray.  For c > 0 (convex) the geometric normal already
    # points toward -z near the axis.  For c < 0 (concave) it points
    # toward +z, so we flip it.  A robust rule: ensure normal_z < 0.
    sphere_normal = jnp.where(sphere_normal[2] > 0, -sphere_normal, sphere_normal)

    is_flat = jnp.abs(curvature) < _EPS
    normal = jnp.where(is_flat, plane_normal, sphere_normal)
    return normal
