"""Ray fan generation for multi-field-angle tracing.

Generates grids of rays across the entrance pupil for each field angle,
suitable for feeding into ``trace_all_to_sensor``.
"""

import jax.numpy as jnp


def generate_ray_fan(field_angles, num_pupil_samples, entrance_pupil_radius):
    """Generate a fan of rays across the entrance pupil for each field angle.

    Rays originate from a plane at ``z = 0`` and are directed so that the
    chief ray for each field angle passes through the center of the entrance
    pupil at the given angle to the optical axis.

    Parameters
    ----------
    field_angles : (F,) array
        Half-field angles in radians (e.g. [0, 0.05, 0.1, 0.25]).
    num_pupil_samples : int
        Number of samples along one side of the square pupil grid.
        Total rays per field angle = num_pupil_samples ** 2.
    entrance_pupil_radius : float
        Half-diameter of the entrance pupil (mm).

    Returns
    -------
    origins : (F * S, 3) array
        Ray starting positions on the entrance pupil plane (z = 0).
        S = num_pupil_samples ** 2.
    directions : (F * S, 3) array
        Unit direction vectors for each ray.
    field_indices : (F * S,) int array
        Index into *field_angles* for each ray, useful for grouping
        results by field angle after tracing.
    """
    # Square grid on the entrance pupil, excluding the very edge
    n = num_pupil_samples
    # Linspace from -r to +r with a small margin to avoid exact edge rays
    margin = entrance_pupil_radius / n
    coords = jnp.linspace(
        -entrance_pupil_radius + margin,
        entrance_pupil_radius - margin,
        n,
    )
    gx, gy = jnp.meshgrid(coords, coords)
    pupil_x = gx.ravel()  # (S,)
    pupil_y = gy.ravel()  # (S,)
    S = n * n

    # Filter to circular aperture: keep only points inside the pupil radius
    # (For a square grid this masks out corners.)
    r = jnp.sqrt(pupil_x ** 2 + pupil_y ** 2)
    inside = r <= entrance_pupil_radius

    # For each field angle, the chief ray direction is
    #   d = [0, sin(theta), cos(theta)]
    # All pupil rays share the same direction (collimated beam at field angle)
    # and differ only in their origin on the entrance pupil.
    F = field_angles.shape[0]

    # Build per-field directions: shape (F, 3)
    sin_a = jnp.sin(field_angles)
    cos_a = jnp.cos(field_angles)
    zeros = jnp.zeros_like(field_angles)
    field_dirs = jnp.stack([zeros, sin_a, cos_a], axis=-1)  # (F, 3)

    # Tile pupil positions for all field angles: (F*S, 3)
    # Origins: (pupil_x, pupil_y, 0) repeated F times
    pupil_z = jnp.zeros_like(pupil_x)
    pupil_origins = jnp.stack([pupil_x, pupil_y, pupil_z], axis=-1)  # (S, 3)
    all_origins = jnp.tile(pupil_origins, (F, 1))  # (F*S, 3)

    # Directions: each field direction repeated S times
    all_directions = jnp.repeat(field_dirs, S, axis=0)  # (F*S, 3)

    # Field indices for grouping
    field_indices = jnp.repeat(jnp.arange(F), S)  # (F*S,)

    # Validity mask for circular aperture (tiled for all fields)
    circular_mask = jnp.tile(inside, F)  # (F*S,)

    return all_origins, all_directions, field_indices, circular_mask
