"""Core data structures for the differentiable ray tracer.

All representations are JAX-compatible: lens systems are stored as stacked
arrays so that jax.lax.scan and jax.grad work through the entire trace.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Ray representation
# ---------------------------------------------------------------------------

class Ray(NamedTuple):
    """A ray defined by an origin point and a unit direction vector.

    Both fields are shape (3,) JAX arrays: [x, y, z].
    """
    origin: jnp.ndarray
    direction: jnp.ndarray


# ---------------------------------------------------------------------------
# Surface parameter layout (stacked array columns)
# ---------------------------------------------------------------------------
# Each surface is stored as a row in a (num_surfaces, NUM_SURFACE_PARAMS) array.
# The column indices are defined here so every module agrees on the layout.

CURVATURE = 0       # 1/radius  (0 → flat surface)
THICKNESS = 1       # axial distance to the *next* surface
N_BEFORE = 2        # refractive index on the incoming side
N_AFTER = 3         # refractive index on the outgoing side
APERTURE_RADIUS = 4 # clear aperture half-diameter
CONIC_CONSTANT = 5  # 0 → sphere, -1 → paraboloid (stretch goal)

NUM_SURFACE_PARAMS = 6


# ---------------------------------------------------------------------------
# Helpers: build individual surfaces and full systems
# ---------------------------------------------------------------------------

def make_surface(
    curvature: float = 0.0,
    thickness: float = 0.0,
    n_before: float = 1.0,
    n_after: float = 1.0,
    aperture_radius: float = 25.0,
    conic_constant: float = 0.0,
) -> jnp.ndarray:
    """Return a single surface as a 1-D array of length NUM_SURFACE_PARAMS."""
    return jnp.array([
        curvature,
        thickness,
        n_before,
        n_after,
        aperture_radius,
        conic_constant,
    ])


def make_lens_system(*surfaces: jnp.ndarray) -> jnp.ndarray:
    """Stack individual surface arrays into a (num_surfaces, NUM_SURFACE_PARAMS) array."""
    return jnp.stack(surfaces, axis=0)


# ---------------------------------------------------------------------------
# Target specification (passed alongside params during optimization)
# ---------------------------------------------------------------------------

class TargetSpec(NamedTuple):
    """Describes the desired optical performance of the lens system."""
    focal_length: float          # target effective focal length (mm)
    field_angles: jnp.ndarray    # half-field angles to evaluate (radians)
    wavelength: float            # design wavelength (nm)
    entrance_pupil_radius: float # entrance pupil half-diameter (mm)
    # weights for merit function terms
    w_spot: float = 1.0
    w_focal: float = 1.0
    w_distortion: float = 0.1
    w_constraints: float = 10.0


# ---------------------------------------------------------------------------
# Convenience: extract readable parameters from a surfaces array
# ---------------------------------------------------------------------------

def get_curvatures(surfaces: jnp.ndarray) -> jnp.ndarray:
    return surfaces[:, CURVATURE]

def get_thicknesses(surfaces: jnp.ndarray) -> jnp.ndarray:
    return surfaces[:, THICKNESS]

def get_n_before(surfaces: jnp.ndarray) -> jnp.ndarray:
    return surfaces[:, N_BEFORE]

def get_n_after(surfaces: jnp.ndarray) -> jnp.ndarray:
    return surfaces[:, N_AFTER]

def get_aperture_radii(surfaces: jnp.ndarray) -> jnp.ndarray:
    return surfaces[:, APERTURE_RADIUS]


# ---------------------------------------------------------------------------
# Example lens: simple plano-convex singlet (useful for early testing)
# ---------------------------------------------------------------------------

def example_plano_convex_singlet(
    curvature: float = 0.02,   # 1/R, R=50mm
    thickness: float = 5.0,    # center thickness mm
    n_glass: float = 1.5168,   # BK7 at 587.6 nm
    aperture: float = 12.5,    # 25mm diameter
    back_focal_distance: float = 45.0,
) -> jnp.ndarray:
    """A minimal 2-surface lens system for smoke-testing the tracer.

    Surface 0: curved front face (air → glass)
    Surface 1: flat rear face  (glass → air), with thickness = back focal distance to image plane
    """
    s0 = make_surface(
        curvature=curvature,
        thickness=thickness,
        n_before=1.0,
        n_after=n_glass,
        aperture_radius=aperture,
    )
    s1 = make_surface(
        curvature=0.0,
        thickness=back_focal_distance,
        n_before=n_glass,
        n_after=1.0,
        aperture_radius=aperture,
    )
    return make_lens_system(s0, s1)


def example_doublet(
    c1: float = 0.0147,    # front crown surface
    c2: float = -0.0125,   # cemented interface
    c3: float = -0.0040,   # rear flint surface
    t1: float = 6.0,       # crown element thickness
    t2: float = 3.0,       # flint element thickness
    n_crown: float = 1.5168,  # BK7
    n_flint: float = 1.6727,  # SF2
    aperture: float = 12.5,
    back_focal_distance: float = 90.0,
) -> jnp.ndarray:
    """A Fraunhofer-style cemented doublet for validation.

    Surface 0: air → crown  (curvature c1)
    Surface 1: crown → flint (curvature c2, cemented)
    Surface 2: flint → air  (curvature c3)
    """
    s0 = make_surface(curvature=c1, thickness=t1, n_before=1.0, n_after=n_crown, aperture_radius=aperture)
    s1 = make_surface(curvature=c2, thickness=t2, n_before=n_crown, n_after=n_flint, aperture_radius=aperture)
    s2 = make_surface(curvature=c3, thickness=back_focal_distance, n_before=n_flint, n_after=1.0, aperture_radius=aperture)
    return make_lens_system(s0, s1, s2)
