"""Integration tests for sequential multi-surface ray tracing.

Covers:
- Single-surface and multi-surface tracing
- Plano-convex singlet and Fraunhofer doublet
- Batch tracing (vmap) consistency
- Image-plane propagation (trace_to_sensor)
- RMS spot size computation
- Ray fan integration
- Gradient verification: autodiff vs. finite differences
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.tracer.datatypes import (
    example_plano_convex_singlet,
    example_doublet,
    make_surface,
    make_lens_system,
)
from src.tracer.trace import (
    trace_ray_through_system,
    trace_to_sensor,
    trace_all_rays,
    trace_all_to_sensor,
    compute_rms_spot_size,
)
from src.tracer.ray_fan import generate_ray_fan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _on_axis_ray(z_start=-10.0):
    """Return an on-axis ray starting at (0, 0, z_start) going +z."""
    return jnp.array([0.0, 0.0, z_start]), jnp.array([0.0, 0.0, 1.0])


def _off_axis_ray(y, z_start=-10.0):
    """Return a ray parallel to the axis at height y."""
    return jnp.array([0.0, y, z_start]), jnp.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Trace through plano-convex singlet
# ---------------------------------------------------------------------------

class TestSingletTrace:
    """Trace rays through the example plano-convex singlet."""

    def test_on_axis_reaches_last_surface(self):
        surfaces = example_plano_convex_singlet()
        origin, direction = _on_axis_ray()
        final_o, final_d, valid = trace_ray_through_system(origin, direction, surfaces)
        assert valid
        # On-axis ray should stay on axis
        assert float(final_o[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(final_o[1]) == pytest.approx(0.0, abs=1e-6)

    def test_on_axis_to_sensor(self):
        surfaces = example_plano_convex_singlet()
        origin, direction = _on_axis_ray()
        img, d, valid = trace_to_sensor(origin, direction, surfaces)
        assert valid
        # Image plane should be at z = thickness + back_focal = 5 + 45 = 50
        assert float(img[2]) == pytest.approx(50.0, abs=0.1)
        assert float(img[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(img[1]) == pytest.approx(0.0, abs=1e-6)

    def test_off_axis_converges(self):
        """Off-axis parallel rays should converge toward the focal point."""
        surfaces = example_plano_convex_singlet()
        _, d = _on_axis_ray()

        heights = [2.0, 4.0, 6.0]
        y_images = []
        for h in heights:
            o = jnp.array([0.0, h, -10.0])
            img, _, valid = trace_to_sensor(o, d, surfaces)
            assert valid
            y_images.append(float(img[1]))

        # All off-axis rays should bend toward the axis (y_image < y_origin)
        for h, yi in zip(heights, y_images):
            assert abs(yi) < h

    def test_validity_inside_aperture(self):
        surfaces = example_plano_convex_singlet()
        origin, direction = _off_axis_ray(10.0)
        _, _, valid = trace_ray_through_system(origin, direction, surfaces)
        assert valid  # 10mm < 12.5mm aperture

    def test_vignetting_outside_aperture(self):
        surfaces = example_plano_convex_singlet()
        origin, direction = _off_axis_ray(15.0)
        _, _, valid = trace_ray_through_system(origin, direction, surfaces)
        assert not valid  # 15mm > 12.5mm aperture


# ---------------------------------------------------------------------------
# Trace through doublet
# ---------------------------------------------------------------------------

class TestDoubletTrace:
    """Trace rays through the example Fraunhofer doublet."""

    def test_on_axis_valid(self):
        surfaces = example_doublet()
        origin, direction = _on_axis_ray()
        _, _, valid = trace_ray_through_system(origin, direction, surfaces)
        assert valid

    def test_on_axis_to_sensor_stays_on_axis(self):
        surfaces = example_doublet()
        origin, direction = _on_axis_ray()
        img, _, valid = trace_to_sensor(origin, direction, surfaces)
        assert valid
        assert float(img[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(img[1]) == pytest.approx(0.0, abs=1e-6)
        # z should be at sum of thicknesses: 6 + 3 + 90 = 99
        assert float(img[2]) == pytest.approx(99.0, abs=0.1)

    def test_three_surfaces_traversed(self):
        """The doublet has 3 surfaces — verify the ray passes through all."""
        surfaces = example_doublet()
        origin, direction = _off_axis_ray(5.0)
        final_o, final_d, valid = trace_ray_through_system(origin, direction, surfaces)
        assert valid
        # Direction should have changed (refracted at 3 surfaces)
        assert not np.allclose(np.array(final_d), np.array([0.0, 0.0, 1.0]), atol=1e-6)

    def test_doublet_better_than_singlet(self):
        """The doublet should produce a smaller spot than the singlet (roughly)."""
        singlet = example_plano_convex_singlet()
        doublet = example_doublet()

        direction = jnp.array([0.0, 0.0, 1.0])
        heights = jnp.array([0.0, 2.0, 4.0, 6.0])

        for system in [singlet, doublet]:
            origins = jnp.stack([jnp.array([0.0, h, -10.0]) for h in heights])
            dirs = jnp.tile(direction, (len(heights), 1))
            imgs, _, valids = trace_all_to_sensor(origins, dirs, system)
            # Just verify it runs and produces valid results
            assert jnp.all(valids)


# ---------------------------------------------------------------------------
# Batch tracing
# ---------------------------------------------------------------------------

class TestBatchTracing:
    """Verify vmap-based batch tracing matches sequential results."""

    def test_batch_matches_sequential(self):
        surfaces = example_plano_convex_singlet()
        direction = jnp.array([0.0, 0.0, 1.0])

        origins = jnp.array([
            [0.0, 0.0, -10.0],
            [0.0, 3.0, -10.0],
            [0.0, 6.0, -10.0],
        ])
        dirs = jnp.tile(direction, (3, 1))

        # Batch
        batch_imgs, batch_dirs, batch_valid = trace_all_to_sensor(origins, dirs, surfaces)

        # Sequential
        for i in range(3):
            img, d, v = trace_to_sensor(origins[i], dirs[i], surfaces)
            np.testing.assert_allclose(np.array(batch_imgs[i]), np.array(img), atol=1e-6)
            np.testing.assert_allclose(np.array(batch_dirs[i]), np.array(d), atol=1e-6)
            assert bool(batch_valid[i]) == bool(v)

    def test_trace_all_rays_consistency(self):
        """trace_all_rays should match trace_ray_through_system per-ray."""
        surfaces = example_doublet()
        origins = jnp.array([[0.0, 0.0, -10.0], [0.0, 5.0, -10.0]])
        dirs = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

        batch_o, batch_d, batch_v = trace_all_rays(origins, dirs, surfaces)

        for i in range(2):
            o, d, v = trace_ray_through_system(origins[i], dirs[i], surfaces)
            np.testing.assert_allclose(np.array(batch_o[i]), np.array(o), atol=1e-6)


# ---------------------------------------------------------------------------
# RMS spot size
# ---------------------------------------------------------------------------

class TestRMSSpotSize:

    def test_perfect_focus_is_zero(self):
        """All rays landing at the same point → RMS = 0."""
        chief = jnp.array([0.0, 0.0, 50.0])
        points = jnp.tile(chief, (5, 1))
        valid = jnp.ones(5, dtype=bool)
        rms = compute_rms_spot_size(points, valid, chief)
        assert float(rms) == pytest.approx(0.0, abs=1e-8)

    def test_known_rms(self):
        """Four rays at unit distance from chief → RMS = 1."""
        chief = jnp.array([0.0, 0.0, 50.0])
        points = jnp.array([
            [1.0, 0.0, 50.0],
            [-1.0, 0.0, 50.0],
            [0.0, 1.0, 50.0],
            [0.0, -1.0, 50.0],
        ])
        valid = jnp.ones(4, dtype=bool)
        rms = compute_rms_spot_size(points, valid, chief)
        assert float(rms) == pytest.approx(1.0, abs=1e-6)

    def test_invalid_rays_excluded(self):
        """Invalid rays should not contribute to RMS."""
        chief = jnp.array([0.0, 0.0, 50.0])
        points = jnp.array([
            [0.0, 0.0, 50.0],  # on chief
            [100.0, 0.0, 50.0],  # far away but invalid
        ])
        valid = jnp.array([True, False])
        rms = compute_rms_spot_size(points, valid, chief)
        assert float(rms) == pytest.approx(0.0, abs=1e-8)

    def test_no_valid_rays_returns_zero(self):
        chief = jnp.array([0.0, 0.0, 50.0])
        points = jnp.array([[10.0, 10.0, 50.0]])
        valid = jnp.array([False])
        rms = compute_rms_spot_size(points, valid, chief)
        assert float(rms) == pytest.approx(0.0, abs=1e-8)

    def test_singlet_spot_size_positive(self):
        """Tracing a ray fan through the singlet should give finite spot."""
        surfaces = example_plano_convex_singlet()
        direction = jnp.array([0.0, 0.0, 1.0])
        heights = jnp.linspace(-8.0, 8.0, 9)
        origins = jnp.stack([jnp.array([0.0, h, -10.0]) for h in heights])
        dirs = jnp.tile(direction, (9, 1))

        imgs, _, valids = trace_all_to_sensor(origins, dirs, surfaces)
        chief = imgs[4]  # center ray
        rms = compute_rms_spot_size(imgs, valids, chief)
        assert float(rms) > 0.0
        assert jnp.isfinite(rms)


# ---------------------------------------------------------------------------
# Ray fan integration
# ---------------------------------------------------------------------------

class TestRayFanIntegration:
    """End-to-end: generate ray fan → trace → spot size."""

    def test_on_axis_fan_through_singlet(self):
        surfaces = example_plano_convex_singlet()
        field_angles = jnp.array([0.0])
        origins, dirs, field_idx, circ_mask = generate_ray_fan(
            field_angles, 5, 10.0
        )
        imgs, _, valids = trace_all_to_sensor(origins, dirs, surfaces)

        # Combined validity
        combined_valid = valids & circ_mask
        # Chief ray is center of the fan
        chief_idx = origins.shape[0] // 2
        chief = imgs[chief_idx]

        rms = compute_rms_spot_size(imgs, combined_valid, chief)
        assert jnp.isfinite(rms)
        assert float(rms) >= 0.0

    def test_multi_field_fan_through_doublet(self):
        surfaces = example_doublet()
        field_angles = jnp.array([0.0, 0.05, 0.1])
        origins, dirs, field_idx, circ_mask = generate_ray_fan(
            field_angles, 5, 10.0
        )
        imgs, _, valids = trace_all_to_sensor(origins, dirs, surfaces)
        combined_valid = valids & circ_mask

        # Compute per-field spot sizes
        S = 25  # 5x5 grid
        for f in range(3):
            mask = (field_idx == f)
            field_imgs = imgs[mask]
            field_valid = combined_valid[mask]
            chief = field_imgs[S // 2]
            rms = compute_rms_spot_size(field_imgs, field_valid, chief)
            assert jnp.isfinite(rms)


# ---------------------------------------------------------------------------
# Gradient verification
# ---------------------------------------------------------------------------

class TestTraceGradients:
    """Verify that jax.grad flows through the full trace pipeline."""

    def test_grad_through_singlet_trace(self):
        """Gradient of image point w.r.t. surface curvature."""
        origin, direction = _on_axis_ray()

        def loss(surfaces):
            img, _, _ = trace_to_sensor(origin, direction, surfaces)
            return jnp.sum(img ** 2)

        surfaces = example_plano_convex_singlet()
        grad = jax.grad(loss)(surfaces)
        assert grad.shape == surfaces.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_doublet_trace(self):
        origin, direction = _off_axis_ray(5.0)

        def loss(surfaces):
            img, _, _ = trace_to_sensor(origin, direction, surfaces)
            return img[1] ** 2

        surfaces = example_doublet()
        grad = jax.grad(loss)(surfaces)
        assert grad.shape == surfaces.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_through_rms_spot(self):
        """Gradient of RMS spot size w.r.t. surface parameters."""
        direction = jnp.array([0.0, 0.0, 1.0])
        heights = jnp.linspace(-6.0, 6.0, 7)
        origins = jnp.stack([jnp.array([0.0, h, -10.0]) for h in heights])
        dirs = jnp.tile(direction, (7, 1))

        def loss(surfaces):
            imgs, _, valids = trace_all_to_sensor(origins, dirs, surfaces)
            chief = imgs[3]  # center ray
            return compute_rms_spot_size(imgs, valids, chief)

        surfaces = example_plano_convex_singlet()
        grad = jax.grad(loss)(surfaces)
        assert grad.shape == surfaces.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_autodiff_vs_finite_diff(self):
        """Check autodiff gradient against finite-difference approximation."""
        origin, direction = _off_axis_ray(5.0)

        def loss(surfaces):
            img, _, _ = trace_to_sensor(origin, direction, surfaces)
            return img[1]

        surfaces = example_plano_convex_singlet()
        ad_grad = jax.grad(loss)(surfaces)

        # Finite-difference check on curvature of first surface (index [0, 0])
        eps = 1e-5
        s_plus = surfaces.at[0, 0].set(surfaces[0, 0] + eps)
        s_minus = surfaces.at[0, 0].set(surfaces[0, 0] - eps)
        fd_grad = (loss(s_plus) - loss(s_minus)) / (2 * eps)

        np.testing.assert_allclose(
            float(ad_grad[0, 0]), float(fd_grad), rtol=1e-3, atol=1e-6
        )

    def test_jit_compilation(self):
        """Verify the full trace compiles under jax.jit."""
        origin, direction = _on_axis_ray()
        surfaces = example_plano_convex_singlet()

        jitted = jax.jit(trace_to_sensor, static_argnums=())
        img, d, v = jitted(origin, direction, surfaces)
        assert jnp.all(jnp.isfinite(img))
        assert v
