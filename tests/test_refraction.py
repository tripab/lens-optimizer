"""Unit tests for Snell's law (vector form) and ray propagation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.tracer.refraction import propagate, snell_refraction


# ---- helpers ----------------------------------------------------------------

def _angle_between(v1, v2):
    """Angle in radians between two unit vectors."""
    cos = jnp.clip(jnp.dot(v1, v2), -1.0, 1.0)
    return jnp.arccos(cos)


def _normalize(v):
    return v / jnp.linalg.norm(v)


# ---- snell_refraction tests -------------------------------------------------


class TestSnellNormalIncidence:
    """Ray hitting a surface head-on should pass straight through."""

    def test_normal_incidence_air_to_glass(self):
        incident = jnp.array([0.0, 0.0, 1.0])
        normal = jnp.array([0.0, 0.0, -1.0])  # pointing back toward incoming ray
        refracted, valid = snell_refraction(incident, normal, n1=1.0, n2=1.5168)

        assert valid
        # Direction should be unchanged (still along z)
        np.testing.assert_allclose(float(refracted[0]), 0.0, atol=1e-7)
        np.testing.assert_allclose(float(refracted[1]), 0.0, atol=1e-7)
        np.testing.assert_allclose(float(refracted[2]), 1.0, atol=1e-7)

    def test_normal_incidence_glass_to_air(self):
        incident = jnp.array([0.0, 0.0, 1.0])
        normal = jnp.array([0.0, 0.0, -1.0])
        refracted, valid = snell_refraction(incident, normal, n1=1.5168, n2=1.0)

        assert valid
        np.testing.assert_allclose(float(refracted[0]), 0.0, atol=1e-7)
        np.testing.assert_allclose(float(refracted[1]), 0.0, atol=1e-7)
        np.testing.assert_allclose(float(refracted[2]), 1.0, atol=1e-7)


class TestSnellOblique:
    """Verify n1*sin(theta1) = n2*sin(theta2) for oblique incidence."""

    @pytest.mark.parametrize("n1,n2,angle_deg", [
        (1.0, 1.5168, 45.0),   # air → BK7 at 45°
        (1.0, 1.5168, 30.0),   # air → BK7 at 30°
        (1.5168, 1.0, 20.0),   # BK7 → air at 20°
        (1.0, 1.8, 60.0),      # air → high-index glass at 60°
    ])
    def test_snells_law_holds(self, n1, n2, angle_deg):
        angle_rad = jnp.radians(angle_deg)
        # Incident ray in the x-z plane, tilted by angle_rad from the z-axis
        incident = _normalize(jnp.array([jnp.sin(angle_rad), 0.0, jnp.cos(angle_rad)]))
        normal = jnp.array([0.0, 0.0, -1.0])

        refracted, valid = snell_refraction(incident, normal, n1, n2)
        assert valid

        # Measure angles w.r.t. the surface normal (flipped to face outward along +z)
        outward_normal = jnp.array([0.0, 0.0, 1.0])
        theta_i = _angle_between(incident, outward_normal)
        # Refracted ray goes in +z direction; angle w.r.t. +z
        theta_r = _angle_between(refracted, outward_normal)

        lhs = n1 * jnp.sin(theta_i)
        rhs = n2 * jnp.sin(theta_r)
        np.testing.assert_allclose(float(lhs), float(rhs), atol=1e-6)

    def test_45_degree_air_to_bk7_refracted_angle(self):
        """Check that the refracted angle matches the analytical value."""
        n1, n2 = 1.0, 1.5168
        angle_i = jnp.radians(45.0)
        incident = _normalize(jnp.array([jnp.sin(angle_i), 0.0, jnp.cos(angle_i)]))
        normal = jnp.array([0.0, 0.0, -1.0])

        refracted, valid = snell_refraction(incident, normal, n1, n2)
        assert valid

        # Analytical: sin(theta_t) = (n1/n2) * sin(theta_i)
        expected_sin_t = (n1 / n2) * jnp.sin(angle_i)
        expected_angle_t = jnp.arcsin(expected_sin_t)

        outward_normal = jnp.array([0.0, 0.0, 1.0])
        actual_angle_t = _angle_between(refracted, outward_normal)

        np.testing.assert_allclose(float(actual_angle_t), float(expected_angle_t), atol=1e-6)


class TestTotalInternalReflection:
    """TIR: glass→air at steep angles should flag invalid."""

    def test_tir_flagged(self):
        # Critical angle for BK7→air: arcsin(1/1.5168) ≈ 41.2°
        # Use 60° — well beyond critical angle
        angle_rad = jnp.radians(60.0)
        incident = _normalize(jnp.array([jnp.sin(angle_rad), 0.0, jnp.cos(angle_rad)]))
        normal = jnp.array([0.0, 0.0, -1.0])

        _, valid = snell_refraction(incident, normal, n1=1.5168, n2=1.0)
        assert not valid

    def test_just_below_critical_angle_is_valid(self):
        # Critical angle ≈ 41.2°; use 40°
        angle_rad = jnp.radians(40.0)
        incident = _normalize(jnp.array([jnp.sin(angle_rad), 0.0, jnp.cos(angle_rad)]))
        normal = jnp.array([0.0, 0.0, -1.0])

        _, valid = snell_refraction(incident, normal, n1=1.5168, n2=1.0)
        assert valid

    def test_tir_returns_finite_direction(self):
        """Even under TIR the returned direction must be finite (for gradients)."""
        angle_rad = jnp.radians(80.0)
        incident = _normalize(jnp.array([jnp.sin(angle_rad), 0.0, jnp.cos(angle_rad)]))
        normal = jnp.array([0.0, 0.0, -1.0])

        refracted, _ = snell_refraction(incident, normal, n1=1.5168, n2=1.0)
        assert jnp.all(jnp.isfinite(refracted))


class TestSnellOutputUnit:
    """Refracted direction should always be a unit vector."""

    @pytest.mark.parametrize("angle_deg", [0.0, 15.0, 30.0, 45.0, 60.0])
    def test_unit_length(self, angle_deg):
        angle_rad = jnp.radians(angle_deg)
        incident = _normalize(jnp.array([jnp.sin(angle_rad), 0.0, jnp.cos(angle_rad)]))
        normal = jnp.array([0.0, 0.0, -1.0])

        refracted, _ = snell_refraction(incident, normal, n1=1.0, n2=1.5)
        np.testing.assert_allclose(float(jnp.linalg.norm(refracted)), 1.0, atol=1e-6)


class TestSnellGradients:
    """jax.grad must produce finite, reasonable gradients through refraction."""

    def test_grad_wrt_n2(self):
        incident = _normalize(jnp.array([0.3, 0.0, 0.95]))

        def loss_fn(n2):
            normal = jnp.array([0.0, 0.0, -1.0])
            refracted, _ = snell_refraction(incident, normal, n1=1.0, n2=n2)
            return jnp.sum(refracted ** 2)

        grad_val = jax.grad(loss_fn)(1.5)
        assert jnp.isfinite(grad_val)

    def test_grad_wrt_incident_direction(self):
        def loss_fn(inc_x):
            incident = _normalize(jnp.array([inc_x, 0.0, 1.0]))
            normal = jnp.array([0.0, 0.0, -1.0])
            refracted, _ = snell_refraction(incident, normal, n1=1.0, n2=1.5)
            return jnp.sum(refracted ** 2)

        grad_val = jax.grad(loss_fn)(0.3)
        assert jnp.isfinite(grad_val)

    def test_grad_finite_difference_agreement(self):
        """Autodiff gradient should agree with finite-difference approximation."""
        eps = 1e-5

        def loss_fn(n2):
            incident = _normalize(jnp.array([0.3, 0.0, 0.95]))
            normal = jnp.array([0.0, 0.0, -1.0])
            refracted, _ = snell_refraction(incident, normal, n1=1.0, n2=n2)
            # Use the x-component which varies meaningfully with n2
            return refracted[0]

        n2_val = 1.5
        auto_grad = float(jax.grad(loss_fn)(n2_val))
        fd_grad = float((loss_fn(n2_val + eps) - loss_fn(n2_val - eps)) / (2 * eps))

        np.testing.assert_allclose(auto_grad, fd_grad, rtol=5e-3)


# ---- propagate tests --------------------------------------------------------


class TestPropagate:

    def test_forward_along_z(self):
        origin = jnp.array([0.0, 0.0, 0.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        result = propagate(origin, direction, 10.0)
        np.testing.assert_allclose(result, jnp.array([0.0, 0.0, 10.0]), atol=1e-7)

    def test_diagonal_propagation(self):
        origin = jnp.array([1.0, 2.0, 3.0])
        direction = _normalize(jnp.array([1.0, 1.0, 1.0]))
        dist = 3.0
        result = propagate(origin, direction, dist)
        expected = origin + dist * direction
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_zero_distance(self):
        origin = jnp.array([5.0, -3.0, 7.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        result = propagate(origin, direction, 0.0)
        np.testing.assert_allclose(result, origin, atol=1e-7)

    def test_negative_distance(self):
        origin = jnp.array([0.0, 0.0, 10.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        result = propagate(origin, direction, -5.0)
        np.testing.assert_allclose(result, jnp.array([0.0, 0.0, 5.0]), atol=1e-7)
