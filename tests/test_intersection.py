"""Unit tests for src.tracer.intersection — ray-sphere intersection and normals."""

import jax
import jax.numpy as jnp
import pytest

from src.tracer.intersection import intersect_sphere, surface_normal


# ---- helpers ---------------------------------------------------------------

def _normalize(v):
    return v / jnp.linalg.norm(v)


# ---- intersect_sphere tests ------------------------------------------------

class TestIntersectPlanar:
    """Curvature = 0 → flat surface at z = z_offset."""

    def test_on_axis_ray(self):
        origin = jnp.array([0.0, 0.0, 0.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        t, hit = intersect_sphere(origin, direction, curvature=0.0, z_offset=10.0)
        assert float(t) == pytest.approx(10.0, abs=1e-5)
        assert float(hit[2]) == pytest.approx(10.0, abs=1e-5)

    def test_off_axis_ray(self):
        origin = jnp.array([3.0, 4.0, 5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        t, hit = intersect_sphere(origin, direction, curvature=0.0, z_offset=15.0)
        assert float(t) == pytest.approx(10.0, abs=1e-5)
        # x, y should be unchanged for a z-only direction
        assert float(hit[0]) == pytest.approx(3.0, abs=1e-5)
        assert float(hit[1]) == pytest.approx(4.0, abs=1e-5)

    def test_tilted_ray(self):
        origin = jnp.array([0.0, 0.0, 0.0])
        direction = _normalize(jnp.array([1.0, 0.0, 1.0]))
        t, hit = intersect_sphere(origin, direction, curvature=0.0, z_offset=10.0)
        assert float(hit[2]) == pytest.approx(10.0, abs=1e-5)
        assert float(hit[0]) == pytest.approx(10.0, abs=1e-5)  # 45° → x == z


class TestIntersectConvex:
    """Curvature > 0 → center of curvature to the right of the vertex."""

    def test_on_axis_ray_hits_vertex(self):
        """An on-axis ray should hit at the vertex (x=0, y=0, z≈z_offset)."""
        origin = jnp.array([0.0, 0.0, -20.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        t, hit = intersect_sphere(origin, direction, curvature=0.02, z_offset=0.0)
        assert float(t) > 0
        # On axis the hit z should be very close to the vertex
        assert float(hit[2]) == pytest.approx(0.0, abs=1e-4)
        assert float(hit[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(hit[1]) == pytest.approx(0.0, abs=1e-6)

    def test_hit_lies_on_sphere(self):
        """The hit point must satisfy |hit - center| = R."""
        origin = jnp.array([0.0, 8.0, -15.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        c = 0.02  # R = 50
        t, hit = intersect_sphere(origin, direction, curvature=c, z_offset=0.0)
        center = jnp.array([0.0, 0.0, 1.0 / c])
        dist = float(jnp.linalg.norm(hit - center))
        assert dist == pytest.approx(50.0, abs=1e-3)

    def test_off_axis_sag(self):
        """For height h on a sphere of radius R, sag ≈ h²/(2R) for small h."""
        h = 5.0
        R = 50.0
        c = 1.0 / R
        origin = jnp.array([0.0, h, -10.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, hit = intersect_sphere(origin, direction, curvature=c, z_offset=0.0)
        expected_sag = h ** 2 / (2 * R)  # 0.25
        assert float(hit[2]) == pytest.approx(expected_sag, abs=0.01)


class TestIntersectConcave:
    """Curvature < 0 → center of curvature to the left of the vertex."""

    def test_hit_lies_on_sphere(self):
        origin = jnp.array([0.0, 5.0, -10.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        c = -0.02  # R = -50
        t, hit = intersect_sphere(origin, direction, curvature=c, z_offset=0.0)
        center = jnp.array([0.0, 0.0, 1.0 / c])  # (0, 0, -50)
        dist = float(jnp.linalg.norm(hit - center))
        assert dist == pytest.approx(50.0, abs=1e-3)

    def test_concave_sag_negative(self):
        """Concave surface: hit z should be < z_offset for off-axis ray."""
        h = 5.0
        R = 50.0
        c = -1.0 / R
        origin = jnp.array([0.0, h, -10.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, hit = intersect_sphere(origin, direction, curvature=c, z_offset=0.0)
        expected_sag = -(h ** 2 / (2 * R))  # -0.25
        assert float(hit[2]) == pytest.approx(expected_sag, abs=0.01)


class TestIntersectEdgeCases:
    """Edge cases: very small curvature, large offsets."""

    def test_very_small_curvature_behaves_like_flat(self):
        origin = jnp.array([0.0, 3.0, 0.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        _, hit_flat = intersect_sphere(origin, direction, curvature=0.0, z_offset=20.0)
        _, hit_tiny = intersect_sphere(origin, direction, curvature=1e-15, z_offset=20.0)
        assert float(hit_flat[2]) == pytest.approx(float(hit_tiny[2]), abs=0.1)

    def test_positive_t(self):
        """Intersection t should always be positive for a ray in front of the surface."""
        origin = jnp.array([0.0, 0.0, -5.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        t, _ = intersect_sphere(origin, direction, curvature=0.01, z_offset=0.0)
        assert float(t) > 0

    def test_nonzero_z_offset(self):
        """Vertex at z_offset=100 should shift everything."""
        origin = jnp.array([0.0, 0.0, 90.0])
        direction = jnp.array([0.0, 0.0, 1.0])
        t, hit = intersect_sphere(origin, direction, curvature=0.0, z_offset=100.0)
        assert float(t) == pytest.approx(10.0, abs=1e-5)
        assert float(hit[2]) == pytest.approx(100.0, abs=1e-5)


# ---- surface_normal tests --------------------------------------------------

class TestSurfaceNormalPlanar:
    def test_flat_normal(self):
        hit = jnp.array([5.0, 3.0, 10.0])
        n = surface_normal(hit, curvature=0.0, z_offset=10.0)
        assert float(n[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(n[1]) == pytest.approx(0.0, abs=1e-6)
        assert float(n[2]) == pytest.approx(-1.0, abs=1e-6)


class TestSurfaceNormalSphere:
    def test_convex_on_axis(self):
        """On-axis hit on convex surface: normal should be [0, 0, -1]."""
        # On axis, hit is at vertex (0, 0, 0)
        hit = jnp.array([0.0, 0.0, 0.0])
        n = surface_normal(hit, curvature=0.02, z_offset=0.0)
        assert float(n[2]) == pytest.approx(-1.0, abs=1e-4)

    def test_concave_on_axis(self):
        """On-axis hit on concave surface: normal should be [0, 0, -1]."""
        hit = jnp.array([0.0, 0.0, 0.0])
        n = surface_normal(hit, curvature=-0.02, z_offset=0.0)
        assert float(n[2]) == pytest.approx(-1.0, abs=1e-4)

    def test_normal_is_unit_vector(self):
        hit = jnp.array([0.0, 8.0, 0.6414])  # approx sag for h=8, R=50
        n = surface_normal(hit, curvature=0.02, z_offset=0.0)
        assert float(jnp.linalg.norm(n)) == pytest.approx(1.0, abs=1e-4)

    def test_normal_opposes_propagation(self):
        """Normal z-component must be negative (opposing +z ray direction)."""
        for c in [0.02, -0.02, 0.05, -0.05]:
            hit = jnp.array([0.0, 5.0, 0.25])
            n = surface_normal(hit, curvature=c, z_offset=0.0)
            assert float(n[2]) < 0, f"normal_z should be < 0 for curvature {c}"

    def test_convex_normal_tilts_outward(self):
        """For a convex surface, off-axis hit should have normal with
        positive y-component (pointing away from axis in same direction as hit)."""
        hit = jnp.array([0.0, 10.0, 1.0])
        n = surface_normal(hit, curvature=0.02, z_offset=0.0)
        # Hit is at positive y, so normal y-component should be positive
        # (normal points radially outward from the sphere center)
        assert float(n[1]) > 0


# ---- gradient checks -------------------------------------------------------

class TestIntersectionGradients:
    def test_grad_wrt_curvature(self):
        """Autodiff and finite-difference should agree for d(hit_z)/d(curvature)."""

        def hit_z(c):
            o = jnp.array([0.0, 8.0, -15.0])
            d = jnp.array([0.0, 0.0, 1.0])
            _, hit = intersect_sphere(o, d, c, 0.0)
            return hit[2]

        c0 = 0.02
        ad_grad = float(jax.grad(hit_z)(c0))
        eps = 1e-4
        fd_grad = float((hit_z(c0 + eps) - hit_z(c0 - eps)) / (2 * eps))
        assert ad_grad == pytest.approx(fd_grad, rel=0.01)

    def test_grad_wrt_ray_origin(self):
        """Gradient of hit position w.r.t. ray origin y-coordinate."""

        def hit_y(oy):
            o = jnp.array([0.0, oy, -10.0])
            d = jnp.array([0.0, 0.0, 1.0])
            _, hit = intersect_sphere(o, d, 0.02, 0.0)
            return hit[1]

        ad_grad = float(jax.grad(hit_y)(5.0))
        eps = 1e-4
        fd_grad = float((hit_y(5.0 + eps) - hit_y(5.0 - eps)) / (2 * eps))
        assert ad_grad == pytest.approx(fd_grad, abs=0.01)

    def test_grad_is_finite(self):
        """Gradients should be finite for all reasonable inputs."""

        def loss(c):
            o = jnp.array([0.0, 5.0, -10.0])
            d = jnp.array([0.0, 0.0, 1.0])
            t, hit = intersect_sphere(o, d, c, 0.0)
            return t + jnp.sum(hit ** 2)

        for c in [0.0, 0.01, -0.01, 0.05, -0.05]:
            g = float(jax.grad(loss)(c))
            assert jnp.isfinite(g), f"Non-finite gradient for curvature={c}"

    def test_jit_compiles(self):
        """intersect_sphere and surface_normal must be JIT-compilable."""
        jitted_intersect = jax.jit(intersect_sphere)
        jitted_normal = jax.jit(surface_normal)

        o = jnp.array([0.0, 5.0, -10.0])
        d = jnp.array([0.0, 0.0, 1.0])
        t, hit = jitted_intersect(o, d, 0.02, 0.0)
        n = jitted_normal(hit, 0.02, 0.0)
        assert jnp.isfinite(t)
        assert jnp.all(jnp.isfinite(n))
