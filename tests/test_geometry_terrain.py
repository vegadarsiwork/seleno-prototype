"""Independent lattice, native measurement and DEM coordinate regressions."""
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from scipy.ndimage import fourier_shift

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from seleno.tool import terrain, warp_model
from seleno.tool import evaluation as E
from seleno.tool.methods import Correspondences
from seleno.tool.coordinates import project

RG = importlib.import_module("seleno.tool.register")


class GeometryLattice(unittest.TestCase):
    def lattice(self):
        pixels, scans = np.linspace(0, 100, 6), np.linspace(0, 200, 11)
        s, l = np.meshgrid(pixels, scans)
        x, y = s + .002 * (l - 100) ** 2, l
        pts = np.column_stack([x.ravel(), y.ravel()])
        return RG._LatticeInverse(x, y, pixels, scans,
                                  RG._Extrapolating(pts, s.ravel()),
                                  RG._Extrapolating(pts, l.ravel()))

    def test_curved_edge_hull_sliver_does_not_become_edge_column(self):
        inverse = self.lattice()
        # Inside the nodes' convex hull, but outside the curved right edge.
        x, y = np.array([110., 60., 100.25, 100.75]), np.full(4, 100.)
        s, l = inverse.solve(x, y)
        self.assertTrue(np.isnan(s[[0, 3]]).all())
        self.assertTrue(np.isnan(l[[0, 3]]).all())
        np.testing.assert_allclose(s[[1, 2]], [60., 100.25], atol=1e-8)

    def test_bilinear_roundtrip_and_query_cache(self):
        inverse = self.lattice()
        s, l = np.array([10., 30., 80.]), np.array([15., 65., 185.])
        (x, y), _ = inverse.forward(s, l)
        np.testing.assert_allclose(inverse.sample(x, y), s, atol=1e-8)
        np.testing.assert_allclose(inverse.line(x, y), l, atol=1e-8)
        x[1] += 2.  # Same buffers and endpoints, changed interior query.
        np.testing.assert_allclose(inverse.sample(x, y), s + [0, 2, 0], atol=1e-8)

    def test_rim_extrapolation_is_bounded_and_handles_four_nodes(self):
        pts = np.array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
        f = RG._Extrapolating(pts, 2 * pts[:, 0] - pts[:, 1] + 7)
        actual = f(np.array([-.25, 5., -100., np.nan]), np.array([5., 5., 5., 5.]))
        np.testing.assert_allclose(actual[:2], [1.5, 12.])
        self.assertTrue(np.isnan(actual[2:]).all())


class TerrainCoordinates(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(terrain._SAMPLERS.clear)
        self.root = Path(tmp.name)
        self.label = self.root / "height.lbl"
        self.label.write_text("LINES = 8\nLINE_SAMPLES = 10\nSAMPLE_BITS = 16\n"
                              "SCALING_FACTOR = 0.5\nOFFSET = 1737400\n"
                              "MAP_SCALE = 2\nMAXIMUM_LATITUDE = -80\n")
        row, col = np.mgrid[:8, :10]
        (4 * col + 6 * row).astype("<i2").tofile(self.root / "height.img")
        self.spec = {"dem": str(self.label), "reference_crs": terrain._LOLA_CRS,
                     "reference_transform": [2, 0, -10, 0, -2, 8],
                     "grid_to_reference": np.eye(3).tolist()}

    def test_height_centres_bilinear_edges_invalid_and_grid_transform(self):
        sample = terrain.HeightSampler(self.spec)
        p = np.array([[0., 0.], [2.25, 3.5], [9., 7.], [-1., 2.], [np.nan, 0]])
        np.testing.assert_allclose(sample(p)[:3], [0., 15., 39.], atol=1e-6)
        self.assertTrue(np.isnan(sample(p)[3:]).all())
        spec = dict(self.spec, grid_to_reference=[[2, 0, 1], [0, 2, 1], [0, 0, 1]])
        np.testing.assert_allclose(terrain.HeightSampler(spec)([[1., 2.]]), [21.], atol=1e-6)

    def test_term_serialization_forward_inverse_and_composed_conjugation(self):
        import json
        term = {"sampler": self.spec, "height_origin_m": 10.,
                "coefficient_px_per_m": [.01, -.02]}
        p = np.array([[2., 2.], [5., 4.], [7., 5.]])
        h = 2 * p[:, 0] + 3 * p[:, 1]
        np.testing.assert_allclose(terrain.term(term, p), (h - 10)[:, None] * [.01, -.02])
        model = json.loads(json.dumps({"matrix": np.eye(3).tolist(), "parallax": term}))
        q = warp_model.inverse_points(model, p)
        np.testing.assert_allclose(warp_model.forward_points(model, q), p, atol=1e-7)
        first = np.array([[3., 0, 11], [0, 2., -4], [0, 0, 1]])
        second = np.array([[.5, 0, -3], [0, .25, 2], [0, 0, 1]])
        carried = warp_model.conjugate(warp_model.conjugate(model, first), second)
        np.testing.assert_allclose(warp_model.inverse_points(carried, project(second @ first, p)),
                                   project(second @ first, q), atol=1e-7)
        # Outside the DEM the documented fallback is the base mapping.
        np.testing.assert_array_equal(terrain.term(term, [[-100., 0.]]), [[0., 0.]])


class CoarseLatticeMeasurements(unittest.TestCase):
    def test_patch_cross_validation_ignores_test_and_validation_measurements(self):
        rng = np.random.default_rng(13)
        src = rng.uniform(10, 790, (1200, 2))
        split = E.SpatialSplit((800, 800))
        make = lambda ref: Correspondences(src, ref, np.ones(len(src)), "test", "dense")
        coarse = make(src + [2., -1.] + rng.normal(0, .6, src.shape))
        fine = make(src + [2., -1.] + rng.normal(0, .1, src.shape))
        chosen, table = RG._select_fine_patch({41: coarse, 61: fine}, split, {})
        self.assertEqual(chosen, 61, table)
        held = split.labels(src) != split.FIT
        coarse.ref[held] += rng.normal(0, 5000, (held.sum(), 2))
        fine.ref[held] = np.nan
        self.assertEqual(RG._select_fine_patch({41: coarse, 61: fine}, split, {}), (chosen, table))
        # A tiny, easy subset cannot earn a larger patch by losing coverage.
        subset = E.subset(fine, np.arange(len(src)) % 2 == 0)
        chosen, table = RG._select_fine_patch({41: coarse, 61: subset}, split, {})
        self.assertEqual(chosen, 41, table)

    def test_locked_translation_is_refined_with_fixed_distributed_seeds(self):
        a = cv2.GaussianBlur(np.random.default_rng(912).random((192, 192)).astype(np.float32),
                             (0, 0), 2.)
        delta = np.array([2.35, -1.25])
        b = np.fft.ifftn(fourier_shift(np.fft.fftn(a), delta[::-1])).real
        valid = np.ones(a.shape, bool)
        c = RG._lattice_tiepoints(a, valid, b, valid, 2, -1, (6, 6))
        self.assertIsNotNone(c)
        self.assertGreater(len(c), 60)
        self.assertLess(np.median(np.linalg.norm(c.ref - c.src - delta, axis=1)), .06)
        self.assertGreater(np.ptp(c.src[:, 0]), 140)
        self.assertGreater(np.ptp(c.src[:, 1]), 140)
        self.assertGreaterEqual(c.detail["seeds"], len(c))
        self.assertIsNone(RG._lattice_tiepoints(a, valid & False, b, valid, 0, 0, (6, 6)))
        self.assertIsNone(RG._lattice_tiepoints(a, valid, b, valid, 300, 0, (6, 6)))

    def test_fine_patch_setting_reaches_measurement(self):
        a = np.ones((80, 80), np.float32)
        with patch.object(RG, "_FINE_PATCH", 61), patch.object(RG.M, "refine_correspondences") as refine:
            RG._measure_seeds(a, a > 0, a, a > 0, [[40., 40.]], 8)
        self.assertEqual(refine.call_args.kwargs["patch"], 61)


if __name__ == "__main__":
    unittest.main()
