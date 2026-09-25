"""Dense tie points, the local correction field, consensus significance and check points."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from seleno import verify as V
from seleno.tool import local_model as LM
from seleno.tool import warp_model as WM
from seleno.tool import evaluation as E
from seleno.tool.methods import Correspondences
from seleno.tool.register import _consistent, _fit_dense, _square_cells
from seleno.tool.scene import Scene, NativeBand
from seleno.tool.export import write_registered_native


def smooth_field(p):
    return np.column_stack([2.5 * np.sin(p[:, 1] / 180.0) + 0.001 * p[:, 0],
                            1.5 * np.cos(p[:, 0] / 150.0 + p[:, 1] / 400.0)])


class LocalField(unittest.TestCase):
    def test_field_recovers_a_smooth_distortion_through_outliers(self):
        rng = np.random.default_rng(0)
        shape = (900, 600)
        q = rng.uniform([0, 0], [600, 900], (3000, 2))
        r = smooth_field(q) + rng.normal(0, .2, q.shape)
        bad = rng.random(len(q)) < .1
        r[bad] += rng.normal(0, 15, (bad.sum(), 2))
        cells = E.SpatialSplit(shape).cells(q)
        cands = [(s, lam) for s in (150, 75) for lam in (.1, 1.)]
        best, table = LM.cross_validate(q, r, cells, shape, cands)
        self.assertIsNotNone(best, table)
        field = LM.fit(q, r, shape, *best)
        probe = rng.uniform([0, 0], [600, 900], (500, 2))
        err = np.linalg.norm(LM.evaluate(field, probe) - smooth_field(probe), axis=1)
        self.assertLess(np.median(err), .1)

    def test_null_model_wins_when_there_is_nothing_to_model(self):
        rng = np.random.default_rng(1)
        shape = (800, 800)
        q = rng.uniform(0, 800, (2000, 2))
        r = rng.normal(0, .3, q.shape)              # pure measurement noise
        best, table = LM.cross_validate(q, r, E.SpatialSplit(shape).cells(q), shape,
                                        [(100, .1), (200, 1.)])
        self.assertIsNone(best, table)

    def test_field_travels_with_the_model_to_another_grid(self):
        rng = np.random.default_rng(2)
        q = rng.uniform(0, 400, (800, 2))
        field = LM.fit(q, smooth_field(q), (400, 400), 100, .1)
        model = {"matrix": np.eye(3).tolist(), "local_field": field}
        D = np.array([[3., 0, 11], [0, 3., -4], [0, 0, 1]])
        carried = WM.conjugate(model, D)
        p = rng.uniform(20, 380, (50, 2))
        expected = (WM.inverse_points(model, p) * 3 + [11, -4])
        np.testing.assert_allclose(WM.inverse_points(carried, p * 3 + [11, -4]), expected, atol=1e-9)
        # forward is the numerical inverse of the complete field
        fwd = WM.forward_points(model, WM.inverse_points(model, p))
        np.testing.assert_allclose(fwd, p, atol=1e-6)


class DenseFit(unittest.TestCase):
    def test_dense_fit_uses_the_field_and_keeps_distorted_points(self):
        rng = np.random.default_rng(3)
        shape = (1000, 700)
        split = E.SpatialSplit(shape, grid=_square_cells((8, 8), shape))
        src = rng.uniform([0, 0], [700, 1000], (4000, 2))
        H = np.array([[1.01, .02, 5.], [-.015, .99, -3.], [0, 0, 1]])
        ref = np.column_stack([src, np.ones(len(src))]) @ H.T
        ref = ref[:, :2]
        # a smooth distortion several times the tight threshold
        truth = {"matrix": H.tolist(), "local_field": LM.fit(
            ref, smooth_field(ref), shape, 80., .01)}
        ref = WM.forward_points(truth, src)
        ref += rng.normal(0, .1, ref.shape)
        wrong = rng.random(len(src)) < .08
        ref[wrong] += rng.uniform(-40, 40, (wrong.sum(), 2))
        fit = split.labels(src) == split.FIT
        c = Correspondences(src[fit], ref[fit], np.ones(fit.sum(), np.float32), "t", "dense")
        vr, field, info = _fit_dense(c, "affine", shape, (12, 12), split, tight=1.)
        self.assertIsNotNone(field, info.get("field_cv"))
        # nearly every correct point is an inlier, nearly no mismatch is
        good = ~wrong[fit]
        self.assertGreater(vr.inlier_mask[good].mean(), .95)
        self.assertLess(vr.inlier_mask[~good].mean(), .1)
        # and the unseen folds are predicted to a fraction of a pixel
        model = {"matrix": vr.model.tolist(), "local_field": field}
        held = ~fit & ~wrong
        err = np.linalg.norm(WM.forward_points(model, src[held]) - ref[held], axis=1)
        self.assertLess(np.median(err), .5)

    def test_neighbour_screen_removes_isolated_mismatches_not_offsets(self):
        rng = np.random.default_rng(4)
        src = rng.uniform(0, 500, (600, 2))
        v = np.tile([4., -3.], (600, 1)) + rng.normal(0, .1, (600, 2))   # a shared offset
        v[:10] += [25., 30.]                                          # isolated mismatches
        keep = _consistent(src, v, floor=1.)
        self.assertFalse(keep[:10].any())
        self.assertTrue(keep[10:].all())


class Significance(unittest.TestCase):
    def test_random_consensus_is_not_significant_but_real_one_is(self):
        area = 256 * 256
        self.assertGreater(V.log10_nfa(45, 7, "homography", 3., area), 0)
        self.assertLess(V.log10_nfa(45, 30, "homography", 3., area), -20)
        # the same count means more when the null hypothesis spreads points further
        self.assertLess(V.log10_nfa(20, 10, "similarity", 3., area),
                        V.log10_nfa(20, 10, "similarity", 3., 25 * 25))

    def test_degenerate_models_are_named(self):
        pts = np.array([[10., 10.], [200., 15.], [30., 180.], [190., 190.]])
        flip = np.array([[1., 0, 0], [0, -1., 256], [0, 0, 1]])
        self.assertIn("folds", V.degenerate(flip, pts))
        wild = np.array([[1., 0, 0], [0, 1., 0], [.004, 0, 1]])
        self.assertIn("varies", V.degenerate(wild, pts, extent=(0, 0, 255, 255)))
        rot = np.array([[0., -1, 256], [1., 0, 0], [0, 0, 1]])        # 90 degrees is allowed
        self.assertIsNone(V.degenerate(rot, pts, extent=(0, 0, 255, 255)))


class CheckPoints(unittest.TestCase):
    def test_check_points_are_scored_beside_every_held_out_point(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        split = E.SpatialSplit((200, 200))
        src = np.random.default_rng(5).uniform(0, 200, (300, 2))
        ref = src + [.2, 0]
        ref[:3] += 80
        model = {"matrix": np.eye(3).tolist(), "application": {"kind": "global"}}
        import json
        with open(Path(tmp.name) / "transform.json", "w") as fh:
            json.dump(model, fh)
        test = Correspondences(src, ref, np.ones(len(src)), "t", "dense")
        check = np.ones(len(src), bool)
        check[:3] = False
        acc = E.score_export(tmp.name, test, split, check_points=check, check_point_rule="rule")
        self.assertEqual(acc["check_point_rejected_n"], 3)
        self.assertAlmostEqual(acc["check_point_rmse_px"], .2, places=9)
        self.assertGreater(acc["held_out_rmse_px"], 5)


class ExportWindow(unittest.TestCase):
    def test_export_is_cropped_and_averages_a_finer_source(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # a 4x finer checkerboard: point sampling aliases, averaging gives grey
        fine = (np.indices((400, 400)).sum(axis=0) % 2).astype(np.float32)
        src = Scene(path="s", array=fine, valid=None, reader="t", gsd_m=.25,
                    native_bands=(NativeBand(fine),))
        ref = Scene(path="r", array=np.zeros((100, 100), np.float32), valid=None,
                    reader="t", gsd_m=1.)
        info = write_registered_native(tmp.name, src, ref, {}, {"matrix": np.eye(3).tolist()},
                                       window=(10, 20, 60, 90))
        self.assertEqual(info["shape"], [50, 70])
        self.assertEqual(info["supersampling"], [4])
        with rasterio.open(Path(tmp.name) / "registered.tif") as ds:
            np.testing.assert_allclose(ds.read(1), .5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
