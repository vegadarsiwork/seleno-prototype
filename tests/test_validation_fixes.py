"""Regressions for the 2026-09-23 functional audit. Run directly with Python."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool import register
from seleno.tool import evaluation as E
from seleno.tool.methods import Correspondences
from seleno.verify import transfer_error

REG = importlib.import_module("seleno.tool.register")


class ValidationFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="seleno_fixes_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(12)
        self.a = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32), (0, 0), 2)
        self.a = ((self.a - self.a.min()) / np.ptp(self.a) * 200 + 20).astype(np.uint8)
        self.src = self.root / "source.png"
        self.ref = self.root / "reference.png"
        cv2.imwrite(str(self.src), self.a)
        cv2.imwrite(str(self.ref), self.a)

    def run_pair(self, **kw):
        return register(str(self.src), str(self.ref), str(self.root / "out"),
                        max_side=512, verbose=False, **kw)

    def test_exported_matrix_is_the_evaluated_and_warped_matrix(self):
        with patch.object(REG, "_warp", wraps=REG._warp) as warp:
            result = self.run_pair(fine=False)
        self.assertNotEqual(result.status, "failed")
        path = Path(result.out_dir)
        transform = json.loads((path / "transform.json").read_text())
        evidence = json.loads((path / "evaluation.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        matrix = np.asarray(transform["matrix"], dtype=np.float64)
        self.assertEqual(matrix.tobytes(), np.asarray(warp.call_args_list[0].args[2]).tobytes())
        self.assertEqual(E.matrix_digest(matrix), evidence["matrix_sha256"])
        residual = transfer_error(matrix, np.array(evidence["source"]), np.array(evidence["reference"]))
        # Every held-out correspondence is scored; the headline is over the
        # check points, which the evidence marks individually.
        self.assertEqual(float(np.sqrt(np.mean(residual ** 2))), metrics["accuracy"]["held_out_rmse_px"])
        check = np.asarray(evidence["check_point"], bool)
        self.assertEqual(len(check), len(residual))
        self.assertEqual(float(np.sqrt(np.mean(residual[check] ** 2))), metrics["accuracy"]["rmse_px"])
        self.assertEqual(len(residual), metrics["accuracy"]["held_out_n"])

    def test_poisoning_test_points_cannot_change_selection_or_export(self):
        split = E.SpatialSplit((256, 256))
        y, x = np.mgrid[16:256:16, 16:256:16]
        src = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
        keep = split.labels(src) == split.TEST

        def execute(poison):
            ref = src.copy()
            if poison:
                ref[keep] += [90, -70]
            c = Correspondences(src, ref, np.ones(len(src), np.float32), "fixed", "sparse")
            original = REG.V.verify

            def checked(a, b, **kw):
                self.assertTrue(np.all(split.labels(a) == split.FIT))
                return original(a, b, **kw)

            with patch.object(REG, "_run_one", return_value=(c, "")), \
                    patch.object(REG.V, "verify", side_effect=checked):
                return self.run_pair(fine=False, subpixel=False)

        clean, poisoned = execute(False), execute(True)
        self.assertEqual(clean.transform["matrix"], poisoned.transform["matrix"])
        self.assertEqual(clean.metrics["method_used"], poisoned.metrics["method_used"])
        self.assertEqual(clean.metrics["matches"], poisoned.metrics["matches"])
        self.assertLess(clean.metrics["accuracy"]["rmse_px"], 1e-6)
        self.assertGreater(poisoned.metrics["accuracy"]["rmse_px"], 100)

    def test_heldout_does_not_control_ecc_adoption_or_segment_fits(self):
        split = E.SpatialSplit((256, 256))
        y, x = np.mgrid[16:256:16, 16:256:16]
        source = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
        labels = split.labels(source)
        proposal = np.eye(3)
        proposal[:2, 2] = [.25, -.25]
        models = []
        for poison in (False, True):
            reference = source.copy()
            reference[labels == split.VALIDATION] += [.25, -.25]
            if poison:
                reference[labels == split.TEST] += [90, -70]
            corr = Correspondences(source, reference, np.ones(len(source), np.float32), "fixed", "sparse")
            def polish(a, am, b, bm, initial, model):
                self.assertFalse(np.any(am & ~split.fit_mask()))
                return proposal.copy(), {"attempted": True, "adopted": False}
            with patch.object(REG, "_run_one", return_value=(corr, "")), \
                    patch.object(REG, "_ecc_polish", side_effect=polish):
                result = self.run_pair(fine=False, subpixel=True, segments=4)
            self.assertTrue(result.metrics["accuracy"]["ecc"]["adopted"])
            self.assertLess(result.metrics["accuracy"]["ecc"]["validation_rmse_px_after"], 1e-6)
            models.append(result.transform)
        self.assertEqual(models[0]["matrix"], models[1]["matrix"])
        self.assertEqual(models[0]["segments"], models[1]["segments"])
        self.assertGreater(result.metrics["accuracy"]["rmse_px"], 100.)

    def test_spatial_membership_does_not_depend_on_reference_or_confidence(self):
        split = E.SpatialSplit((1024, 512))
        rng = np.random.default_rng(2)
        pts = rng.uniform([0, 0], [511, 1023], (1000, 2))
        labels = split.labels(pts)
        self.assertEqual(set(labels), {split.FIT, split.VALIDATION, split.TEST})
        self.assertTrue(np.array_equal(labels, split.labels(pts.copy())))
        for k in range(3):
            self.assertTrue(set(split.cells(pts[labels == k])).isdisjoint(
                split.cells(pts[labels != k])))

    def test_segment_export_blends_coordinates_and_scores_applied_model(self):
        from seleno.tool import warp_model as wm
        def translated(dx):
            h = np.eye(3)
            h[0, 2] = dx
            return h.tolist()
        segments = [{"axis": "row", "from": 0., "to": 128., "matrix": translated(8)},
                    {"axis": "row", "from": 128., "to": 256., "matrix": translated(-8)}]
        with patch.object(REG, "_segment_fit", return_value=segments):
            result = self.run_pair(fine=False, subpixel=False, segments=2)
        path = Path(result.out_dir)
        model = json.loads((path / "transform.json").read_text())
        evidence = json.loads((path / "evaluation.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        with rasterio.open(path / "registered.tif") as ds:
            exported = ds.read(1)
        # Independently sample the original input at three positions in the
        # actual inverse field; global identity would fail at either end.
        q = np.array([[100., 32.], [100., 128.], [100., 224.]])
        src = wm.inverse_points(model, q)
        np.testing.assert_allclose(src[:, 0], [92, 100, 108], atol=1e-9)
        # Raster export preserves the source digital numbers.
        expected = self.a[src[:, 1].astype(int), src[:, 0].astype(int)]
        np.testing.assert_allclose(exported[q[:, 1].astype(int), q[:, 0].astype(int)], expected, atol=1e-6)
        np.testing.assert_allclose(wm.forward_points(model, src), q, atol=1e-7)
        seam = np.column_stack([np.full(1001, 100.), np.linspace(127.5, 128.5, 1001)])
        field = wm.inverse_points(model, seam)
        self.assertLess(np.max(np.abs(np.diff(field[:, 0]))), .001)
        residual = wm.residuals(model, np.asarray(evidence["source"]), np.asarray(evidence["reference"]))
        self.assertEqual(float(np.sqrt(np.mean(residual ** 2))), metrics["accuracy"]["rmse_px"])
        self.assertEqual(metrics["accuracy"]["applied_model"]["kind"], "blended_segments")
        self.assertEqual(wm.digest(model), evidence["transform_sha256"])
        self.assertGreater(metrics["accuracy"]["rmse_px"], 3.)

    def test_illumination_criterion_detects_scale_error_with_zero_centre_shift(self):
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT / "scripts"))
        from benchmark_illumination import evaluate
        y, x = np.mgrid[32:512:32, 32:512:32]
        source = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
        identity = evaluate(SimpleNamespace(kp_src=source, kp_ref=source.copy()))
        self.assertTrue(identity["solved"])
        # Median translation is zero, but the corners are > 3 px wrong.
        reference = (source - 256) * 1.04 + 256
        scaled = evaluate(SimpleNamespace(kp_src=source, kp_ref=reference))
        self.assertFalse(scaled["solved"])
        self.assertGreater(scaled["corner_error_px"], 10.)
        self.assertEqual(len(scaled["corner_errors_px"]), 4)

    def test_sensor_defaults_and_aspect_aware_memory_cap(self):
        from seleno.tool.profiles import Profiles
        profile = Profiles.load().get("iirs")["registration"]
        self.assertEqual(profile, {"max_side": 6144, "grid": [12, 12]})
        with patch.object(REG, "_available_bytes", return_value=4_000_000_000):
            square, _ = REG._cap_max_side(6144, shape=(10000, 10000))
            strip, _ = REG._cap_max_side(6144, shape=(10000, 500))
        self.assertLess(square, 2048)
        self.assertEqual(strip, 6144)
        # Actual dispatch must apply profile defaults, but keep explicit options.
        src = self.root / "iirs_source.png"
        cv2.imwrite(str(src), self.a)
        for options, expected in [({}, 6144), ({"max_side": 512}, 512)]:
            with patch.object(REG, "_cap_max_side", side_effect=lambda value, **kw: (value, None)) as cap:
                result = register(str(src), str(self.ref), str(self.root / "out"),
                                  fine=False, verbose=False, **options)
            self.assertNotEqual(result.status, "failed")
            self.assertEqual(cap.call_args.args[0], expected)
            self.assertEqual(result.metrics["distribution"]["grid"], [12, 12])

    def test_pass_requires_native_precision_and_statement_uses_source_pixels(self):
        from seleno.tool.scene import Scene
        from types import SimpleNamespace
        scene = Scene(path="scaled", array=self.a, valid=self.a > 0, reader="test")
        conv = {"reference_decimation": 2, "source_px_per_reference_px": 3.,
                "reference_m_per_px": 6.}
        # Use the actual conversion schema, overriding its measured scale.
        conv = REG._unit_conversions(scene, scene, {"reference_decimation": 2}) | conv
        vr = SimpleNamespace(inlier_mask=np.ones(100, bool), n_inliers=100, inlier_ratio=1.)
        for error, subpixel in [(.2, False), (.1, True), (None, None)]:
            m = REG._write_metrics(str(self.root), "test", scene, scene, {}, {}, [],
                {"name": "test", "model": "affine"}, vr, {"held_out_rmse_px": error},
                .9, .7, 12., [], [], {}, 0., (8, 8), list(range(64)), .1, conv, {})
            self.assertEqual(m["status"], "warning")
            self.assertFalse(m["acceptance"]["passed"])
            self.assertFalse(m["accuracy"]["independently_verified_subpixel"])
            self.assertIs(m["accuracy"]["subpixel"], subpixel)
            self.assertIn("reference sampling scale 3.00 source px", m["accuracy_statement"])
            self.assertIn("not an accuracy guarantee", m["status_meaning"])
        # Native precision is measured directly, rather than inferred from
        # nominal GSD. Even excellent internal residuals need external controls
        # before they can certify source-pixel accuracy.
        errors = np.tile([[.1, 0], [-.1, 0]], (10, 1))
        acc = {"held_out_rmse_px": .1} | {
            "held_out_source_" + key: value for key, value in E.error_statistics(errors).items()}
        m = REG._write_metrics(str(self.root), "test", scene, scene, {}, {}, [],
            {"name": "test", "model": "affine"}, vr, acc,
            .9, .7, 12., [], [], {}, 0., (8, 8), list(range(64)), .1, conv, {})
        self.assertEqual(m["status"], "pass")
        self.assertAlmostEqual(m["accuracy"]["rmse_source_px"], .1)
        self.assertFalse(m["accuracy"]["independently_verified_subpixel"])

    def test_finite_nodata_is_excluded_at_native_and_decimated_resolution(self):
        from seleno.tool.scene import Scene
        data = np.ones((256, 256), np.float32)
        data[64:192, 64:192] = -9999
        scene = Scene(path="finite-fill", array=data, valid=data != -9999,
                      reader="test", nodata=-9999, meta_scale=2., meta_offset=3.)
        for side, missing in [(256, 128 ** 2), (128, 64 ** 2)]:
            step, origin, raster, valid, taps = REG._target_grid(scene, side)
            self.assertEqual(int((~valid).sum()), missing)
            np.testing.assert_array_equal(raster[valid], 5.)
        source = self.root / "nodata.tif"
        with rasterio.open(source, "w", driver="GTiff", height=256, width=256,
                           count=1, dtype="float32", nodata=-9999) as ds:
            ds.write(np.where(data == -9999, data, self.a).astype(np.float32), 1)
        self.src = self.ref = source
        result = self.run_pair(fine=False)
        with rasterio.open(Path(result.out_dir) / "registered.tif") as ds:
            self.assertTrue(np.isnan(ds.read(1)[80:176, 80:176]).all())

    def test_rotated_geotiff_scales_entire_affine_and_preserves_centres(self):
        from seleno.tool.scene import Scene
        transform = Affine(2, .3, 100, .2, -2, 200)
        scene = Scene(path="rotated", array=self.a, valid=np.ones(self.a.shape, bool),
                      reader="test", transform=transform,
                      crs="+proj=stere +lat_0=-90 +R=1737400 +units=m")
        frame = {"reference_decimation": 6, "reference_origin": [30, 12],
                 "reference_sample_offset": [1.5, 1.5]}
        REG._write_registered(str(self.root), self.a, np.ones(self.a.shape, bool), scene, frame)
        with rasterio.open(self.root / "registered.tif") as ds:
            expected = transform * Affine.translation(11, 29) * Affine.scale(6)
            np.testing.assert_allclose(tuple(ds.transform), tuple(expected), atol=1e-12)
            np.testing.assert_allclose(ds.transform * (.5, .5), transform * (14, 32), atol=1e-12)

    def test_identical_images_export_identical_pixel_centres(self):
        for georeferenced in (False, True):
            with self.subTest(georeferenced=georeferenced):
                if georeferenced:
                    path = self.root / "identity.tif"
                    with rasterio.open(path, "w", driver="GTiff", height=256, width=256,
                                       count=1, dtype="uint8",
                                       crs="+proj=stere +lat_0=-90 +R=1737400 +units=m",
                                       transform=Affine(2, 0, 100, 0, -2, 100)) as ds:
                        ds.write(self.a, 1)
                    self.src = self.ref = path
                result = self.run_pair(fine=False)
                self.assertNotEqual(result.status, "failed")
                points = np.genfromtxt(Path(result.out_dir) / "matches.csv", delimiter=",", names=True)
                np.testing.assert_allclose(points["src_x"], points["ref_x"], atol=1e-6, rtol=0)
                np.testing.assert_allclose(points["src_y"], points["ref_y"], atol=1e-6, rtol=0)

    def test_zero_displacement_points_are_remeasured_not_kept_on_the_prewarp(self):
        from seleno.tool.methods import refine_quantised
        rng = np.random.default_rng(4)
        A = cv2.GaussianBlur(rng.random((200, 200)).astype(np.float32), (0, 0), 2.0) * 100
        d = np.array([0.3, -0.4])        # B(x) = A(x - d): ground moved by +d
        B = cv2.warpAffine(A, np.float32([[1, 0, d[0]], [0, 1, d[1]]]), (200, 200),
                           flags=cv2.INTER_CUBIC)
        ok = np.ones(A.shape, bool)
        seeds = np.array([[x, y] for y in range(50, 151, 25) for x in range(50, 151, 25)], np.float32)
        subpix = np.array([[60.25, 70.75]], np.float32)
        c = Correspondences(np.vstack([seeds, subpix]), np.vstack([seeds, subpix + d]),
                            np.ones(len(seeds) + 1, np.float32), "orb", "sparse")
        out = refine_quantised(c, A, ok, B, ok)
        self.assertEqual(out.detail["quantised"], len(seeds))
        self.assertEqual(out.detail["dropped"], 0)
        step = out.ref[:-1].astype(float) - out.src[:-1].astype(float)
        np.testing.assert_allclose(np.median(step, axis=0), d, atol=0.05)
        self.assertFalse(np.any(np.all(np.abs(step - np.round(step)) < 1e-6, axis=1)))
        np.testing.assert_array_equal(out.src[-1], subpix[0])      # already sub-pixel: untouched
        np.testing.assert_array_equal(out.ref[-1], (subpix + d).astype(np.float32)[0])
        # A genuine zero offset (same image) is measured as zero and kept, not dropped.
        same = refine_quantised(Correspondences(seeds, seeds.copy(), np.ones(len(seeds), np.float32),
                                                "orb", "sparse"), A, ok, A, ok)
        self.assertEqual(len(same), len(seeds))
        np.testing.assert_allclose(same.ref - same.src, 0, atol=0.05)
        # Dense tie points are already this measurement and pass through untouched.
        dense = Correspondences(seeds, seeds.copy(), np.ones(len(seeds), np.float32), "dense-ncc", "dense")
        self.assertIs(refine_quantised(dense, A, ok, B, ok), dense)

    def test_fractional_backmap_and_decimation_centres(self):
        from seleno.tool.coordinates import grid_to_reference, project, sample_backmap
        frame = {"reference_decimation": 6, "reference_origin": [30, 12],
                 "reference_sample_offset": [1.5, 1.5]}
        yy, xx = np.indices((20, 20))
        bx, by = 12 + 1.5 + 6 * xx, 30 + 1.5 + 6 * yy
        pts = np.array([[2.123456789, 3.87654321], [10.5, 9.25]])
        np.testing.assert_allclose(sample_backmap(bx, by, pts),
                                   project(grid_to_reference(frame), pts), atol=1e-10, rtol=0)


if __name__ == "__main__":
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    unittest.main()
