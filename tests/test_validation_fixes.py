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
        self.assertEqual(float(np.sqrt(np.mean(residual ** 2))), metrics["accuracy"]["rmse_px"])
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
