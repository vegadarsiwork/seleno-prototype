"""Independent precision checks; no matcher-produced points serve as truth."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from seleno.tool import evaluation as E
from seleno.tool.methods import Correspondences


class PrecisionEvaluation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="seleno_precision_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = {"matrix": np.eye(3).tolist(), "application": {"kind": "global"},
                      "frame": {"reference_decimation": 1}}
        self.write_model()
        y, x = np.mgrid[8:128:16, 8:128:16]
        self.points = np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)

    def write_model(self):
        (self.root / "transform.json").write_text(json.dumps(self.model))

    def correspondence(self, reference):
        return Correspondences(self.points, reference, np.ones(len(reference)), "test", "sparse")

    def test_native_source_error_uses_mapping_not_nominal_gsd(self):
        # An anisotropic source grid makes symmetric working-grid errors a
        # particularly misleading proxy: (0.4, 0.3) -> (4, 0.6) native pixels.
        corr = self.correspondence(self.points + [.4, .3])
        stats = E.score_export(self.root, corr, E.SpatialSplit((128, 128)),
                               working_to_source=lambda p: p * [10., 2.])
        self.assertAlmostEqual(stats["held_out_rmse_px"], .5)
        self.assertAlmostEqual(stats["held_out_source_rmse_px"], np.hypot(4., .6))
        self.assertEqual(stats["held_out_source_within_1_px"], 0.)
        np.testing.assert_allclose(stats["held_out_source_bias_xy_px"], [4., .6])

    def test_native_observations_are_not_replaced_by_prewarp_coordinates(self):
        corr = self.correspondence(self.points)
        corr.detail.update(back_x=self.points[:, 0] + 2., back_y=self.points[:, 1])
        stats = E.score_export(self.root, corr, E.SpatialSplit((128, 128)),
                               working_to_source=lambda p: p)
        self.assertEqual(stats["held_out_rmse_px"], 0.)
        self.assertEqual(stats["held_out_source_rmse_px"], 2.)

    def test_reference_statistics_use_native_forward_transfer(self):
        self.model["frame"]["reference_decimation"] = 4
        self.write_model()
        stats = E.score_export(self.root, self.correspondence(self.points + [.4, .3]),
                               E.SpatialSplit((128, 128)))
        self.assertAlmostEqual(stats["held_out_rmse_px"], .5)
        self.assertAlmostEqual(stats["held_out_reference_rmse_px"], 2.)
        self.assertEqual(stats["held_out_reference_within_1_px"], 0.)

    def test_acceptance_does_not_hide_systematic_bias_or_poor_spread(self):
        distribution = {"coverage_fraction": .8, "extrapolation_fraction": .1}
        good = {"held_out_source_" + k: v for k, v in E.error_statistics(np.zeros((20, 2))).items()}
        accepted = E.acceptance_report(good, distribution)
        self.assertTrue(accepted["passed"])
        self.assertFalse(accepted["independently_verified"])
        self.assertEqual(accepted["status"], "unverified")
        self.assertFalse(E.acceptance_report(good, distribution | {"coverage_fraction": .2})["passed"])
        self.assertFalse(E.acceptance_report(good, distribution | {"extrapolation_fraction": .8})["passed"])
        biased = {"held_out_source_" + k: v for k, v in E.error_statistics(np.tile([.8, 0], (20, 1))).items()}
        failed = E.acceptance_report(biased, distribution)
        self.assertFalse(failed["passed"])
        self.assertTrue(any("bias" in reason for reason in failed["reasons"]))

    def test_bad_tail_and_invalid_predictions_are_kept(self):
        errors = np.zeros((20, 2))
        errors[-2:, 0] = 2.
        stats = E.error_statistics(errors)
        self.assertLess(stats["rmse_px"], 1.)
        self.assertEqual(stats["within_1_px"], .9)
        self.assertTrue(E._precision_failures(stats))
        errors[-1] = np.nan
        stats = E.error_statistics(errors)
        self.assertEqual(stats["n"], 20)
        self.assertEqual(stats["invalid_n"], 1)
        self.assertEqual(stats["valid_n"], 19)
        self.assertAlmostEqual(stats["rmse_px"], np.sqrt(4. / 19))
        self.assertTrue(any("invalid" in r for r in E._precision_failures(stats)))

    def test_empty_all_invalid_and_single_valid_statistics(self):
        for errors in ([], [[np.nan, 0.], [0., np.inf]]):
            stats = E.error_statistics(errors)
            self.assertEqual(stats["valid_n"], 0)
            self.assertIsNone(stats["rmse_px"])
            self.assertTrue(E._precision_failures(stats))
        stats = E.error_statistics([[3., 4.], [np.nan, 0.]])
        self.assertEqual(stats["rmse_px"], 5.)
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["invalid_n"], 1)

    def test_invalid_prediction_outside_check_points_still_fails(self):
        corr = self.correspondence(self.points)
        def mapping(p):
            q = p.copy()
            q[0] = np.nan
            return q
        check = np.arange(len(self.points)) != 0
        acc = E.score_export(self.root, corr, E.SpatialSplit((128, 128)),
                             working_to_source=mapping, check_points=check)
        self.assertEqual(acc["held_out_source_invalid_n"], 1)
        self.assertEqual(acc["held_out_source_rmse_px"], 0.)
        self.assertEqual(acc["check_point_source_invalid_n"], 0)
        result = E.acceptance_report(acc, {"coverage_fraction": 1., "extrapolation_fraction": 0.})
        self.assertFalse(result["passed"])
        self.assertTrue(any("invalid" in reason for reason in result["reasons"]))

    def manifest(self):
        source, reference = self.root / "source.bin", self.root / "reference.bin"
        source.write_bytes(b"source native detector raster")
        reference.write_bytes(b"independent reference raster")
        document = {"schema_version": 1, "coordinates": "native_pixel_centres",
            "source": {"path": source.name, "sha256": E.file_sha256(source), "shape": [128, 128]},
            "reference": {"path": reference.name, "sha256": E.file_sha256(reference), "shape": [128, 128]},
            "provenance": {"kind": "synthetic_transform", "reference": "identity construction, seed 0",
                           "independent_of_registration": True, "uncertainty_source_px": 0.},
            "points": [{"id": str(k), "source": p.tolist(), "reference": p.tolist()}
                       for k, p in enumerate(self.points)]}
        path = self.root / "truth.json"
        path.write_text(json.dumps(document))
        args = dict(source_path=source, reference_path=reference,
                    source_shape=(128, 128), reference_shape=(128, 128))
        return path, document, args

    def test_independent_reference_scores_exact_export_and_detects_offset(self):
        path, _, args = self.manifest()
        result = E.score_ground_truth(self.root, path, working_to_source=lambda p: p, **args)
        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(result["statistics"]["rmse_px"], 0.)
        self.model["matrix"][0][2] = 2.5
        self.write_model()
        result = E.score_ground_truth(self.root, path, working_to_source=lambda p: p, **args)
        self.assertFalse(result["passed"])
        self.assertEqual(result["statistics"]["rmse_px"], 2.5)
        self.assertEqual(len(result["point_ids"]), len(self.points))

    def test_reference_rejects_wrong_images_nonindependence_and_duplicate_controls(self):
        for mutation in (lambda d: d["source"].update(sha256="incorrect"),
                         lambda d: d["provenance"].update(independent_of_registration=False),
                         lambda d: d["points"][1].update(source=d["points"][0]["source"]),
                         lambda d: d["points"][0].update(reference=[np.nan, 1.]),
                         lambda d: d["points"][0].update(source=[999., 1.])):
            path, document, args = self.manifest()
            mutation(document)
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                E.load_ground_truth(path, **args)

    def test_uncertain_reference_cannot_certify_subpixel_accuracy(self):
        path, document, args = self.manifest()
        document["provenance"]["uncertainty_source_px"] = 1.
        path.write_text(json.dumps(document))
        result = E.score_ground_truth(self.root, path, working_to_source=lambda p: p, **args)
        self.assertFalse(result["passed"])
        self.assertTrue(any("uncertainty" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
