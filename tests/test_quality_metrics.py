"""Quality diagnostics: tails, invalid denominators, units and spatial support."""
import copy
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from seleno import spatial
from seleno.tool import evaluation as E, quality as Q, warp_model as WM
from seleno.tool.methods import Correspondences


class QualityMetrics(unittest.TestCase):
    def test_thresholds_are_strict_and_include_invalid_predictions(self):
        stats = E.error_statistics([[0., 0.], [.25, 0.], [.5, 0.], [1., 0.], [2., 0.], [np.nan, 0.]])
        self.assertEqual([r["count"] for r in stats["thresholds"]], [1, 2, 3, 4])
        self.assertEqual(stats["thresholds"][2]["fraction_all"], .5)
        self.assertEqual(stats["thresholds"][2]["fraction_valid"], .6)
        self.assertAlmostEqual(stats["p95_px"], 1.8)
        self.assertEqual(stats["invalid_n"], 1)
        json.dumps(stats, allow_nan=False)

    def test_robust_scatter_does_not_hide_bias_or_tail(self):
        stats = E.error_statistics([[5., 2.], [5., 2.], [5., 2.], [25., 2.]])
        self.assertEqual(stats["nmad_xy_px"], [0., 0.])
        self.assertEqual(stats["bias_xy_px"], [10., 2.])
        self.assertGreater(stats["p95_px"], stats["median_px"])
        self.assertAlmostEqual(stats["rmse_xy_px"][0], np.sqrt(175.))

    def test_empty_and_all_invalid_have_no_fabricated_distances(self):
        for errors in ([], [[None, 0], [np.inf, 1]]):
            stats = E.error_statistics(errors)
            self.assertIsNone(stats["p95_px"])
            self.assertIsNone(stats["nmad_xy_px"])
            self.assertEqual(stats["thresholds"][0]["fraction_all"], None if not errors else 0.)
            json.dumps(stats, allow_nan=False)

    def evidence(self):
        transform = {"matrix": np.eye(3).tolist(), "application": {"kind": "global"}, "frame": {"reference_decimation": 2}}
        fingerprint = WM.digest(transform)
        metrics = {"accuracy": {"transform_sha256": fingerprint,
                                "units": {"metres_per_reference_px": 100}},
                   "source": {"shape": [100, 10]}, "acceptance": {"passed": False}}
        evidence = {"transform_sha256": fingerprint, "source": [[1, 2], [2, 4], [3, 6]],
                    "native_source": [[1, 2], [2, 4], [3, 6]],
                    "predicted_native_source": [[1, 2], [5, 8], [None, None]],
                    "check_point": [True, True, False],
                    "reference_error_vectors": [[0, 0], [6, 8], [0, 0]]}
        return metrics, evidence, transform

    def test_raw_and_screened_scores_retain_invalids_and_native_units(self):
        m, e, t = self.evidence()
        q = Q.from_evidence(m, e, t)
        self.assertEqual(q["source"]["all"]["invalid_n"], 1)
        self.assertEqual(q["source"]["screened"]["n"], 2)
        self.assertAlmostEqual(q["source"]["screened"]["rmse_px"], np.sqrt(12.5))
        self.assertAlmostEqual(q["reference"]["screened"]["rmse_px"], np.sqrt(50.))
        self.assertEqual(q["residuals"][2]["dx"], None)
        self.assertFalse(q["acceptance"]["passed"])
        json.dumps(q, allow_nan=False)

    def test_stale_transform_or_metrics_is_rejected(self):
        m, e, t = self.evidence()
        other = copy.deepcopy(t)
        other["matrix"][0][2] = 10
        with self.assertRaisesRegex(ValueError, "exported transform"):
            Q.from_evidence(m, e, other)
        m["accuracy"]["transform_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            Q.from_evidence(m, e, t)

    def test_old_reference_summary_does_not_invent_p95(self):
        m, e, t = self.evidence()
        del e["reference_error_vectors"]
        e["reference_errors"] = {"rmse_px": 2.6}
        q = Q.from_evidence(m, e, t)
        self.assertEqual(q["reference"]["all"]["rmse_px"], 2.6)
        self.assertNotIn("p95_px", q["reference"]["all"])
        self.assertIsNotNone(q["source"]["all"]["p95_px"])

    def test_read_only_loading_and_api_work_for_old_jobs(self):
        import tool_routes
        m, e, t = self.evidence()
        with tempfile.TemporaryDirectory() as root:
            job = Path(root) / "saved"
            job.mkdir()
            for name, value in (("metrics", m), ("evaluation", e), ("transform", t)):
                (job / (name + ".json")).write_text(json.dumps(value))
            before = {p.name: p.read_bytes() for p in job.iterdir()}
            with patch.object(tool_routes, "OUTPUTS", root), patch.dict(tool_routes._JOBS, {"test-quality": {"job_dir_id": "saved"}}):
                result = tool_routes.quality("test-quality")
                self.assertEqual(result["n"], 3)
            self.assertEqual(before, {p.name: p.read_bytes() for p in job.iterdir()})

    def test_score_export_records_directional_vectors_for_reproduction(self):
        _, _, model = self.evidence()
        points = np.array([[1, 1], [2, 2], [3, 3]], float)
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "transform.json").write_text(json.dumps(model))
            corr = Correspondences(points, points + [.3, .4], np.ones(3), "fixture", "sparse")
            acc = E.score_export(root, corr, E.SpatialSplit((10, 10)),
                                 working_to_source=lambda p: p * [10, 2], check_points=np.ones(3, bool))
            e = json.loads((Path(root) / "evaluation.json").read_text())
            np.testing.assert_allclose(e["reference_error_vectors"], np.tile([-.6, -.8], (3, 1)))
            self.assertAlmostEqual(acc["check_point_reference_p95_px"], 1.)
            self.assertAlmostEqual(acc["check_point_source_p95_px"], np.hypot(3., .8))

    def test_reference_headline_does_not_convert_symmetric_residual(self):
        import importlib
        from types import SimpleNamespace
        from seleno.tool.scene import Scene
        reg = importlib.import_module("seleno.tool.register")
        scene = Scene(path="test", array=np.zeros((4, 4)), valid=np.ones((4, 4), bool), reader="test")
        vr = SimpleNamespace(inlier_mask=np.ones(20, bool), n_inliers=20, inlier_ratio=1.)
        acc = {"held_out_rmse_px": 10., "held_out_n": 20,
               "check_point_n": 20, "check_point_rejected_n": 0,
               "check_point_rmse_px": 9., "check_point_reference_rmse_px": 2.6}
        conv = {"reference_decimation": 2, "metres_per_reference_px": 100,
                "source_px_per_reference_px": 2.}
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        m = reg._write_metrics(tmp.name, "test", scene, scene, {}, {}, [],
                               {"name": "test", "model": "affine"}, vr, acc,
                               1., 1., 200., [], [], {}, 0., (2, 2), [1, 2, 3, 4], 0., conv, {})
        self.assertEqual(m["accuracy"]["rmse_reference_px"], 2.6)
        self.assertEqual(m["accuracy"]["rmse_m"], 260.)
        self.assertIsNone(m["accuracy"]["subpixel_attainable"])

    def test_narrow_strip_support_uses_valid_pixels_not_cell_centres(self):
        mask = np.zeros((100, 100), bool)
        mask[:, 8:13] = True
        points = [[8, 0], [12, 0], [12, 99], [8, 99]]
        self.assertEqual(spatial.extrapolation_fraction(np.array(points), mask.shape, (10, 2)), 1.)
        support = spatial.valid_pixel_support(points, mask)
        self.assertEqual(support["supported_fraction"], 1.)
        self.assertEqual(support["valid_pixels"], 500)
        self.assertEqual(support["sampled_pixels"], 500)

    def test_support_sampling_is_bounded_and_reports_empty_or_degenerate(self):
        mask = np.ones((1200, 1000), bool)
        points = [[0, 0], [999, 0], [999, 1199], [0, 1199]]
        support = spatial.valid_pixel_support(points, mask, max_samples=37)
        self.assertEqual(support["sampled_pixels"], 37)
        self.assertEqual(support["supported_fraction"], 1.)
        self.assertEqual(spatial.valid_pixel_support([[1, 1], [2, 2]], mask, 37)["extrapolation_fraction"], 1.)
        self.assertIsNone(spatial.valid_pixel_support(points, np.zeros((3, 3), bool))["extrapolation_fraction"])


class GroundMetres(unittest.TestCase):
    """Surface metres come from the datum, not the nominal pixel size."""

    def setUp(self):
        from rasterio.crs import CRS
        from rasterio.transform import Affine
        self.CRS, self.Affine = CRS, Affine
        R = 1737400.
        self.eqc = CRS.from_proj4("+proj=eqc +lat_ts=0 +lon_0=0 +R=%g +units=m +no_defs" % R)
        # 100 m pixels, row 0 at 90 N: WAC-like.
        self.wac = Affine(100., 0, -np.pi * R, 0, -100., np.pi / 2 * R)

    def row_at(self, lat):
        R = 1737400.
        return (np.pi / 2 * R - np.radians(lat) * R) / 100. - .5

    def test_cylindrical_east_shrinks_with_latitude(self):
        from seleno.tool import geodesy
        for lat in (0., -37.4, -60., -72.1):
            J = geodesy.ground_jacobian(self.eqc, self.wac, [[1000., self.row_at(lat)]])[0]
            self.assertAlmostEqual(J[0, 0], 100 * np.cos(np.radians(lat)), places=3)
            self.assertAlmostEqual(J[1, 1], -100., places=3)     # +row is south
            self.assertAlmostEqual(J[0, 1], 0., places=6)

    def test_error_vectors_become_east_north_metres(self):
        from seleno.tool import geodesy
        p = [[1000., self.row_at(-60.)]] * 2
        g = geodesy.ground_errors(self.eqc, self.wac, p, [[2., 0.], [0., 1.]])
        np.testing.assert_allclose(g, [[100., 0.], [0., -100.]], atol=1e-3)

    def test_geographic_and_polar_crs(self):
        from seleno.tool import geodesy
        ll = self.CRS.from_proj4("+proj=longlat +R=1737400 +no_defs")
        d = 1 / 256
        J = geodesy.ground_jacobian(ll, self.Affine(d, 0, 141, 0, -d, -15), [[0, 0]])[0]
        self.assertAlmostEqual(J[1, 1], -np.radians(d) * 1737400, places=3)
        ps = self.CRS.from_proj4("+proj=stere +lat_0=-90 +lon_0=0 +k=1 +R=1737400 +units=m +no_defs")
        J = geodesy.ground_jacobian(ps, self.Affine(1, 0, 0, 0, -1, 20000), [[0, 0]])[0]
        np.testing.assert_allclose(np.linalg.svd(J, compute_uv=False), 1., atol=1e-3)
        self.assertEqual(geodesy.describe(ps)["semi_major_m"], 1737400.)

    def test_score_export_records_ground_errors(self):
        from seleno.tool import geodesy
        transform = {"matrix": np.eye(3).tolist(), "application": {"kind": "global"},
                     "frame": {"reference_decimation": 1, "reference_origin": [0, 0]}}
        row = self.row_at(-60.)
        src = np.array([[10., row], [20., row], [30., row], [40., row]])
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "transform.json").write_text(json.dumps(transform))
            corr = Correspondences(src, src + [1., 0.], np.ones(4), "fixture", "sparse")
            acc = E.score_export(root, corr, E.SpatialSplit((5000, 100)),
                                 reference_ground=lambda p, e: geodesy.ground_errors(self.eqc, self.wac, p, e),
                                 ground_frame=geodesy.describe(self.eqc))
            e = json.loads((Path(root) / "evaluation.json").read_text())
        # Forward error is -1 reference px in x: 50 m west at 60 S, not 100 m.
        self.assertAlmostEqual(acc["held_out_ground_rmse_m"], 50., places=2)
        np.testing.assert_allclose(acc["held_out_ground_bias_en_m"], [-50., 0.], atol=1e-2)
        self.assertEqual(len(e["ground_error_vectors"]), 4)
        self.assertIn("datum", e["ground_frame"])


class StatisticsForms(unittest.TestCase):
    def test_component_sigma_and_metre_keys(self):
        stats = E.error_statistics([[1., 2.], [3., 2.], [5., 2.]], unit="m", axes="en", thresholds=())
        self.assertEqual(stats["bias_en_m"], [3., 2.])
        np.testing.assert_allclose(stats["std_en_m"], [2., 0.])
        self.assertNotIn("within_1_px", stats)
        self.assertEqual(stats["thresholds"], [])
        self.assertIsNone(E.error_statistics([[1., 1.]])["std_xy_px"])

    def test_pixel_keys_are_unchanged_for_clients(self):
        stats = E.error_statistics([[.5, 0.], [0., .5]])
        for key in ("rmse_px", "within_1_px", "bias_xy_px", "nmad_xy_px", "rmse_xy_px", "std_xy_px"):
            self.assertIn(key, stats)
        self.assertEqual(stats["thresholds"][2]["below_px"], 1.)


class Diagnostics(unittest.TestCase):
    def test_profiles_keep_empty_regions_and_measure_drift(self):
        y = np.linspace(0, 999, 60)
        observed = np.column_stack([np.full(60, 5.), y])
        errors = np.column_stack([y / 1000., np.zeros(60)])     # 1 px drift per 1000 lines
        observed[:5, 0] = np.nan                                  # unusable positions are ignored
        p = Q.profiles(observed, errors, (2000, 10))
        self.assertEqual(p["along_axis"], "line")
        self.assertEqual(sum(b["n"] for b in p["along"]), 55)
        self.assertEqual(p["along"][-1]["n"], 0)                  # lines 1000-1999 unmeasured
        self.assertAlmostEqual(p["drift_xy_px_per_1000"][0], 1., places=6)

    def test_small_bins_do_not_report_p95(self):
        observed = np.column_stack([np.zeros(4), np.arange(4.)])
        p = Q.profiles(observed, np.ones((4, 2)), (4, 1))
        self.assertTrue(all(b["p95_px"] is None for b in p["along"]))

    def test_block_bootstrap_is_reproducible_and_brackets_the_estimate(self):
        rng = np.random.default_rng(3)
        vectors = rng.normal(0, 1, (200, 2))
        blocks = np.repeat(np.arange(20), 10)
        a = Q.block_bootstrap(vectors, blocks, replicates=300)
        b = Q.block_bootstrap(vectors, blocks, replicates=300)
        self.assertEqual(a, b)
        rmse = np.sqrt(np.mean(np.sum(vectors ** 2, axis=1)))
        self.assertLess(a["rmse_px"][0], rmse)
        self.assertGreater(a["rmse_px"][1], rmse)
        self.assertIn("reason", Q.block_bootstrap(vectors, np.repeat(np.arange(4), 50)))

    def test_bootstrap_counts_invalid_predictions_as_failures(self):
        vectors = np.vstack([np.zeros((50, 2)), np.full((50, 2), np.nan)])
        out = Q.block_bootstrap(vectors, np.repeat(np.arange(10), 10), replicates=200)
        self.assertLess(out["below_1_px_fraction"][1], 1.)

    def test_warp_validity_identity_and_fold(self):
        model = {"matrix": np.eye(3).tolist(), "frame": {"target_shape": [200, 100]}}
        w = Q.warp_validity(model, (200, 100))
        self.assertTrue(w["available"])
        self.assertEqual(w["folded_fraction"], 0.)
        self.assertAlmostEqual(w["relative_area"]["median"], 1.)
        self.assertEqual(w["flags"], [])
        # A strong field whose x-derivative reverses the map on part of the grid.
        from seleno.tool import local_model
        pts = np.column_stack([np.tile(np.linspace(0, 99, 20), 20), np.repeat(np.linspace(0, 199, 20), 20)])
        bump = np.column_stack([-60 * np.exp(-((pts[:, 0] - 50) / 8) ** 2), np.zeros(len(pts))])
        field = local_model.fit(pts, bump, (200, 100), spacing=10., smoothness=1e-6)
        folded = dict(model, local_field=field)
        w = Q.warp_validity(folded, (200, 100))
        self.assertGreater(w["folded_fraction"], 0.)
        self.assertTrue(any("folds" in f for f in w["flags"]))

    def test_mirrored_pair_is_not_a_fold(self):
        mirror = np.array([[-1., 0, 99], [0, 1, 0], [0, 0, 1]])
        w = Q.warp_validity({"matrix": mirror.tolist(), "frame": {"target_shape": [50, 100]}}, (50, 100))
        self.assertEqual(w["folded_fraction"], 0.)

    def test_support_reports_distance_to_nearest_fit_point(self):
        mask = np.ones((10, 100), bool)
        s = spatial.valid_pixel_support([[0, 0], [0, 9], [9, 0], [9, 9]], mask, scale=2., unit="native reference px")
        self.assertAlmostEqual(s["distance_to_fit_point"]["max"], 2 * np.hypot(90, 4), places=6)
        self.assertEqual(s["distance_to_fit_point"]["unit"], "native reference px")
        self.assertGreater(s["extrapolation_fraction"], .8)


class Conventions(unittest.TestCase):
    def quality(self):
        m, e, t = QualityMetrics.evidence(None)
        e["ground_error_vectors"] = [[10., -5.], [30., 5.], [None, None]]
        m["source"]["gsd_m"] = 50.
        return Q.from_evidence(m, e, t)

    def test_agency_rows_restate_our_errors_in_their_forms(self):
        q = self.quality()
        rows = {c["id"]: c for c in q["conventions"]}
        self.assertEqual(set(rows), {"asp", "isis", "kaguya", "circular_error", "sldem"})
        self.assertEqual(rows["asp"]["guidance_met"], "not met")
        self.assertFalse(rows["asp"]["count_met"])                  # fewer than a dozen
        kaguya = rows["kaguya"]
        self.assertEqual(kaguya["ours"]["east_m"][0], 20.)          # screened: two finite points
        self.assertAlmostEqual(kaguya["ours"]["east_m"][1], np.std([10., 30.], ddof=1))
        np.testing.assert_allclose(kaguya["published_per_gsd"]["longitude"], [.54, .8])
        for row in q["conventions"]:
            self.assertTrue(row["different"])                         # every row says how it differs
        self.assertEqual(q["residuals"][0]["de"], 10.)
        json.dumps(q, allow_nan=False)

    def test_asp_guidance_levels(self):
        q = {"source": {"all": E.error_statistics([[.3, 0.]] * 20)},
             "residuals": [{"dx": .3, "dy": 0., "screened": None}] * 20}
        row = Q.conventions(q)[0]
        self.assertEqual(row["guidance_met"], "under 0.5 px")
        self.assertTrue(row["count_met"])


def _killed_like_oom(src, ref, options, out, channel):
    """Stands in for a registration the kernel kills at the memory cap."""
    import os
    import signal
    channel.send(("log", "settings  : about to be killed"))
    os.kill(os.getpid(), signal.SIGKILL)


class AppWorker(unittest.TestCase):
    """Each app registration runs in its own process and reports back."""

    def run_job(self, **patches):
        import time
        import tool_routes
        req = tool_routes.RunRequest(source="data/fixtures/synthetic_source.png",
                                     reference="data/fixtures/synthetic_reference.png")
        job = {"id": "worker-test", "state": "running", "stage": "starting", "log": [],
               "started": time.time(), "finished": None, "metrics": None, "status": None,
               "reason": None, "job_dir_id": None, "error": None}
        with tempfile.TemporaryDirectory() as out, \
                patch.object(tool_routes, "OUTPUTS", out), \
                patch.dict(tool_routes._JOBS, {"worker-test": job}):
            for name, value in patches.items():
                self.enterContext(patch.object(tool_routes, name, value))
            tool_routes._worker("worker-test", req, str(ROOT / req.source), str(ROOT / req.reference))
            written = os.listdir(out)
        return job, written

    def test_registration_runs_in_a_child_and_streams_its_log(self):
        job, written = self.run_job()
        self.assertEqual(job["state"], "done", job.get("error"))
        self.assertEqual(job["status"], "pass")
        self.assertIn(job["job_dir_id"], written)
        self.assertTrue(any(line["line"].startswith("settings") for line in job["log"]))
        grid = job["metrics"]["working_grid"]
        self.assertEqual(grid["used"], grid["planned"])
        self.assertIn(grid["basis"], ("requested", "memory limit"))

    def test_registrations_queue_instead_of_sharing_the_memory_cap(self):
        import threading
        import time
        import tool_routes
        active, peak, stages = [0], [0], []
        guard = threading.Lock()

        def fake_run(job, req, src, ref):
            with guard:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.3)
            with guard:
                active[0] -= 1
            job["state"] = "done"

        req = tool_routes.RunRequest(source="a", reference="b")
        jobs = {k: {"id": k, "state": "running", "stage": "starting", "log": [], "started": 0}
                for k in ("one", "two", "three")}
        with patch.object(tool_routes, "_run_job", fake_run), patch.dict(tool_routes._JOBS, jobs):
            threads = [threading.Thread(target=tool_routes._worker, args=(k, req, "a", "b"))
                       for k in jobs]
            for t in threads:
                t.start()
            time.sleep(0.1)
            stages = sorted(j["stage"] for j in jobs.values())
            for t in threads:
                t.join()
        self.assertEqual(peak[0], 1)
        self.assertEqual(stages.count("queued"), 2)
        self.assertTrue(all(j["state"] == "done" for j in jobs.values()))

    def test_a_killed_child_is_reported_not_left_running(self):
        job, _ = self.run_job(_run_child=_killed_like_oom)
        self.assertEqual(job["state"], "crashed")
        self.assertIn("signal 9", job["error"])
        self.assertIn("memory cap", job["error"])
        self.assertEqual(job["log"][0]["line"], "settings  : about to be killed")


if __name__ == "__main__":
    unittest.main()
