"""Known-truth local metrology and image-only viewpoint regression tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import fourier_shift

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from seleno.tool.methods import Correspondences, refine_correspondences, sparse
from seleno import verify


class RefinedMeasurements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(823)
        cls.a = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32), (0, 0), 2)
        cls.valid = np.ones(cls.a.shape, bool)
        cls.seeds = np.asarray([(x + .137, y + .713) for y in range(40, 220, 20)
                               for x in range(40, 220, 20)], np.float64)

    def measure(self, b, source=None, predicted=None, **kwargs):
        source = self.seeds if source is None else source
        predicted = source.copy() if predicted is None else predicted
        c = Correspondences(source, predicted, np.ones(len(source)), "grid", "dense")
        return refine_correspondences(c, self.a * 30000 + 400, self.valid,
                                      b, self.valid, **kwargs)

    def test_fractional_seeds_are_remeasured_and_preserved(self):
        # Fourier translation supplies independent continuous geometry without
        # OpenCV's interpolation table quantisation or cubic-kernel shift bias.
        delta = np.array([1.35, -2.25])
        b = np.fft.ifftn(fourier_shift(np.fft.fftn(self.a), delta[::-1])).real
        measured = self.measure(b * 8000 + 900, predicted=self.seeds + [.19, -.16])
        self.assertEqual(len(measured), len(self.seeds))
        np.testing.assert_array_equal(measured.src, self.seeds)
        errors = np.linalg.norm(measured.ref - measured.src - delta, axis=1)
        self.assertLess(float(np.sqrt(np.mean(errors ** 2))), .07)
        self.assertLess(float(np.median(errors)), .04)
        self.assertEqual(measured.detail["kept_unrefined"], 0)

    def test_contrast_reversal_and_dn_scale_do_not_change_measurement(self):
        delta = np.array([.3, -.4])
        b = np.fft.ifftn(fourier_shift(np.fft.fftn(self.a), delta[::-1])).real
        direct = self.measure(b * 80 + 30)
        inverted = self.measure(1000 - b * 650)
        self.assertEqual(len(inverted), len(self.seeds))
        np.testing.assert_allclose(direct.ref, inverted.ref, atol=.002)
        self.assertLess(np.linalg.norm(np.median(inverted.ref-inverted.src, axis=0)-delta), .04)
        self.assertEqual(inverted.detail["refinement_counts"]["inverted_intensity"], len(self.seeds))

    def test_unmeasurable_seeds_are_dropped(self):
        blank = np.zeros_like(self.a)
        c = Correspondences(self.seeds, self.seeds.copy(), np.ones(len(self.seeds)), "orb", "sparse")
        result = refine_correspondences(c, blank, self.valid, blank, self.valid)
        self.assertEqual(len(result), 0)
        self.assertEqual(result.detail["dropped"], len(c))
        # Every seed is seven pixels from the true endpoint, outside this
        # explicitly bounded local search. No prediction can count as a match.
        delta = np.array([7., 0.])
        b = np.fft.ifftn(fourier_shift(np.fft.fftn(self.a), delta[::-1])).real
        result = self.measure(b, search=2)
        self.assertLess(len(result), len(self.seeds) // 10)

    def test_nodata_and_edges_do_not_create_observations(self):
        source = np.array([[2.1, 8.3], [80.2, 90.7], [140.2, 150.7]])
        mask = self.valid.copy()
        mask[80:100, 70:90] = False
        c = Correspondences(source, source.copy(), np.ones(3), "grid", "dense")
        result = refine_correspondences(c, self.a, mask, self.a, self.valid)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result.src, source[2:])
        self.assertLess(np.linalg.norm(result.ref - result.src), .08)


if __name__ == "__main__":
    unittest.main()
