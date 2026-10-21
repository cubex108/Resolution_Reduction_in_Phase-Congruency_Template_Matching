"""Small scientific-contract tests. No dataset benchmark is launched."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pywt
import tifffile

import imaging_core as core


class InputTests(unittest.TestCase):
    def test_nir_uses_raw_fourth_band_even_with_minisblack_metadata(self):
        y, x = np.mgrid[:16, :20]
        data = np.stack((x + y, x * 2 + y, 40 - x, 3 * y + (x % 4) * 7), axis=-1).astype(np.uint16)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "RGBIR.tif"
            tifffile.imwrite(path, data, photometric="minisblack", planarconfig="contig",
                             metadata={"axes": "YXS"})
            actual, info = core.read_image(path, "nir", return_info=True)
            np.testing.assert_array_equal(actual, core.normalize_image(data[..., 3]))
            self.assertEqual(info["selected_channels_zero_based"], [3])
            self.assertEqual(info["decoder"], "tifffile")
            self.assertFalse(np.array_equal(actual, core.normalize_image(data[..., 0])))
            with self.assertRaisesRegex(ValueError, "lacks RGB metadata"):
                core.read_image(path, "gray")

    def test_rgb_is_rgb_order_and_never_bgr(self):
        data = np.array([[[0, 0, 0], [255, 0, 0]], [[0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.tif"
            tifffile.imwrite(path, data, photometric="rgb")
            actual, info = core.read_image(path, "rgb", return_info=True)
            expected = np.array([[0, .299], [.587, .114]]) / .587
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            self.assertGreater(actual[0, 1], actual[1, 1])
            self.assertEqual(info["rgb_weights"], [.299, .587, .114])
            with self.assertRaisesRegex(ValueError, "four-band"):
                core.read_image(path, "nir")

    def test_single_channel_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gray.tif"
            tifffile.imwrite(path, np.full((12, 14), 25, dtype=np.uint8))
            actual, info = core.read_image(path, return_info=True)
            self.assertTrue(info["constant"])
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, 0)
        with self.assertRaises(ValueError):
            core.normalize_image([1, np.nan])
        with self.assertRaisesRegex(ValueError, "normalized"):
            core.extract_feature(np.ones((16, 16)) * 255, "full_pc")
        with self.assertRaisesRegex(ValueError, "scalar"):
            core.extract_feature(np.zeros((16, 16, 3)), "full_pc")


class RandomControlsTests(unittest.TestCase):
    def test_reproducible_independent_streams(self):
        a = core.independent_rng(12, "p1:opt", "noise", 1, 0).normal(size=100)
        same = core.independent_rng(12, "p1:opt", "noise", 1, 0).normal(size=100)
        other_image = core.independent_rng(12, "p1:sar", "noise", 1, 0).normal(size=100)
        other_realization = core.independent_rng(13, "p1:opt", "noise", 1, 0).normal(size=100)
        np.testing.assert_array_equal(a, same)
        self.assertFalse(np.array_equal(a, other_image))
        self.assertFalse(np.array_equal(a, other_realization))
        with self.assertRaisesRegex(ValueError, "image_key"):
            core.independent_rng(1, None, "noise", 1, 0)

    def test_noise_matches_realized_rms_not_only_expected_variance(self):
        detail = np.random.default_rng(3).normal(size=(30, 33)) * .12 + .02
        actual = core.matched_noise(detail, np.random.default_rng(10))
        self.assertAlmostEqual(core.rms(actual), core.rms(detail), places=14)
        self.assertAlmostEqual(float(actual.mean()), 0.0, places=14)
        np.testing.assert_array_equal(core.matched_noise(np.zeros((10, 10)), np.random.default_rng(0)), 0)

    def test_surrogate_preserves_spectrum_mean_and_energy_even_and_odd(self):
        for shape in [(31, 34), (32, 33), (32, 34)]:
            detail = np.random.default_rng(2).normal(size=shape) - .12
            actual = core.phase_surrogate(detail, np.random.default_rng(7))
            np.testing.assert_allclose(np.abs(np.fft.rfft2(actual)), np.abs(np.fft.rfft2(detail)),
                                       rtol=1e-12, atol=1e-12)
            self.assertAlmostEqual(float(actual.mean()), float(detail.mean()), places=13)
            self.assertAlmostEqual(core.rms(actual), core.rms(detail), places=13)
            self.assertFalse(np.allclose(actual, detail))


class FeatureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, x = np.mgrid[:64, :66]
        cls.image = core.normalize_image(np.sin(x / 3.1) + np.cos(y / 5.3) + (x > y) * 2)

    def test_original_float_mfpc_definition_and_no_unused_detail_pc(self):
        image = self.image.astype(np.float64)
        approximation, detail = pywt.wavedec2(image, "db2", mode="symmetric", level=1)
        # An arbitrary bounded nonconstant PC map exposes detail retention and
        # clipping differences while allowing an exact call-count assertion.
        fake_pc = core.normalize_image(approximation).astype(np.float64) * .3
        expected = core.normalize_image(np.clip(pywt.waverec2([fake_pc, detail], "db2",
                                                               mode="symmetric")[:64, :66], 0, 1))
        with patch.object(core, "_pc", return_value=fake_pc) as pc:
            actual = core.extract_feature(image, "mfpc")
        self.assertEqual(pc.call_count, 1)
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_signed_details_reach_pc_without_clipping_or_uint8_wrap(self):
        inputs = []
        def fake(array, parameters):
            inputs.append(array.copy())
            return np.zeros_like(array)
        with patch.object(core, "_pc", side_effect=fake):
            core.extract_feature(self.image, "all_subbands")
        self.assertEqual(len(inputs), 4)
        self.assertTrue(any(np.any(a < 0) for a in inputs[1:]))
        self.assertTrue(all(np.issubdtype(a.dtype, np.floating) for a in inputs))

    def test_scale_matched_control_is_explicit_and_matches_rms_definition(self):
        image = self.image.astype(np.float64)
        approximation, detail = pywt.wavedec2(image, "db2", level=1)
        pc_a = core.normalize_image(approximation).astype(np.float64) * .2
        gain = core.rms(pc_a) / core.rms(approximation)
        expected = core.normalize_image(np.clip(pywt.waverec2([pc_a, tuple(d * gain for d in detail)],
                                                               "db2")[:64, :66], 0, 1))
        with patch.object(core, "_pc", return_value=pc_a):
            control = core.extract_feature(image, "mfpc_scale_matched")
            original = core.extract_feature(image, "mfpc")
        np.testing.assert_allclose(control, expected, atol=1e-7)
        self.assertFalse(np.allclose(control, original))

    def test_real_phasepack_all_named_strategies_return_finite_correct_shapes(self):
        for strategy in core.STRATEGIES:
            with self.subTest(strategy=strategy):
                result = core.extract_feature(self.image, strategy, seed=9, image_key="test:opt")
                expected = (*self.image.shape, core.PC_DEFAULTS["norient"]) if strategy == "pc_orientation_channels" else self.image.shape
                self.assertEqual(result.shape, expected)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(result)))
                self.assertGreaterEqual(float(result.min()), 0)
                self.assertLessEqual(float(result.max()), 1.0 + 1e-7)
                self.assertGreater(float(np.ptp(result)), .01)
        directional = core.extract_feature(self.image, "pc_orientation_channels")
        norms = np.sqrt((directional ** 2).sum(axis=-1))
        np.testing.assert_allclose(norms[norms > 0], 1, atol=2e-7)

    def test_zero_detail_scale_and_independent_image_noise(self):
        no_detail = core.extract_feature(self.image, "no_detail")
        zero = core.extract_feature(self.image, "mfpc", detail_scale=0)
        np.testing.assert_array_equal(no_detail, zero)
        a = core.extract_feature(self.image, "noise", seed=4, image_key="a")
        same = core.extract_feature(self.image, "noise", seed=4, image_key="a")
        b = core.extract_feature(self.image, "noise", seed=4, image_key="b")
        np.testing.assert_array_equal(a, same)
        self.assertFalse(np.array_equal(a, b))
        with self.assertRaisesRegex(ValueError, "image_key"):
            core.extract_feature(self.image, "noise")

    def test_odd_shape_multilevel_and_constant_image(self):
        image = self.image[:63, :65]
        for strategy in ["mfpc", "no_detail", "gated", "surrogate"]:
            result = core.extract_feature(image, strategy, level=2, image_key="odd:test")
            self.assertEqual(result.shape, image.shape)
        for strategy in core.STRATEGIES:
            result = core.extract_feature(np.ones((64, 66)) * .2, strategy, image_key="constant")
            self.assertTrue(np.all(np.isfinite(result)))
            if strategy != "raw":
                np.testing.assert_array_equal(result, 0)
        with self.assertRaisesRegex(ValueError, "boundary-safe"):
            core.extract_feature(image, "mfpc", level=10)
        with self.assertRaisesRegex(ValueError, "Unknown strategy"):
            core.extract_feature(image, "invented")


if __name__ == "__main__":
    unittest.main(verbosity=2)
