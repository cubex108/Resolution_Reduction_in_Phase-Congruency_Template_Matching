"""Small analytic tests; these do not run the image benchmark."""

import math
import unittest

import numpy as np

from matching_metrics import quantize_fixed, similarity_maps, summarize_map
from summarize_results import (aggregate_pairs, exact_signed_rank, holm_adjust,
                               planned_contrasts, shape_error_correlations)


def reference_scores(x, y, bins):
    """Independent probability-table definition for one fixed-quantized patch."""
    hist = np.zeros((bins, bins), dtype=float)
    for a, b in zip(x.ravel(), y.ravel()):
        hist[int(a), int(b)] += 1
    joint = hist / hist.sum()
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    mi = sum(joint[i, j] * math.log(joint[i, j] / (px[i] * py[j]))
             for i in range(bins) for j in range(bins) if joint[i, j] > 0)
    ccre = 0.0
    for i in range(bins):
        for j in range(bins):
            g = joint[i + 1:, j].sum()
            gx = px[i + 1:].sum()
            if g > 0:
                ccre += g * math.log(g / (gx * py[j])) / bins
    return mi, ccre


class SimilarityTests(unittest.TestCase):
    def test_binary_identity_has_known_scores(self):
        x = np.array([[0., 1.], [0., 1.]])
        out = similarity_maps(x, x, bins=2)
        self.assertAlmostEqual(out['MI'][0, 0], math.log(2), places=12)
        self.assertAlmostEqual(out['CCRE'][0, 0], math.log(2) / 4, places=12)

    def test_independent_balanced_table_is_zero(self):
        x = np.array([[0., 0.], [1., 1.]])
        y = np.array([[0., 1.], [0., 1.]])
        out = similarity_maps(x, y, bins=2)
        self.assertAlmostEqual(out['MI'][0, 0], 0., places=12)
        self.assertAlmostEqual(out['CCRE'][0, 0], 0., places=12)

    def test_ccre_direction_is_not_symmetric(self):
        x = np.array([[0., 0., 0.], [1., 1., 2.]])
        y = np.array([[0., 1., 1.], [0., 2., 2.]])
        xy = similarity_maps(x, y, bins=3)
        yx = similarity_maps(y, x, bins=3)
        self.assertAlmostEqual(xy['MI'][0, 0], yx['MI'][0, 0], places=12)
        self.assertGreater(abs(xy['CCRE'][0, 0] - yx['CCRE'][0, 0]), 1e-3)

    def test_all_candidates_match_fixed_histogram_reference(self):
        rng = np.random.default_rng(18)
        template = rng.normal(size=(3, 4))
        search = rng.normal(size=(7, 8))
        # Deliberate outlier and explicit whole-work-map ranges test fixed bins.
        search[-1, -1] = 30.0
        for bins in (32, 64):
            output = similarity_maps(template, search, step=2, bins=bins,
                                     template_range=(-5, 5), search_range=(-5, 35))
            tx = quantize_fixed(template, bins, (-5, 5))
            sy = quantize_fixed(search, bins, (-5, 35))
            for i, row in enumerate(output['rows']):
                for j, col in enumerate(output['cols']):
                    mi, ccre = reference_scores(tx, sy[row:row+3, col:col+4], bins)
                    self.assertAlmostEqual(output['MI'][i, j], mi, places=11)
                    self.assertAlmostEqual(output['CCRE'][i, j], ccre, places=11)

    def test_quantization_endpoints_clipping_and_constant(self):
        values = np.array([[-1., 0., 0.25, 0.5, 0.75, 1., 2.]])
        np.testing.assert_array_equal(quantize_fixed(values, 4, (0., 1.)),
                                      [[0, 0, 1, 2, 3, 3, 3]])
        np.testing.assert_array_equal(quantize_fixed(np.ones((2, 2)), 32), np.zeros((2, 2)))
        out = similarity_maps(np.ones((2, 2)), np.ones((4, 4)), bins=32)
        self.assertTrue(np.all(out['MI'] == 0) and np.all(out['CCRE'] == 0))

    def test_invalid_inputs_do_not_silently_change_analysis(self):
        with self.assertRaises(ValueError):
            similarity_maps([[np.nan]], [[0.]])
        with self.assertRaises(ValueError):
            similarity_maps(np.ones((3, 3)), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            similarity_maps(np.ones((2, 2)), np.ones((3, 3)), step=0)


class GeometryTests(unittest.TestCase):
    def test_error_uses_pixel_coordinates_not_grid_indices(self):
        scores = np.array([[0., 1., 0.], [0., 2., 0.]])
        out = summarize_map(scores, [0, 8], [0, 16, 32], [7, 14], [48, 48])
        self.assertAlmostEqual(out['err'], math.sqrt(5))
        self.assertTrue(out['success3'])
        self.assertEqual(out['estimated_top_left'], [8., 16.])
        self.assertEqual(out['estimated_center'], [31.5, 39.5])

    def test_baseline_corrected_drop_and_psr_are_affine_invariant(self):
        rows = cols = np.arange(0, 62, 2)
        radius2 = (rows[:, None] - 30) ** 2 + (cols[None, :] - 30) ** 2
        score = np.exp(-radius2 / 80.)
        a = summarize_map(score, rows, cols, [30, 30], [64, 64])
        b = summarize_map(3.4 * score + 8., rows, cols, [30, 30], [64, 64])
        self.assertEqual(a['err'], 0.)
        self.assertEqual(a['ring_samples_available'], 16)
        self.assertIsNotNone(a['baseline_corrected_peak_drop'])
        self.assertAlmostEqual(a['baseline_corrected_peak_drop'], b['baseline_corrected_peak_drop'], places=12)
        self.assertAlmostEqual(a['PSR'], b['PSR'], places=10)

    def test_flat_map_is_not_a_success_and_edge_radius_is_not_clipped(self):
        flat = summarize_map(np.ones((20, 20)), np.arange(20), np.arange(20), [0, 0], [4, 4])
        self.assertTrue(flat['degenerate'])
        self.assertFalse(flat['success3'])
        self.assertIsNone(flat['baseline_corrected_peak_drop'])
        self.assertIsNone(flat['PSR'])
        edge = np.zeros((20, 20)); edge[0, 0] = 1
        summary = summarize_map(edge, np.arange(20), np.arange(20), [0, 0], [4, 4])
        self.assertIsNone(summary['baseline_corrected_peak_drop'])
        self.assertLess(summary['ring_samples_available'], 16)

    def test_threshold_is_strict_three_pixels(self):
        score = np.array([[1., 0.], [0., 0.]])
        out = summarize_map(score, [0, 2], [0, 2], [0, 3], [4, 4])
        self.assertEqual(out['err'], 3.)
        self.assertFalse(out['success3'])


class PairAnalysisTests(unittest.TestCase):
    def test_realisations_do_not_increase_pair_n_or_offset_weight(self):
        rows = []
        # Offset 0 has ten repeats with error zero. Offset 1 has one error ten.
        # The pair's MSE must be 50, not 100/11; n remains one image pair.
        for seed in range(10):
            rows.append(dict(dataset='SAR', pair_id='p1', strategy='raw', measure='CCRE',
                             offset=[0, 0], seed=seed, err=0., success3=True))
        rows.append(dict(dataset='SAR', pair_id='p1', strategy='raw', measure='CCRE',
                         offset=[1, 0], seed=0, err=10., success3=False))
        pairs, offsets = aggregate_pairs(rows, expected_offsets=[[0, 0], [1, 0]])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(offsets), 2)
        self.assertAlmostEqual(pairs[0]['mse'], 50.)
        self.assertAlmostEqual(pairs[0]['rmse'], math.sqrt(50))
        self.assertEqual(pairs[0]['success3'], .5)
        with self.assertRaises(ValueError):
            aggregate_pairs(rows + [rows[0]])

    def test_exact_signed_rank_handles_ties_and_zeros(self):
        self.assertAlmostEqual(exact_signed_rank([1, 2, 3, 4])['p'], .125)
        # Four equal nonzero magnitudes have 16 equally likely sign patterns;
        # all four positive (or negative) gives two-sided p=2/16.
        self.assertAlmostEqual(exact_signed_rank([1, 1, 1, 1, 0])['p'], .125)
        self.assertEqual(exact_signed_rank([0, 0])['p'], 1.)
        self.assertEqual(exact_signed_rank([1, 1, -1, -1])['p'], 1.)
        np.testing.assert_allclose(holm_adjust([.01, .03, .04]), [.03, .06, .06])
        self.assertEqual(holm_adjust([.01, None]), [.02, None])

    def test_missing_pair_and_offset_are_not_silently_imputed(self):
        rows = []
        for dataset in ('SAR', 'Label', 'NIR'):
            for strategy in ('raw', 'full_pc'):
                for pair in range(3):
                    for offset in ([0, 0], [1, 0]):
                        if strategy == 'full_pc' and pair == 2 and offset == [1, 0]:
                            continue
                        rows.append(dict(dataset=dataset, pair_id=str(pair), strategy=strategy,
                                         measure='CCRE', offset=offset, seed=0,
                                         err=4. if strategy == 'raw' else 2., success3=strategy != 'raw'))
        pairs, _ = aggregate_pairs(rows, expected_offsets=[[0, 0], [1, 0]])
        contrasts = planned_contrasts(pairs, n_boot=100)
        self.assertEqual(len(contrasts), 42)
        actual = next(r for r in contrasts if r['dataset'] == 'SAR'
                      and r['strategy_a'] == 'full_pc' and r['strategy_b'] == 'raw')
        self.assertEqual(actual['n_pairs'], 2)
        self.assertEqual(actual['missing_a'], ['2'])
        self.assertEqual(actual['delta_pair_rmse_px'], -2.)
        self.assertEqual(actual['ci95'], [-2., -2.])
        self.assertEqual(actual['family_size'], 21)

    def test_correlation_bootstrap_respects_pair_block_and_missing_shapes(self):
        rows = []
        for pair in range(6):
            for off in ([0, 0], [1, 0]):
                rows.append(dict(dataset='SAR', pair_id=str(pair), strategy='raw', measure='CCRE',
                                 offset=off, seed=0, err=float(pair + 1), success3=pair < 2,
                                 baseline_corrected_peak_drop=None if pair == 5 else float(pair),
                                 PSR=float(pair), map_range=1.))
        pairs, offsets = aggregate_pairs(rows)
        correlations = shape_error_correlations(pairs, offsets, n_boot=100)
        primary = next(r for r in correlations if r['shape_metric'] == 'baseline_corrected_peak_drop'
                       and r['analysis'] == 'pair_mean_shape_vs_pair_rmse')
        self.assertEqual(primary['n_pairs'], 5)
        self.assertEqual(primary['excluded_pair_ids'], ['5'])
        self.assertAlmostEqual(primary['rho'], 1.)
        pooled = next(r for r in correlations if r['shape_metric'] == 'PSR'
                      and r['analysis'] == 'pooled_offsets_pair_cluster_bootstrap')
        self.assertEqual(pooled['n_pairs'], 6)
        self.assertEqual(pooled['n_observations'], 12)
        np.testing.assert_allclose(pooled['ci95'], [1., 1.])


if __name__ == '__main__':
    unittest.main()
