import unittest
import numpy as np
from stress_model import gamma_multiplier, apply_multiplier, perturb
from summarize_stress import aggregate_pairs, summarize_pairs


class StressModelTests(unittest.TestCase):
    def test_unit_mean_gamma_distribution(self):
        # Population tolerances comfortably exceed Monte Carlo standard errors.
        for k in (1,4,16):
            g=gamma_multiplier((1000,1000),k,'statistical-test',0)
            self.assertLess(abs(g.mean()-1),0.005)
            self.assertLess(abs(g.var()-1/k),0.015/k)
            self.assertGreaterEqual(g.min(),0)

    def test_stream_identity_and_independence(self):
        a=gamma_multiplier((16,16),4,'a',7)
        np.testing.assert_array_equal(a,gamma_multiplier((16,16),4,'a',7))
        for pair,k,seed in [('b',4,7),('a',1,7),('a',4,8)]:
            self.assertFalse(np.array_equal(a,gamma_multiplier((16,16),k,pair,seed)))

    def test_clipping_no_realization_rescaling(self):
        b=np.array([[0.,0.25],[0.5,1.]],dtype=np.float32)
        g=np.array([[100.,2.],[3.,.5]])
        result,d=apply_multiplier(b,g)
        np.testing.assert_array_equal(result,[[0.,.5],[1.,.5]])
        self.assertEqual(d['upper_clipped_fraction'],.25)
        self.assertEqual(d['clipped_mass_mean'],.125)
        self.assertEqual(d['pre_clip_mean'],.625)
        self.assertEqual(d['post_clip_mean'],.5)

    def test_unperturbed_identity(self):
        b=np.array([[.1,.2],[.3,.4]],dtype=np.float32)
        out,info=perturb(b,None,'a',0)
        np.testing.assert_array_equal(out,b)
        self.assertEqual(out.dtype,np.float32)
        self.assertEqual(info['multiplier_mean'],1.)
        self.assertEqual(info['multiplier_variance'],0.)
        self.assertEqual(info['upper_clipped_fraction'],0.)

    def test_input_checks(self):
        for k in (0,-1,float('nan')):
            with self.assertRaises(ValueError):gamma_multiplier((2,2),k,'a',0)
        for image,g in [(np.ones((2,2))*2,np.ones((2,2))),
                        (np.zeros((2,2)),np.ones((3,3))),
                        (np.zeros((2,2)),np.ones((2,2))*-1)]:
            with self.assertRaises(ValueError):apply_multiplier(image,g)


class StatisticalUnitTests(unittest.TestCase):
    def fixture(self):
        p=dict(pairs=[dict(pair_id=str(i)) for i in range(3)], offsets=[[0,0],[5,0]],
               strategies=['raw'],gamma_shape_k=[1],realizations=30,bootstrap_seed=7,bootstrap_replicates=200)
        rows=[]
        for pair in range(3):
            for measure in ('MI','CCRE'):
                for offset in p['offsets']:
                    rows.append(dict(pair_id=str(pair),condition='unperturbed',strategy='raw',measure=measure,
                                     offset=offset,seed=None,err=float(pair+1),success3=pair+1<3))
                    for seed in range(30):
                        rows.append(dict(pair_id=str(pair),condition='gamma_k1',strategy='raw',measure=measure,
                                         offset=offset,seed=seed,err=float(pair+2),success3=pair+2<3))
        return rows,p

    def test_seeds_do_not_inflate_pair_count(self):
        rows,p=self.fixture(); pairs=aggregate_pairs(rows,p)
        summaries,differences,paired=summarize_pairs(pairs,p)
        self.assertEqual(len(pairs),12)
        self.assertTrue(all(x['n_pairs']==3 for x in summaries))
        self.assertTrue(all(abs(x['mean_paired_rmse_delta_px']-1)<1e-12 for x in differences))
        self.assertTrue(all(x['mean_paired_rmse_delta_ci95']==[1.,1.] for x in differences))
        self.assertTrue(all(x['delta_pooled_rmse_px']!=x['mean_paired_rmse_delta_px'] for x in differences))

    def test_incomplete_and_duplicates_rejected(self):
        rows,p=self.fixture()
        with self.assertRaises(ValueError):aggregate_pairs(rows[:-1],p)
        with self.assertRaises(ValueError):aggregate_pairs(rows+[rows[0]],p)


if __name__=='__main__':unittest.main()
