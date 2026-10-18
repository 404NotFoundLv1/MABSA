import importlib.util
import unittest

import torch

from mact.crf import ConstrainedCRF
from mact.labels import aspects_to_tags


class CalibratedFusionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('mact.calibrated_fusion'),
                             'calibrated fusion module missing')
        from mact import calibrated_fusion
        self.f = calibrated_fusion
        torch.set_num_threads(1)

    def test_conditional_logits_equal_exhaustive_path_score_differences(self):
        # Missing either transition, wrong polarity order, or endpoint handling fails.
        crf = ConstrainedCRF()
        with torch.no_grad():
            crf.transitions.copy_(torch.arange(25).reshape(5,5) / 13.)
            crf.start_transitions.copy_(torch.tensor([0., .8, -.4, .3, 0.]))
            crf.end_transitions.copy_(torch.tensor([.2, -.3, .7, .9, .1]))
        emissions = torch.tensor([[.1,.5,-.2,.8,.3],[.6,-.4,.9,.2,-.2],
                                  [.8,.7,-.1,.4,.6],[.5,-.2,.3,.1,.9]])
        for aspects in [[(0,1,2),(1,3,0),(3,4,1)], [(1,2,1)], []]:
            got = self.f.aspect_polarity_logits(crf, emissions, aspects)
            self.assertEqual(got.shape, (len(aspects),3))
            for i, (start,end,_) in enumerate(aspects):
                scores = []
                for p in range(3):
                    variant = list(aspects); variant[i] = (start,end,p)
                    tags = torch.tensor(aspects_to_tags(4,variant))
                    score = crf.start_transitions[tags[0]] + crf.end_transitions[tags[-1]]
                    score += emissions[torch.arange(4),tags].sum()
                    score += crf.transitions[tags[:-1],tags[1:]].sum()
                    scores.append(score)
                expected = torch.stack(scores)
                torch.testing.assert_close(got[i]-got[i,0], expected-expected[0])

    def test_temperature_recovers_known_optimum_and_does_not_change_argmax(self):
        logits = torch.tensor([[8.,1.,1.]]).log().repeat(8,1)
        labels = torch.tensor([0,0,0,0,1,1,2,2])
        t = self.f.fit_temperature(logits,labels)
        self.assertAlmostEqual(t,3.,places=3)
        torch.testing.assert_close((logits/t).argmax(-1),logits.argmax(-1))

    def test_mixture_uses_both_experts_and_preserves_endpoints_and_ties(self):
        b = torch.tensor([[.8,.1,.1],[.1,.8,.1],[1.,1.,1.]]).log()
        t = torch.tensor([[.05,.9,.05],[.1,.2,.7],[1.,1.,1.]]).log()
        base, target = torch.tensor([0,1,2]), torch.tensor([1,2,0])
        for a, expected in [(0.,[0,1,2]),(.25,[0,1,2]),(.75,[1,2,2]),(1.,[1,2,0])]:
            self.assertEqual(self.f.fuse_labels(b,t,base,target,1.,1.,a).tolist(),expected)

    def test_rejects_nonfinite_scores_bad_temperatures_and_bad_labels(self):
        z, labels = torch.ones(2,3),torch.tensor([0,2])
        for temp, alpha in [(0.,.5),(float('nan'),.5),(1.,-1.),(1.,1.1)]:
            with self.assertRaises(ValueError):
                self.f.fuse_labels(z,z,labels,labels,temp,1.,alpha)
        with self.assertRaises(ValueError):
            self.f.fit_temperature(z,torch.tensor([0,-1]))
        with self.assertRaises(ValueError):
            self.f.fit_temperature(torch.full((2,3),float('nan')),labels)
        with self.assertRaises(ValueError):
            self.f.aspect_polarity_logits(ConstrainedCRF('seven'),torch.ones(2,7),[(0,1,0)])

    def test_empty_decisions_and_empty_calibration(self):
        z, labels = torch.empty(0,3),torch.empty(0,dtype=torch.long)
        self.assertEqual(self.f.fuse_labels(z,z,labels,labels,1.,1.,.5).numel(),0)
        with self.assertRaises(ValueError):
            self.f.fit_temperature(z,labels)
