import unittest
from cafcl.joint.curriculum import compile_plan
from cafcl.joint.comparison import planned_budget,match_status,aggregate,default_contrasts


class ComparisonTests(unittest.TestCase):
    def test_parent_matching_and_epoch_budget(self):
        labels=[c for c in range(4) for _ in range(40)]
        events=[(0,0,(0,1)),(0,1,(0,1)),(1,0,(2,3)),(1,1,(2,3))]
        a=compile_plan(events,labels,num_clients=2,num_classes=4,head_policy='semantic_shared').to_dict()
        b=compile_plan(events,labels,num_clients=2,num_classes=4,head_policy='semantic_shared',epoch_segments=2).to_dict()
        x=planned_budget(a,10);y=planned_budget(b,5)
        self.assertTrue(match_status(x,y,'atomic_data_and_epochs'))
        self.assertEqual(x['planned_fresh_draws'],y['planned_fresh_draws'])
        self.assertNotEqual(x['communication_opportunities'],y['communication_opportunities'])
        self.assertFalse(match_status(x,planned_budget(b,4),'atomic_data_and_epochs'))

    def test_seed_uncertainty_and_contrast_ids(self):
        self.assertIsNone(aggregate([.1])['ci95_low'])
        result=aggregate([.1,.2,.3])
        self.assertAlmostEqual(result['mean'],.2)
        self.assertLess(result['ci95_low'],.2)
        self.assertEqual(len({x['contrast_id'] for x in default_contrasts()}),len(default_contrasts()))
