"""All-catalog geometry dry run, with no dataset download or GPU training."""
import unittest
from collections import Counter
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
import json
from cafcl.joint.runner import iter_designs,compile_design,settings_for,plan_campaign


class PlannerTests(unittest.TestCase):
    def test_all_catalog_geometry(self):
        labels=[c for c in range(100) for _ in range(450)]
        coarse={c:c//5 for c in range(100)} # synthetic geometry fixture, not real metadata
        keys=set();nominal=Counter();valid=Counter();bad=[]
        designs=list(iter_designs('all',(1,2,3,5),coarse))
        self.assertEqual(len({d.name for d in designs}),len(designs))
        for d in designs:
            settings=settings_for(d);nominal[d.family]+=len(settings)
            try:p=compile_design(d,labels,0)
            except ValueError as exc:
                bad.append((d.name,str(exc),len(settings)));continue
            valid[d.family]+=len(settings)
            for setting in settings:
                keys.add((p['plan_hash'],setting,d.epochs,d.method,d.domain_mode))
        print('JOINT_GEOMETRY_AUDIT='+json.dumps({'nominal':sum(nominal.values()),
            'nominal_by_family':dict(nominal),'valid_before_aliases':sum(valid.values()),
            'unique_geometry_conditions':len(keys),'infeasible':bad},sort_keys=True))
        self.assertEqual(sum(nominal.values()),sum(valid.values())+sum(x[2] for x in bad))
        self.assertEqual(nominal['H'],72) # 48 foundational + 24 depth extensions
        self.assertEqual(nominal['intratask'],48)
        self.assertGreater(len(keys),800)

    def test_manifest_shares_plan_files(self):
        bank=SimpleNamespace(train=SimpleNamespace(targets=[c for c in range(100) for _ in range(450)]),
                             coarse_by_fine={c:c//5 for c in range(100)})
        with TemporaryDirectory() as tmp,patch('cafcl.data.get_data_bank',return_value=bank):
            jobs=plan_campaign(tmp,'data',0,False,suite='smoke')
            manifest=json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual(len(jobs),8)
            self.assertEqual(manifest['nominal_conditions'],8)
            self.assertEqual(len(list((Path(tmp)/'plans').glob('*.json'))),4)
