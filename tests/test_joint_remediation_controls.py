from cafcl.joint.curriculum import compile_plan
from cafcl.joint.remediation_controls import build_controls

def test_control_manifest_is_matched(tmp_path):
    targets=[c for c in range(4) for _ in range(40)]
    plan=compile_plan([(0,0,(0,1)),(0,1,(0,1)),(1,0,(2,3)),(1,1,(2,3))],
        targets,num_clients=2,num_classes=4,offline_clients=1,head_policy='semantic_shared').to_dict()
    job={'job_id':'source','learning_setting':'task_il','method':'replay','epochs':1,'seed':0,'plan_hash':plan['plan_hash']}
    jobs,contrasts=build_controls(job,plan,targets,tmp_path,fixed_support=5)
    assert len({j['job_id'] for j in jobs})==len(jobs)
    assert len(contrasts)==len(jobs)-1
    assert all(j['plan_hash']==plan['plan_hash'] for j in jobs if j['metadata']['control']!='CORR-fixed-support')
