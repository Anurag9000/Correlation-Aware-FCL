"""Build a genuinely runnable sequential job plan; never invent missing methods."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
from .config import RunConfig
from .catalogue import CATALOGUE,IMPLEMENTED,catalogue_report


def base_config(dataset, model='smallcnn', seed=0):
    return RunConfig(dataset=dataset,stages=2 if dataset=='cifar10' else 10,
                     num_clients=10 if dataset=='cifar10' else 20,
                     batch_size=512,model=model,seed=seed)


def _job(base,block,tag,**kw):
    payload=dict(kw)
    payload.update(block_id=block,job_id=f'{block}_{tag}')
    cfg=replace(base,**payload)
    cfg.validate()
    return cfg


def core_suite(dataset,seed=0):
    """60 actual runs on each CIFAR dataset (4 scenarios x 3 x 5)."""
    base=base_config(dataset,seed=seed)
    jobs=[]
    for scenario in ('disjoint','overlap','related','recurrence'):
        for method in ('sgd','gem','agem'):
            for tag,selector,rate in [('full','random',1.),('random50','random',.5),
                                     ('random25','random',.25),('coverage50','coverage',.5),
                                     ('coverage25','coverage',.25)]:
                bid={'disjoint':'A01','overlap':'C02','related':'B02','recurrence':'D07'}[scenario]
                jobs.append(_job(base,bid,f'{scenario}_{method}_{tag}',
                                 scenario=scenario,method=method,selector=selector,participation=rate,
                                 relation='related' if scenario=='related' else 'natural'))
    assert len(jobs)==60
    return jobs


def additional_suite(dataset,seed=0):
    """Concrete supported ablations in addition to the 60-run core."""
    base=base_config(dataset,seed=seed)
    jobs=[]
    def add(block,tag,**kw):jobs.append(_job(base,block,tag,**kw))
    for method in ('sgd','gem'):
        for selector in ('random','fixed','round_robin','least_recent','old','new','overlap','coverage'):
            add('F01' if selector=='random' else {'fixed':'F02','round_robin':'F03',
                'least_recent':'F04','old':'F06','new':'F07','overlap':'F10',
                'coverage':'F05'}[selector],f'{selector}_{method}',selector=selector,method=method,
                scenario='overlap',participation=.5)
    for method in ('sgd','gem','agem','replay','ewc'):
        add({'sgd':'G01','gem':'G02','agem':'G03','replay':'G04','ewc':'G07'}[method],
            f'local_{method}',method=method,scenario='disjoint',participation=.5)
    for sync in (False,True):
        for aggregator in ('fedavg','uniform'):
            add('H01' if not sync else 'H02',f'sync_{sync}_aggregate_{aggregator}',
                full_sync=sync,aggregator=aggregator,scenario='overlap',participation=.5)
    for mode in ('iid','non_iid'):
        for method in ('sgd','agem'):
            add('J01' if mode=='iid' else 'J02',f'{mode}_{method}',scenario=mode,method=method)
    for model in ('linear','mlp','smallcnn'):
        add('M02',f'backbone_{model}',model=model)
    for overlap in (0.,.33,.67,1.):
        add('C01',f'overlap_{int(overlap*100)}',scenario='overlap',overlap=overlap)
    for recurrence_policy in ('same','different'):
        add('D01' if recurrence_policy=='same' else 'D02',f'recurrence_{recurrence_policy}',
            scenario='recurrence',recurrence_client_policy=recurrence_policy)
    for stage in (2,3) if dataset=='cifar100' else (2,):
        for delay in (1,2) if stage==3 else (1,):
            add('D05',f'recurrence_stages{stage}_delay{delay}',scenario='recurrence',
                stages=stage,recurrence_delay=delay)
    for f in (1.,.5,.25):
        for memory in (0,10,20):
            # memory=0 correctly degenerates GEM to SGD.
            add('G12',f'memory{memory}_participation{int(f*100)}',scenario='overlap',
                method='gem',memory_per_class=memory,participation=f)
    for relation in ('natural','related','unrelated'):
        for include in ('include','exclude'):
            add('B02',f'relation{relation}_{include}',scenario='related',relation=relation,
                participation=.5,target_client=2,force_target=include)
    for relation in ('related','unrelated'):
        add('B01',f'class_relationship_{relation}',scenario='related',relation=relation)
    for frac in (1.,.75,.5,.25):
        add('A01',f'participation_{int(frac*100)}',participation=frac)
    for selector in ('related','recurrence'):
        add('N01' if selector=='related' else 'N03',selector,selector=selector,
            scenario='related' if selector=='related' else 'recurrence',participation=.5,
            relation='related' if selector=='related' else 'natural')
    # A simple 2-factor parameter crossing; CIFAR-10 has too few stages for lag 2.
    for part in (.25,.5,1.):
        for scen in ('related','overlap'):
            add('E09',f'{scen}_p{int(part*100)}',scenario=scen,participation=part,
                relation='related' if scen=='related' else 'natural')
    return jobs



def extended_suite(dataset,seed=0):
    """Additional genuinely supported interventions, with independent job IDs.

    The complete 152-block catalogue is still larger. Each block here has
    meaningful software support; no unimplemented method is mapped to SGD.
    """
    base=base_config(dataset,seed=seed)
    jobs=[]
    def add(block,tag,**kw):
        jobs.append(_job(base,block,'extended_'+tag,**kw))

    # A: Participation schedules, timing and repeated participation.
    for schedule in ('constant','increasing','decreasing','alternating'):
        for method in ('sgd','gem'):
            add('A10',f'schedule_{schedule}_{method}',participation=.5,
                participation_schedule=schedule,method=method)
    for selector in ('fixed','round_robin','least_recent'):
        add('A11',f'timing_{selector}',selector=selector,participation=.5)
    for selection in ('old','new','balanced'):
        add('A05',f'old_new_{selection}',selector=selection,scenario='overlap')

    # B and C: Semantic relationships, overlap and local continual training.
    for relation in ('related','unrelated'):
        for method in ('sgd','gem','agem','replay','derpp'):
            add('B11',f'relation_{relation}_local_{method}',scenario='related',
                relation=relation,method=method)
    for overlap in (0.,.5,1.):
        for method in ('sgd','gem','agem'):
            add('C09',f'overlap_{int(overlap*100)}_{method}',
                scenario='overlap',overlap=overlap,method=method)
    for selector in ('diversity','overlap','coverage'):
        add('C08',f'overlap_selector_{selector}',scenario='overlap',selector=selector)

    # D/E: Recurrence interactions across clients and time.
    for policy in ('same','different'):
        for method in ('sgd','gem','agem'):
            add('D12',f'recurrence_{policy}_{method}',scenario='recurrence',
                recurrence_client_policy=policy,method=method)
    for scenario,relation,block in (
        ('mixed','related','E04'),('recurrence','related','E02'),
        ('asynchronous','related','E06'),('overlap','related','E01')):
        for part in (.25,.5,1.):
            add(block,f'{scenario}_{relation}_p{int(part*100)}',
                scenario=scenario,relation=relation,participation=part)
    for aggregator in ('fedavg','uniform','fedadam','fedavgm'):
        add('E10',f'relation_aggregation_{aggregator}',scenario='related',
            relation='related',aggregator=aggregator,
            server_lr=0.001 if aggregator=='fedadam' else 1.)

    # F/N: Client-selection ablations. Metadata selectors are privileged.
    for selector,block in (
        ('balanced','F08'),('diversity','F09'),('fairness','F17'),
        ('stochastic_related','F18'),('historical_coverage','N02')):
        for scenario in ('overlap','related'):
            add(block,f'{selector}_{scenario}',selector=selector,scenario=scenario,
                relation='related' if scenario=='related' else 'natural')

    # G: Independently implemented local objectives (some are documented
    # adaptations rather than original paper reproduction).
    for method,block in (
        ('sgd','G01'),('replay','G04'),('er_ace','G05'),('der','G06'),
        ('derpp','G06'),('mas','G09'),('lwf','G10'),('gem','G02'),
        ('agem','G03'),('ewc','G07')):
        for scenario in ('disjoint','related'):
            add(block,f'local_{method}_{scenario}',method=method,scenario=scenario,
                relation='related' if scenario=='related' else 'natural')
    for mu in (0.,.001,.01):
        for method in ('sgd','gem'):
            add('G15',f'prox_{mu}_{method}',method=method,fedprox_mu=mu)
    for policy in ('random_per_class','recent_per_class'):
        add('G13',f'memory_policy_{policy}',method='replay',memory_policy=policy,
            scenario='recurrence')
    for heterogeneous in (False,True):
        add('G16',f'heterogeneous_{heterogeneous}',client_heterogeneity=heterogeneous,
            scenario='related')

    # H/I: Synchronization, retention and adaptive server optimization.
    for pending in ('latest','retain_all','bounded'):
        for method in ('sgd','gem'):
            add('J08',f'pending_{pending}_{method}',pending_policy=pending,
                max_pending_samples=400 if pending=='bounded' else 0,
                method=method,scenario='asynchronous')
    for epochs in (1,2,4):
        add('H06',f'local_epochs_{epochs}',epochs=epochs,scenario='overlap')
    for aggregation in ('fedavg','uniform','fedavgm','fedadam'):
        for scenario in ('non_iid','related'):
            add('I06',f'aggregate_{aggregation}_{scenario}',aggregator=aggregation,
                server_lr=0.001 if aggregation=='fedadam' else 1.,
                scenario=scenario,relation='related' if scenario=='related' else 'natural')

    # L: Retrospective aggregation counterfactuals on small cohorts only.
    for scenario in ('overlap','related','recurrence'):
        for method in ('sgd','gem'):
            add('L01',f'leaveone_{scenario}_{method}',num_clients=5,
                participation=.6,scenario=scenario,method=method,
                causal_diagnostics=True,causal_max_clients=5,
                relation='related' if scenario=='related' else 'natural')
    for rep in (0,1,2):
        add('M05',f'independent_rep_{rep}',seed=seed+rep,
            scenario='related',relation='related')
    # Independently vary the participating client population and local cost.
    for nclients in (5,10,20):
        add('M03',f'client_population_{nclients}',num_clients=nclients,
            scenario='overlap')
    for epochs in (1,2,4):
        add('M04',f'local_steps_{epochs}',epochs=epochs)
    assert len(jobs)==len({j.job_id for j in jobs})
    return jobs



def causal_fork_suite(dataset,seed=0):
    """Matched multi-round cohort inclusion/exclusion causal trajectories.

    Each baseline saves a checkpoint at the first stream-stage boundary; two
    independent branches then resume from that same *full federated state*.
    The intervention is one selected round (not an all-round selection rule),
    and later client data and memory states evolve independently.
    """
    base=base_config(dataset,seed=seed)
    jobs=[]
    for scenario in ('related','overlap','recurrence'):
        for method in ('sgd','gem'):
            common={'scenario':scenario,'method':method,'participation':.5,
                    'checkpoint_every':1,
                    'relation':'related' if scenario=='related' else 'natural'}
            original=_job(base,'L02',f'forkbaseline_{scenario}_{method}',**common)
            jobs.append(original)
            for arm in ('include','exclude'):
                branch=_job(base,'L02',f'fork{arm}_{scenario}_{method}',
                            **common,force_target=arm,target_client=2,
                            intervention_round=base.rounds_per_stage,
                            fork_source_job=original.job_id,
                            fork_checkpoint_round=base.rounds_per_stage)
                jobs.append(branch)
    assert len(jobs)==18
    return jobs


def all_supported_suite(dataset,seed=0):
    return core_suite(dataset,seed)+additional_suite(dataset,seed)+extended_suite(dataset,seed)+causal_fork_suite(dataset,seed)


def write_plan(path,jobs,dataset):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ids=[j.job_id for j in jobs]
    if len(ids)!=len(set(ids)):raise RuntimeError('Duplicate independent job IDs')
    plan={'dataset':dataset,'batch_size':512,'jobs':[j.to_dict() for j in jobs],
          'catalogue_coverage':catalogue_report(),
          'registered_block_ids':sorted({j.block_id for j in jobs}),
          'coverage_warning':'Unimplemented blocks are NOT represented by fabricated training jobs.'}
    path.write_text(json.dumps(plan,indent=2))
    return plan
