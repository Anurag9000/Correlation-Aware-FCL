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


def research_closure_suite(dataset,seed=0):
    """Additional real, independently stored CIFAR-10/CIFAR-100 interventions.

    Each job maps to a *specific implemented mechanism*. Unsupported claims
    remain excluded from the executable plan and the coverage ledger. The
    CIFAR-100 plan also contains the exact original 20-class/10+10 design.
    """
    base=base_config(dataset,seed=seed)
    jobs=[]
    def add(block,tag,*,base_override=None,**kw):
        b=base if base_override is None else base_override
        jobs.append(_job(b,block,'closure_'+tag,**kw))
    # A: fixed budget, comparable rounds, total updates, and stage schedules.
    for rate in (1.,.5):
        add('A03',f'roundmatched_{int(rate*100)}',participation=rate)
    for tag,rate,rounds in (('full',1.,10),('half',.5,20)):
        add('A04',f'clientupdates_{tag}',participation=rate,rounds_per_stage=rounds)
    for tag,rates in (('earlyfull',(1.,.25)),('latefull',(.25,1.))):
        b=replace(base,stages=2)
        add('A06',tag,base_override=b,participation_schedule='stage',stage_participation=rates)
    for selector in ('coverage','fixed'):
        add('A07',f'diversity_{selector}',scenario='overlap',selector=selector,participation=.5)
    for selector in ('fixed','round_robin'):
        add('A09',f'frequency_{selector}',scenario='overlap',selector=selector,participation=.25)

    # B: exact 10+10 CIFAR-100 reproduction. In CIFAR-10 the analogous
    # semantic pairs are automobile/truck (1/9) and cat/dog (3/5).
    if dataset=='cifar100':
        b=replace(base,dataset='cifar100_20',num_clients=10,stages=2,
                  scenario='disjoint',relation='natural',rounds_per_stage=20)
        holders={'0':(0,),'1':(0,),'12':(3,),'13':(3,)}
        new_pair=(12,13)
        old_pair=(0,1)
    else:
        b=replace(base,num_clients=10,stages=2,
                  scenario='disjoint',relation='natural',rounds_per_stage=20)
        holders={'1':(0,),'3':(0,),'5':(3,),'9':(3,)}
        new_pair=(5,9)
        old_pair=(1,3)
    b=replace(b,holders_override=holders)
    for arm in ('include','exclude'):
        add('B05',f'exact_old{old_pair}_new{new_pair}_{arm}',base_override=b,
            target_client=3,force_target=arm,intervention_round=20)
        add('B12',f'class_spillover_{arm}',base_override=b,prepost_diagnostics=True,
            target_client=3,force_target=arm,intervention_round=20)
    for dose in (.25,.5,1.):
        add('B07',f'related_dose_{int(dose*100)}',base_override=b,
            class_sample_fraction={str(c):dose for c in new_pair})
    for tag,when in (('once',20),('always',-1)):
        add('B08',f'related_participation_{tag}',base_override=b,
            target_client=3,force_target='include',intervention_round=when)
    for arm,cohort in (('related',(0,1,2,3,4)),('unrelated',(0,1,2,4,5))):
        add('B09',f'mixed_related_{arm}',base_override=b,
            participation_overrides={'20':cohort})

    # C: exact label/sample overlap; client-specific domain; update conflict.
    for tag,weights in (('balanced',(.5,.5)),('dominant',(.9,.1))):
        add('C03',tag,scenario='overlap',holders_override={'0':(0,1)},
            class_holder_weights={'0':weights})
    add('C04','same_label_disjoint_images',scenario='overlap',overlap=1.,
        sample_overlap_fraction=0.)
    for fraction in (0.,.25,.5):
        add('C05',f'raw_sample_overlap_{int(fraction*100)}',scenario='overlap',
            overlap=1.,sample_overlap_fraction=fraction)
    add('C06','client_domain_shared_class',scenario='domain',stages=2,num_clients=5,
        learning_setting='domain_il')
    add('C07','overlapping_client_update_cosines',scenario='overlap',
        overlap=1.,gradient_diagnostics=True)
    for selector in ('fixed','round_robin'):
        add('C10',f'longitudinal_{selector}',scenario='overlap',overlap=1.,selector=selector)

    # D/E: recurrence *round* delay, periodic return, inactive holding clients.
    for lag in (1,5,15):
        add('D06',f'lag_{lag}_rounds',scenario='recurrence',stages=2,
            recurrence_round_delay=lag)
    for period in (5,10):
        add('D08',f'periodic_{period}',scenario='recurrence',stages=2,
            recurrence_round_delay=3,recurrence_period=period,recurrence_repeats=3)
    for method in ('sgd','gem'):
        add('D09',f'mixed_old_new_{method}',scenario='mixed',method=method)
    for arm in ('include','exclude'):
        add('D10',f'inactive_recurrence_{arm}',scenario='recurrence',stages=2,
            force_target=arm,target_client=3,intervention_round=20)
    skipped={str(r):(0,1,2,4,5) for r in range(20,25)}
    immediate={'20':(0,1,2,3,4)}
    delayed=dict(skipped);delayed['25']=(0,1,2,3,4)
    for tag,overrides in (('immediate',immediate),('delayed',delayed)):
        add('D11',f'recurrence_participation_{tag}',scenario='recurrence',stages=2,
            num_clients=10,participation=.5,participation_overrides=overrides)
    for method in ('sgd','gem'):
        add('E03',f'overlap_recurrence_{method}',scenario='mixed',method=method)
    for selector in ('gradient_align','gradient_conflict'):
        costbase=b if dataset=='cifar100' else base
        add('E05',f'helpful_harmful_{selector}',base_override=costbase,
            selector=selector,gradient_diagnostics=True)
    for dose in (.1,.5):
        add('E07',f'rare_recurrence_{int(dose*100)}',scenario='recurrence',stages=2,
            class_sample_fraction={'0':dose})

    # F: model-dependent train-only oracle score controls.
    for selector,block in [('feature_max','F11'),('feature_min','F12'),
                           ('gradient_align','F13'),('gradient_conflict','F14'),
                           ('forgetting','F15'),('history_adaptive','F19')]:
        costly=b if dataset=='cifar100' and selector.startswith(('feature','gradient')) else base
        add(block,selector,base_override=costly,selector=selector,
            gradient_diagnostics=selector.startswith('gradient'))
    add('F20','exhaustive_subsets',num_clients=4,participation=.75,
        causal_max_clients=4,subset_diagnostics=True,rounds_per_stage=2,model='linear')

    # G: SI and GEM reference grouping, historical-memory controls.
    for method in ('sgd','si'):
        add('G08',f'synaptic_intelligence_{method}',method=method)
    for grouping in ('class','stage','all'):
        add('G14',f'gem_reference_{grouping}',method='gem',gem_grouping=grouping)
    for method in ('sgd','replay'):
        add('G11',f'memory_comparison_{method}',method=method)

    # H: optimizer state, long dropout and rejoining.
    for policy in ('reset','persist'):
        add('H04',f'momentum_{policy}',optimizer_state_policy=policy,momentum=.9)
    for tag,overrides in (('included',{'20':(0,1,2,3,4)}),
                          ('excluded',{str(r):(0,1,2,4,5) for r in range(20,28)})):
        add('H07',f'rejoin_{tag}',num_clients=10,participation=.5,
            participation_overrides=overrides)

    # I: distinct server aggregation mechanisms.
    for aggregator,block in [('fedavg','I03'),('class_balanced','I03'),
                              ('fedavg','I04'),('conflict_aware','I04'),
                              ('fedavg','I05'),('retention_aware','I05'),
                              ('fedavg','I07'),('conflict_projection','I07'),
                              ('fedavg','I08'),('inverse_participation','I08')]:
        add(block,f'aggregate_{aggregator}_{block}',scenario='overlap',
            aggregator=aggregator)

    # J: client data quantity, domain drift, time and class imbalance.
    for skew in (0.,3.):
        add('J03',f'client_sample_skew_{skew}',scenario='overlap',client_sample_skew=skew)
    for stage in (2,3):
        add('J04',f'domain_feature_skew_{stage}',scenario='domain',stages=stage,
            num_clients=5,learning_setting='domain_il')
    for power in (0.,1.):
        add('J05',f'long_tail_{power}',scenario='overlap',class_imbalance_power=power)
    for tag,lags in (('independent',{}),('dependent',{'0':2,'1':2,'2':2})):
        add('J06',f'correlated_arrivals_{tag}',scenario='asynchronous',
            arrival_lag_by_client=lags)
    for lag in (0,5):
        add('J07',f'client_lag_{lag}',scenario='asynchronous',
            arrival_lag_by_client={'0':lag,'1':lag})
    for period in (0,5,10):
        add('J09',f'recurrence_period_{period}',scenario='recurrence',stages=2,
            recurrence_round_delay=3,recurrence_period=period,
            recurrence_repeats=1 if not period else 3)
    for mode in ('domain','gradual_drift'):
        add('J12',f'temporal_distribution_{mode}',scenario=mode,stages=2,
            num_clients=5,learning_setting='domain_il')

    # K: task-ID-aware, domain-ID-assessed, and boundary-free controls.
    for method in ('sgd','gem'):
        add('K02',f'task_il_{method}',scenario='disjoint',stages=2,
            learning_setting='task_il',method=method)
    for scenario in ('domain','gradual_drift'):
        add('K03',f'domain_il_{scenario}',scenario=scenario,stages=2,
            num_clients=5,learning_setting='domain_il')
    for method in ('gem','agem'):
        add('K04',f'boundary_free_{method}',method=method,boundary_mode='hidden',
            learning_setting='boundary_free')
    for mode in ('known','hidden'):
        add('K06',f'experience_boundary_{mode}',method='gem',boundary_mode=mode,
            learning_setting='boundary_free' if mode=='hidden' else 'class_il')
    for scenario in ('disjoint','asynchronous'):
        add('K07',f'transitions_{scenario}',scenario=scenario)

    # L: local/global loss before/after, subset enumeration and representation.
    for block in ('L03','L04'):
        add(block,f'counterfactual_{block}',num_clients=4,participation=.75,
            causal_max_clients=4,subset_diagnostics=True,rounds_per_stage=2,model='linear')
    for block in ('L05','L06'):
        add(block,f'historical_losses_{block}',prepost_diagnostics=True,
            scenario='related',relation='related')
    for block in ('L07','L10'):
        add(block,f'gradient_interference_{block}',gradient_diagnostics=True,
            prepost_diagnostics=True,scenario='overlap')
    for scenario in ('disjoint','related'):
        add('L08',f'representation_drift_{scenario}',scenario=scenario,
            representation_diagnostics=True)

    # M/N: resource measurement and exploratory mechanism policies.
    for block in ('M06','M07'):
        for selector in ('fixed','round_robin'):
            add(block,f'resource_{selector}',selector=selector)
    for method in ('gem','agem'):
        add('N04',f'gem_informed_{method}',selector='gem_informed',method=method)

    if len(jobs)!=len({j.job_id for j in jobs}):
        raise RuntimeError('Duplicate closure job identifiers')
    for job in jobs:job.validate()
    return jobs


_old_all_supported_suite=all_supported_suite

def all_supported_suite(dataset,seed=0):
    """Every executable suite job, not every unimplemented catalogue block."""
    return _old_all_supported_suite(dataset,seed)+research_closure_suite(dataset,seed)


def additional_research_controls(dataset,seed=0):
    """Explicit controls for catalogue blocks previously represented only by
    implicit behavior. These remain partial where causal matching or a
    publication-level benchmark is still absent.
    """
    base=base_config(dataset,seed=seed)
    jobs=[]
    def add(block,tag,**kw):jobs.append(_job(base,block,'controls_'+tag,**kw))
    for selector in ('random','fixed'):
        add('A02',f'fixed_resampled_{selector}',selector=selector,participation=.5)
    for selector in ('coverage','overlap'):
        add('A08',f'redundancy_{selector}',selector=selector,scenario='overlap',participation=.5)
    for relation in ('related','unrelated'):
        add('B03',f'coarse_relationship_{relation}',scenario='related',relation=relation,
            gradient_diagnostics=True)
        add('B04',f'gradient_relationship_{relation}',scenario='related',relation=relation,
            gradient_diagnostics=True)
        add('L09',f'permuted_class_arrival_{relation}',scenario='related',relation=relation)
    for scenario in ('related','recurrence'):
        add('D03',f'related_new_labels_{scenario}',scenario=scenario,relation='related')
    for shift in ('recurrence','shifted_recurrence'):
        add('D04',f'domain_shift_{shift}',scenario=shift,stages=2)
    for tag,lags in (('persistent',{}),('temporary',{'0':10,'1':10})):
        add('E08',f'client_correlation_{tag}',scenario='asynchronous',
            arrival_lag_by_client=lags)
    for selector in ('random','recurrence'):
        add('F16',f'recurrence_aware_{selector}',scenario='recurrence',selector=selector)
    for full_sync in (False,True):
        add('H03',f'rejoin_fullsync_{full_sync}',full_sync=full_sync,
            scenario='asynchronous')
    for clear in (False,True):
        add('H05',f'private_memory_reset_{clear}',method='gem',
            reset_memory_on_sync=clear)
    for agg,block in (('fedavg','I01'),('uniform','I02')):
        add(block,f'standalone_aggregation_{agg}',aggregator=agg,scenario='overlap')
    for strength in (0.,1.):
        add('J10',f'spurious_patch_{strength}',spurious_strength=strength,
            spurious_ood_eval=bool(strength))
    for rate in (0.,.2):
        add('J11',f'client_label_noise_{rate}',label_noise_rate=rate)
    add('K01','class_incremental_reference',scenario='disjoint',learning_setting='class_il')
    add('M01',f'dataset_generalization_{dataset}',scenario='related',relation='related')
    for feedback in ('constant','feedback'):
        add('N05',f'feedback_participation_{feedback}',scenario='overlap',method='gem',
            participation_schedule=feedback,participation=.5)
    for job in jobs:job.validate()
    if len(jobs)!=len({j.job_id for j in jobs}):raise RuntimeError('Duplicate control job IDs')
    return jobs


_previous_supported_suite=all_supported_suite

def all_supported_suite(dataset,seed=0):
    return _previous_supported_suite(dataset,seed)+additional_research_controls(dataset,seed)
