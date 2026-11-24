"""Explicit X1-X8 interaction designs and short causal motifs."""
from math import ceil
from .curriculum import foundational_stream, exact_groups, explicit_sequences
from .family_catalog import (Design, migration, sequence_order, oracle_cohorts,
                             outage_stream)


def spatial_width(spatial,width):
    """Generalize adjacent sharing to exact one-class intersection."""
    events=[]
    for t in range(100):
        a=width*t
        for i in range(5):
            offset=(0 if spatial=='S1' else
                    (0 if i<2 else 50) if spatial=='S2' else i*(width-1))
            events.append((t,i,tuple((a+offset+j)%100 for j in range(width))))
    return events


def partial_replication(r):
    # r clients share exactly one anchor class; other clients have disjoint
    # pairs. Rotate the entire block to guarantee coverage.
    span=1+r+2*(5-r)
    events=[]
    for t in range(ceil(100/span)):
        a=t*span
        for i in range(5):
            pair=(a,a+i+1) if i<r else (a+r+1+2*(i-r),a+r+2+2*(i-r))
            events.append((t,i,tuple(c%100 for c in pair)))
    return events


def interaction_designs():
    for s in ('S1','S2','S3'):
        for o in (0,1):
            events=foundational_stream(s,o)
            for l in (1,2,5):
                for n in (1,2,3):
                    yield Design(f'X1-{s}-O{o}-L{l}-N{n}','X1',events,local_depth=l,
                                 offline_clients=n,metadata={'spatial':s,'local_overlap':o})
            for l in (2,5):
                for epochs in (1,2,10):
                    yield Design(f'X2-{s}-O{o}-L{l}-E{epochs}','X2',events,local_depth=l,
                                 epochs=epochs,metadata={'spatial':s,'local_overlap':o})
        for width in (3,5,10):
            for l in (1,2):
                yield Design(f'X3-{s}-W{width}-L{l}','X3',spatial_width(s,width),
                             local_depth=l,metadata={'width':width,'spatial':s})
    for o in (0,1):
        for l in (2,3,5):
            for policy in ('aligned','reversed','cyclic','random'):
                events,ends=sequence_order(policy,l,o,matched_horizon=True)
                yield Design(f'X4-O{o}-L{l}-{policy}','X4',events,local_depth=l,
                    boundaries=ends,allocation_order='client_semantic',
                    metadata={'base_task_pool_overlap':o,'order':policy,
                    'note':'Permutation may change realized adjacent overlap'})
    for partial in (False,True):
        for gap in (0,1,2,5,10,15):
            for l in (1,2):
                yield Design(f'X5-P{int(partial)}-G{gap}-L{l}','X5',
                    migration(gap,partial=partial),local_depth=l,
                    metadata={'partial':partial,'gap':gap})
    for partial in (False,True):
        for r in (2,3,4,5):
            groups=(r,)+(1,)*(5-r)
            events=(partial_replication(r) if partial else
                    exact_groups(groups,width=2,steps=ceil(100/(2*len(groups)))))
            for l in (1,2):
                yield Design(f'X6-P{int(partial)}-R{r}-L{l}','X6',events,local_depth=l,
                             metadata={'partial':partial,'replication':r})
    for duration in (1,2,3,5,10,15):
        for comp in ('none','partial','exact'):
            events,cohorts=outage_stream(comp,duration)
            yield Design(f'X7-{comp}-D{duration}','X7',events,cohorts=cohorts,
                         metadata={'duration':duration,'compensation':comp})
    for s in ('S1','S2','S3'):
        events=foundational_stream(s,0)
        for n in (1,2,3):
            for policy in ('random','redundant','unique'):
                cohorts=None if policy=='random' else oracle_cohorts(events,2,policy,n)
                yield Design(f'X8-{s}-{policy}-N{n}','X8',events,local_depth=2,
                    offline_clients=n if policy=='random' else 0,cohorts=cohorts,
                    metadata={'oracle':policy!='random','omission_policy':policy})


def short_designs():
    motifs={
        'aligned_disjoint':{0:[(0,1),(2,3)],1:[(0,1),(2,3)]},
        'reversed_disjoint':{0:[(0,1),(2,3)],1:[(2,3),(0,1)]},
        'aligned_overlap':{0:[(0,1),(1,2)],1:[(0,1),(1,2)]},
        'partial_disjoint':{0:[(0,1),(2,3)],1:[(1,4),(3,5)]},
        'partial_overlap':{0:[(0,1),(1,2)],1:[(1,3),(3,4)]},
        'aligned_three':{0:[(0,1),(2,3),(4,5)],1:[(0,1),(2,3),(4,5)]},
        'reversed_three':{0:[(0,1),(2,3),(4,5)],1:[(4,5),(2,3),(0,1)]},
        'recurrence':{0:[(0,1),(2,3),(0,1)],1:[(0,1),(2,3),(0,1)]},
        'cyclic_three':{0:[(0,1),(2,3),(4,5)],1:[(2,3),(4,5),(0,1)]},
    }
    for name,seq in motifs.items():
        events=explicit_sequences(seq)
        for depth in (1,max(map(len,seq.values()))):
            yield Design(f'SHORT-{name}-L{depth}','short',events,num_clients=2,
                local_depth=depth,require_all_classes=False,allocation_order='client_semantic',
                metadata={'diagnostic_partial_coverage':True,'motif':name})
    for policy in ('occurrence_specific','client_semantic'):
        yield Design('SHORT-head-'+policy,'short',
            explicit_sequences(motifs['recurrence']),num_clients=2,local_depth=3,
            head_policy=policy,require_all_classes=False,allocation_order='client_semantic',
            metadata={'diagnostic_partial_coverage':True,'head_policy':policy})
