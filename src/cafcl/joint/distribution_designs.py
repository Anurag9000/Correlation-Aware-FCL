"""Distribution controls using genuine CIFAR-100 superclass membership."""
from collections import defaultdict
from .family_catalog import Design
from .curriculum import foundational_stream


def semantic_stream(coarse_by_fine,related):
    groups=defaultdict(list)
    for fine,coarse in sorted(coarse_by_fine.items()):
        groups[int(coarse)].append(int(fine))
    sets=[sorted(v) for _,v in sorted(groups.items())]
    if len(sets)!=20 or any(len(g)!=5 for g in sets) or {c for g in sets for c in g}!=set(range(100)):
        raise ValueError('Semantic experiment requires actual complete CIFAR-100 20x5 superclass map')
    events=[]
    for t,group in enumerate(sets):
        for i in range(4):
            private=group[i+1] if related else sets[(t+i+1)%20][i+1]
            events.append((t,i,(group[0],private)))
    return events


def distribution_designs(coarse_by_fine):
    for spatial in ('S1','S3'):
        events=foundational_stream(spatial,0)
        for mode in ('client_fixed','client_periodic','client_corruption'):
            yield Design(f'EXT-P-{spatial}-{mode}','P',events,local_depth=2,domain_mode=mode,
                         metadata={'domain_mode':mode,'replay_domain':'receiver_active_domain'})
    for related in (False,True):
        yield Design('EXT-P-SEMANTIC-'+('related' if related else 'unrelated'),'P',
            semantic_stream(coarse_by_fine,related),num_clients=4,local_depth=2,
            metadata={'semantic_relation':'same_superclass' if related else 'different_superclass',
                      'matched_class_occurrence_counts':True})
