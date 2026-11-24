"""Non-federated references and no-replay controls on matched atomic data.

Centralized sequential is one learner, not a fictitious multi-client FL
run. Local-only never communicates. Both retain the same training examples
as the corresponding canonical timeline allocation.
"""
from .family_catalog import Design
from .curriculum import foundational_stream


def control_designs():
    for spatial in ('S1','S2','S3'):
        for overlap in (0,1):
            events=foundational_stream(spatial,overlap,steps=100)
            key=f'{spatial}-O{overlap}'
            for depth in (2,5):
                yield Design(f'CTRL-NOREPLAY-{key}-L{depth}','controls',events,
                             local_depth=depth,method='sgd',
                             metadata={'control':'no_replay','matched_atomic_stream':key})
            # Local-only evaluation boundaries, not communication opportunities.
            yield Design(f'CTRL-LOCAL-{key}','controls',events,local_depth=5,
                         cohorts=[()]*20,terminal_flush=False,control_kind='local_only',
                         metadata={'control':'local_only','matched_atomic_stream':key})
            # Chronological client-order serialization of identical occurrences.
            # Reallocation in timeline order gives the same class-specific
            # subsets; source client identity is metadata, not an FL client.
            serial=[(j,0,classes) for j,(_,_,classes) in enumerate(sorted(events))]
            yield Design(f'CTRL-CENTRAL-{key}','controls',serial,num_clients=1,
                         local_depth=25,cohorts=[()]*20,
                         control_kind='centralized_sequential',terminal_flush=True,
                         metadata={'control':'centralized_sequential','matched_atomic_stream':key,
                                   'serialization':'atomic_step_then_source_client'})
