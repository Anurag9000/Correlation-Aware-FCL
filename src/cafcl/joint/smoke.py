"""Short actual CIFAR-100 diagnostic plan (not a synthetic substitute)."""
from .family_catalog import Design
from .curriculum import explicit_sequences


def smoke_designs():
    for overlap in (False,True):
        sequences={0:[(0,1),(1,2) if overlap else (2,3)],
                   1:[(0,1),(1,3) if overlap else (2,3)]}
        for depth in (1,2):
            yield Design(f'SMOKE-O{int(overlap)}-L{depth}','smoke',explicit_sequences(sequences),
                num_clients=2,local_depth=depth,epochs=1,require_all_classes=False,
                metadata={'diagnostic_partial_coverage':True,'one_epoch_per_task':True})
