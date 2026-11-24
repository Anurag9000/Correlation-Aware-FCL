"""Epoch-sliced joint FCL, preserving parent task/head/sample identity."""
from .family_catalog import Design
from .curriculum import foundational_stream


def intratask_designs():
    for spatial in ('S1','S2','S3'):
        for overlap in (0,1):
            events=foundational_stream(spatial,overlap)
            for segments in (1,2,5,10):
                yield Design(f'INTRA-{spatial}-O{overlap}-S{segments}','intratask',events,
                    local_depth=1,epochs=10//segments,epoch_segments=segments,
                    metadata={'epochs_per_parent_task':10,'synchronizations_per_task':segments,
                              'replay_consolidation':'parent_task_end',
                              'current_shuffle':'parent_experience_and_epoch_offset'})
