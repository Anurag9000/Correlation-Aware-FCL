"""Fixed, intentionally small production campaign. No seed sweeps."""
from .family_catalog import Design, screening_designs
from .curriculum import foundational_stream
from .controls import control_designs
from .smoke import smoke_designs

SCREENING = ('D-G5','E-G5','N-S3-random-1','N-S3-unique-1')
CONTROLS = ('CTRL-NOREPLAY-S3-O1-L5','CTRL-POOLED-S3-O1-L5',
            'CTRL-LOCAL-S3-O1','CTRL-CENTRAL-S3-O1')


def essential_designs():
    yield from smoke_designs()
    for s in ('S1','S2','S3'):
        for o in (0,1):
            for depth in (1,2,5):
                yield Design(f'H-{s}-O{o}-L{depth}','H',
                             foundational_stream(s,o),local_depth=depth)
    selected={d.name:d for d in screening_designs() if d.name in SCREENING}
    if set(selected)!=set(SCREENING):raise RuntimeError('Missing essential screening arm')
    for name in SCREENING:yield selected[name]
    selected={d.name:d for d in control_designs() if d.name in CONTROLS}
    if set(selected)!=set(CONTROLS):raise RuntimeError('Missing essential control arm')
    for name in CONTROLS:yield selected[name]
