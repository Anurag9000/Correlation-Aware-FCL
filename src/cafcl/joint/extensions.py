"""Additional constructive treatments beyond the initial screening grids."""
from itertools import combinations
from math import ceil
from .family_catalog import Design,coverage_tail,heterogeneous_widths,heterogeneous_depths
from .curriculum import exact_groups,foundational_stream,shared_anchor_stream
from .constraints import solve_overlap,recurrence_gap


def shifted_template(template,num_classes=100):
    span=len({c for labels in template for c in labels})
    return [(t,i,tuple((c+t*span)%num_classes for c in labels))
            for t in range(ceil(num_classes/span)) for i,labels in enumerate(template)]


def extension_designs():
    # Exact pairwise overlap constraints, including feasible high-order sharing.
    pairs=list(combinations(range(5),2))
    for name,widths,edges in [
        ('clique-wide',(5,)*5,{e:2 for e in pairs}),
        ('star-hypergraph',(5,2,2,2,2),{e:int(0 in e) for e in pairs}),
        ('cycle-weighted',(4,)*5,{e:2*int((e[1]-e[0]) in (1,4)) for e in pairs}),
        ('nested-specialists',(6,2,2,2,2),{e:(2 if 0 in e else int(e==(3,4))*2) for e in pairs}),
    ]:
        # Nested case requires one duplicated specialist pair to fit width six.
        if name=='nested-specialists':edges[(3,4)]=2
        template=solve_overlap(widths,edges)
        yield Design('EXT-C-'+name,'C',shifted_template(template),
                     metadata={'requested_widths':widths,'pair_intersections':[
                         [i,j,v] for (i,j),v in edges.items()]})
    # Explicit weighted quantity and class proportion skew, with full raw pool
    # preserved. Do not conflate number of assigned examples with draws.
    events=foundational_stream('S1',0,steps=50)
    for ratio in (2,5,10):
        for mode in ('client_quantity','class_proportion','rotating_quantity'):
            weights={}
            for t,i,labels in events:
                for j,c in enumerate(labels):
                    if mode=='client_quantity':w=ratio if i==0 else 1
                    elif mode=='class_proportion':w=ratio if j==i%2 else 1
                    else:w=ratio if i==t%5 else 1
                    weights[f'{t}:{i}:{c}']=w
            yield Design(f'EXT-P-{mode}-R{ratio}','P',events,local_depth=2,
                         allocation_weights=weights,metadata={'skew':mode,'ratio':ratio})
    for depth in (10,20):
        for spatial in ('S1','S2','S3'):
            for overlap in (0,1):
                yield Design(f'EXT-H-{spatial}-O{overlap}-L{depth}','H',
                    foundational_stream(spatial,overlap),local_depth=depth)
    for width in (3,4,5,10,20):
        for overlap in (0,1):
            for depth in (1,2,5):
                stride=width-overlap
                steps=ceil((100-width)/stride)+1
                events=[(t,i,tuple((t*stride+j)%100 for j in range(width)))
                        for t in range(steps) for i in range(5)]
                yield Design(f'EXT-G-W{width}-O{overlap}-L{depth}','G',events,local_depth=depth)
    for n in (2,3,5,10,20):
        for width in (2,5):
            for depth in (1,2,5):
                yield Design(f'EXT-A-N{n}-W{width}-L{depth}','A',
                    exact_groups((n,),width=width,steps=ceil(100/width)),
                    num_clients=n,local_depth=depth)
    # Deliberately short motif: recurrence of classes in a different client
    # and recombination with new/historical companions.
    for mode,seq in {
        'split':[(0,0,(0,1,2,3)),(0,1,(0,1)),(1,1,(0,4)),(1,2,(2,5))],
        'merge':[(0,0,(0,1)),(0,1,(2,3)),(1,2,(0,1,2,3))],
        'recombine':[(0,0,(0,1)),(0,1,(2,3)),(1,2,(1,2))],
        'new_companion':[(0,0,(0,1)),(0,1,(2,3)),(1,2,(1,4))],
    }.items():
        for depth in (1,2):
            yield Design(f'EXT-E-{mode}-L{depth}','E',coverage_tail(seq,3)[0],
                num_clients=3,local_depth=depth,metadata={'motif':mode,'motif_atomic_steps':2})
