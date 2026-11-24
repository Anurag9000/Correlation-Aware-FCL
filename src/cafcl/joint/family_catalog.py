"""Constructive, joint-only screening designs.

These are explicit initial screens, not the full factorial product. Every
design is compiled against real TRAIN targets and may be marked infeasible.
Treatment motifs and any shared coverage tail are identified in the metadata.
"""
from dataclasses import dataclass, field
from math import ceil
from collections import defaultdict, Counter
import random
from .curriculum import (exact_groups, shared_anchor_stream, foundational_stream,
                         explicit_sequences, stable_seed)


@dataclass
class Design:
    name: str
    family: str
    events: list
    num_clients: int = 5
    local_depth: int = 1
    metadata: dict = field(default_factory=dict)
    boundaries: list | None = None
    cohorts: list | None = None
    offline_clients: int = 0
    outage_duration: int = 1
    allocation_order: str = 'timeline'
    epochs: int = 5
    require_all_classes: bool = True
    head_policy: str = 'semantic_shared'


def coverage_tail(events,k):
    """Append declared shared-task coverage if the motif omits any labels."""
    events=list(events)
    seen={c for _,_,labels in events for c in labels}
    missing=sorted(set(range(100))-seen)
    start=max(t for t,_,_ in events)+1
    for j in range(0,len(missing),2):
        pair=tuple(missing[j:j+2])
        if len(pair)==1:
            pair=tuple(sorted((pair[0],min(seen))))
        events.extend((start+j//2,i,pair) for i in range(k))
    return events,start


def topology_stream(edges,width,steps=None,k=5):
    edges={tuple(sorted(e)) for e in edges}
    labels=[[] for _ in range(k)]
    for c,(i,j) in enumerate(sorted(edges)):
        labels[i].append(c);labels[j].append(c)
    cursor=len(edges)
    for i in range(k):
        if len(labels[i])>width:raise ValueError('Width below graph degree')
        while len(labels[i])<width:
            labels[i].append(cursor);cursor+=1
    if cursor>100:raise ValueError('Graph needs more than 100 classes')
    steps=ceil(100/cursor) if steps is None else steps
    return [(t,i,tuple((c+t*cursor)%100 for c in labels[i]))
            for t in range(steps) for i in range(k)]


def migration(gap, *, same_client=False, partial=False, visits=2):
    """Fixed 18-step motif, swap destination slots to hold task multiset fixed."""
    if not 0<=gap<=15 or not 2<=visits<=5:
        raise ValueError('Migration treatment outside screening range')
    grid={(t,i):(2+2*((t*5+i)%49),3+2*((t*5+i)%49))
          for t in range(18) for i in range(5)}
    grid[0,0]=(0,1)
    dest=0 if same_client else 1
    grid[1,dest]=(1,2) if partial else (0,1)
    target=gap+1
    grid[1,dest],grid[target,dest]=grid[target,dest],grid[1,dest]
    for j in range(2,visits):
        grid[j+1,j]=(1,2) if partial else (0,1)
    events=[(t,i,labels) for (t,i),labels in sorted(grid.items())]
    return coverage_tail(events,5)[0]


def sequence_order(policy,depth,local_overlap=0,matched_horizon=False):
    """Same per-client semantic multiset; communication at block ends."""
    events=[];boundaries=[]
    pairs=([(c,(c+1)%100) for c in range(100)] if local_overlap else
           [(c,c+1) for c in range(0,100,2)]*(2 if matched_horizon else 1))
    t=0
    for first in range(0,len(pairs),depth):
        block=pairs[first:first+depth]
        for i in range(5):
            seq=list(block)
            if policy=='reversed' and i%2:seq.reverse()
            elif policy=='cyclic':
                shift=i%len(seq);seq=seq[shift:]+seq[:shift]
            elif policy=='random':
                random.Random(stable_seed('order',first,i)).shuffle(seq)
            elif policy=='common_first' and len(seq)>1 and i%2:
                seq=seq[:1]+seq[1:][::-1]
            elif policy=='common_last' and len(seq)>1 and i%2:
                seq=seq[:-1][::-1]+seq[-1:]
            events.extend((t+j,i,pair) for j,pair in enumerate(seq))
        t+=len(block);boundaries.append(t)
    return events,boundaries


def replication_stream(pattern):
    if pattern=='uniform':multiplicities=[3]*100
    elif pattern=='binary':multiplicities=[5 if c%2==0 else 1 for c in range(100)]
    elif pattern=='long_tail':multiplicities=[5 if c<10 else 3 if c<30 else 1 for c in range(100)]
    elif pattern=='rotating':multiplicities=[1+c%5 for c in range(100)]
    else:raise ValueError(pattern)
    labels=defaultdict(list)
    for c,m in enumerate(multiplicities):
        for j in range(m):labels[(c+j)%5].append(c)
    seq={}
    for i,classes in labels.items():
        seq[i]=[tuple(classes[j:j+2]) if j+1<len(classes)
                else (classes[j],classes[0]) for j in range(0,len(classes),2)]
    return explicit_sequences(seq)


def heterogeneous_widths(widths):
    span=1+sum(w-1 for w in widths)
    if span>100:raise ValueError('Too many private labels')
    events=[]
    for t in range(ceil(100/span)):
        cursor=1
        for i,w in enumerate(widths):
            labels=[(t*span)%100]+[(t*span+j)%100 for j in range(cursor,cursor+w-1)]
            events.append((t,i,tuple(labels)));cursor+=w-1
    return events


def heterogeneous_depths(depths):
    events=[];boundaries=[];t=0
    while t<50:
        block=min(max(depths),50-t)
        for i,depth in enumerate(depths):
            events.extend((t+j,i,(2*(t+j),2*(t+j)+1)) for j in range(min(depth,block)))
        t+=block;boundaries.append(t)
    return events,boundaries


def oracle_cohorts(events,depth,policy,offline,k=5):
    """Curriculum-only oracle, not a deployable client selector."""
    if not 0<offline<k:raise ValueError('Invalid omission count')
    horizon=max(t for t,_,_ in events)+1
    prior=set();cohorts=[]
    for start in range(0,horizon,depth):
        sets={i:set() for i in range(k)}
        for t,i,labels in events:
            if start<=t<start+depth:sets[i].update(labels)
        owners=Counter(c for labels in sets.values() for c in labels)
        def score(i):
            labels=sets[i]
            if policy=='redundant':return sum(owners[c]>1 for c in labels)
            if policy=='unique':return sum(owners[c]==1 for c in labels)
            if policy=='novel':return len(labels-prior)
            if policy=='historical':return len(labels&prior)
            raise ValueError(policy)
        omitted=set(sorted(range(k),key=lambda i:(-score(i),i))[:offline])
        cohorts.append(tuple(i for i in range(k) if i not in omitted))
        prior.update(c for labels in sets.values() for c in labels)
    return cohorts


def outage_stream(compensation,duration):
    events=[]
    # Matched 50 atomic steps. During the target outage the hidden client's
    # task is fully, partially or not represented among online tasks.
    for t in range(50):
        a=2*t
        for i in range(5):
            if i==0 or t>=duration or compensation=='exact':
                pair=(a,a+1)
            elif compensation=='partial':
                pair=(a,(a+2)%100)
            else:
                pair=((a+2)%100,(a+3)%100)
            events.append((t,i,pair))
    cohorts=[tuple(range(1,5)) if t<duration else tuple(range(5)) for t in range(50)]
    return events,cohorts


def dynamic_stream(policy):
    patterns={
        'increasing':['S3','S2','S1'],
        'decreasing':['S1','S2','S3'],
        'periodic':['S1','S3','S1','S3'],
        'abrupt':['S3','S1'],
    }[policy]
    events=[]
    for t in range(50):
        spatial=patterns[min(len(patterns)-1,t*len(patterns)//50)]
        a=2*t
        for i in range(5):
            offset=0 if spatial=='S1' else (0 if i<2 else 50) if spatial=='S2' else i
            events.append((t,i,((a+offset)%100,(a+offset+1)%100)))
    return coverage_tail(events,5)[0]


def novelty_stream(historical):
    """Target historical fraction in the later exposure phase, not accuracy."""
    if not 0<=historical<=5:raise ValueError(historical)
    events=[]
    # Shared, fixed historical foundation on labels 0..49.
    for t in range(10):
        events.extend((t,i,tuple(range(5*t,5*t+5))) for i in range(5))
    cursor=50;t=10
    while cursor<100 and historical<5:
        old=[(t*5+j)%50 for j in range(historical)]
        fresh=[50+(cursor-50+j)%50 for j in range(5-historical)]
        events.extend((t,i,tuple(old+fresh)) for i in range(5))
        cursor+=5-historical;t+=1
    if historical==5:
        events.extend((10,i,(0,1,2,3,4)) for i in range(5))
    return coverage_tail(events,5)[0]


def screening_designs():
    for groups in [(2,),(2,1),(3,),(2,1,1,1),(2,2,1),(3,1,1),(3,2),(4,1),(5,),
                   (2,)*5,(5,5),(5,3,2),(10,),(5,)*4,(10,10),(20,)]:
        yield Design('A-'+'-'.join(map(str,groups)),'A',
                     exact_groups(groups,width=2,steps=ceil(100/(2*len(groups)))),
                     sum(groups),metadata={'replication_groups':groups})
    for w in (2,3,4,5,10,20):
        for q in (range(1,w) if w<=5 else (1,3,5,7,9) if w==10 else (2,5,10,15,19)):
            span=q+5*(w-q)
            if span>100:continue
            yield Design(f'B-W{w}-Q{q}','B',shared_anchor_stream(num_clients=5,width=w,
                         shared=q,steps=ceil(100/span)),metadata={'width':w,'shared':q})
    graphs={'chain':[(0,1),(1,2),(2,3),(3,4)],
            'ring':[(0,1),(1,2),(2,3),(3,4),(4,0)],
            'star':[(0,i) for i in range(1,5)],
            'clique':[(i,j) for i in range(5) for j in range(i+1,5)],
            'clusters':[(0,1),(0,2),(1,2),(3,4)],
            'bridge':[(0,1),(0,2),(1,2),(2,3),(3,4)],
            'core_periphery':[(0,1),(0,2),(1,2),(0,3),(1,4)]}
    for name,edges in graphs.items():
        yield Design('C-'+name,'C',topology_stream(edges,4),metadata={'edges':edges,'width':4})
    for gap in (0,1,2,5,10,15):
        for family,same,partial in [('D',False,False),('E',False,True),('F',True,False)]:
            yield Design(f'{family}-G{gap}',family,migration(gap,same_client=same,partial=partial),
                         metadata={'gap':gap,'same_client':same,'partial':partial,'motif_steps':18})
    for visits in (3,4,5):
        yield Design(f'D-V{visits}','D',migration(0,visits=visits),metadata={'visits':visits})
    for w in (2,3,4,5,10):
        for q in (range(w) if w<=5 else (0,1,5,9)):
            steps=ceil((100-w)/(w-q))+1
            events=[(t,i,tuple((t*(w-q)+j)%100 for j in range(w))) for t in range(steps) for i in range(5)]
            yield Design(f'G-W{w}-Q{q}','G',events,local_depth=2,metadata={'width':w,'local_overlap':q})
    for depth in (2,3,5,10):
        for policy in ('aligned','reversed','cyclic','random','common_first','common_last'):
            events,ends=sequence_order(policy,depth)
            yield Design(f'I-L{depth}-{policy}','I',events,local_depth=depth,boundaries=ends,
                         allocation_order='client_semantic',metadata={'order':policy})
    for pattern in ('uniform','binary','long_tail','rotating'):
        yield Design('J-'+pattern,'J',replication_stream(pattern),metadata={'replication_pattern':pattern})
    for shared in (1,2,3,4):
        for fixed in (False,True):
            span=5*(5-shared)
            steps=ceil((100-shared)/span) if fixed else ceil(100/(shared+span))
            yield Design(f'K-Q{shared}-'+('fixed' if fixed else 'rotating'),'K',
                         shared_anchor_stream(num_clients=5,width=5,shared=shared,steps=steps,
                         fixed_anchor=fixed),metadata={'shared':shared,'fixed_anchor':fixed})
    for widths in ((2,2,2,2,5),(2,2,2,5,5),(2,3,4,5,6),(2,2,5,5,10),(2,3,4,5,10),(2,5,10,15,20)):
        yield Design('L-'+'-'.join(map(str,widths)),'L',heterogeneous_widths(widths),metadata={'widths':widths})
    for depths in ((1,1,2,2,2),(1,2,2,3,3),(1,2,3,4,5),(1,1,1,1,5),(1,5,5,5,5),(2,2,5,5,10)):
        events,ends=heterogeneous_depths(depths)
        yield Design('M-'+'-'.join(map(str,depths)),'M',events,max(2,len(depths)),max(depths),
                     boundaries=ends,metadata={'depths':depths})
    for s in ('S1','S2','S3'):
        events=foundational_stream(s,0)
        for n in (1,2,3):
            yield Design(f'N-{s}-random-{n}','N',events,local_depth=2,offline_clients=n)
            for policy in ('redundant','unique','novel','historical'):
                yield Design(f'N-{s}-{policy}-{n}','N',events,local_depth=2,
                             cohorts=oracle_cohorts(events,2,policy,n),
                             metadata={'oracle':True,'omission_policy':policy})
    for duration in (1,2,3,5,10,15):
        for comp in ('none','partial','exact'):
            events,cohorts=outage_stream(comp,duration)
            yield Design(f'O-{comp}-D{duration}','O',events,cohorts=cohorts,
                         metadata={'duration':duration,'compensation':comp,'hidden_client':0})
    for policy in ('increasing','decreasing','periodic','abrupt'):
        yield Design('Q-'+policy,'Q',dynamic_stream(policy),metadata={'policy':policy})
    for historical in range(6):
        yield Design(f'R-H{historical}','R',novelty_stream(historical),
                     metadata={'historical_classes':historical,'motif_not_runtime_novelty':True})
