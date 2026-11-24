"""Exact integer overlap feasibility and independently checked constraints.

One variable per possible nonempty client subset denotes how many class
identities are assigned to that subset. Pairwise intersection constraints
do not force a separate class per edge: shared hyperedges are permitted.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint


def solve_overlap(widths, intersections, num_classes=100, *, time_limit=10):
    widths=tuple(map(int,widths));n=len(widths)
    if not 2<=n<=12 or min(widths)<2 or max(widths)>num_classes:
        raise ValueError('Exact subset solver requires 2..12 clients and valid widths')
    pairs=list(combinations(range(n),2))
    if set(intersections)!=set(pairs):
        raise ValueError('Specify exact overlap for every pair, including zero')
    for (i,j),value in intersections.items():
        if int(value)!=value or not 0<=value<=min(widths[i],widths[j]):
            raise ValueError('Invalid requested pair overlap')
    subsets=[]
    for bits in range(1,1<<n):
        group=tuple(i for i in range(n) if bits&(1<<i))
        if any(intersections[e]==0 for e in combinations(group,2)):continue
        subsets.append(group)
    rows=[[int(i in g) for g in subsets] for i in range(n)]
    rows.extend([[int(i in g and j in g) for g in subsets] for i,j in pairs])
    rhs=list(widths)+[intersections[e] for e in pairs]
    rows.append([1]*len(subsets))
    result=milp(c=np.ones(len(subsets)),integrality=np.ones(len(subsets)),
        bounds=Bounds(0,num_classes),constraints=LinearConstraint(np.array(rows,dtype=float),
            np.array(rhs+[0],dtype=float),np.array(rhs+[num_classes],dtype=float)),
        options={'time_limit':float(time_limit)})
    if result.status==2:raise ValueError('Proven infeasible overlap constraints')
    if not result.success or result.x is None:
        raise RuntimeError('Overlap solver did not establish an optimum: '+str(result.message))
    solution=np.rint(result.x).astype(int)
    if np.max(np.abs(solution-result.x))>1e-5:raise RuntimeError('Nonintegral solution')
    tasks=[[] for _ in widths];cursor=0
    for group,count in zip(subsets,solution):
        for _ in range(count):
            for i in group:tasks[i].append(cursor)
            cursor+=1
    if cursor>num_classes or tuple(map(len,tasks))!=widths:
        raise RuntimeError('Solver returned invalid task widths/coverage')
    for i,j in pairs:
        if len(set(tasks[i])&set(tasks[j]))!=intersections[i,j]:
            raise RuntimeError('Solver result failed independent intersection audit')
    return tuple(tuple(x) for x in tasks)


def recurrence_gap(events, source, destination, *, exact=True, shared=None):
    """Verify specific event coordinates; gap means intervening atomic steps."""
    lookup={(t,i):set(labels) for t,i,labels in events}
    a,b=lookup[source],lookup[destination]
    if destination[0]<source[0]:raise ValueError('Destination precedes source')
    if exact and a!=b:raise ValueError('Expected exact task recurrence')
    if not exact and (a==b or not a&b):
        raise ValueError('Expected partial overlap between distinct tasks')
    if shared is not None and len(a&b)!=shared:raise ValueError('Wrong overlap cardinality')
    return destination[0]-source[0]-1
