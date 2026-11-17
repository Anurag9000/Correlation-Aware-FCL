"""Faithful inventory of all 152 user-planned experiment blocks.

A block is not marked executable unless the runner implements its mechanism.
This is a coverage ledger, NOT a claim that all 152 blocks are implemented.
"""
from __future__ import annotations

FAMILIES={
'A':[
'Participation rate','Fixed versus resampled participants','Equal communication rounds','Matched total client updates',
'Old/new class-holder participation','Stage-dependent participation schedule','Class diversity at equal participation',
'Client redundancy with matched class coverage','Short frequent versus long rare participation',
'Monotonically changing participation','Equal lifetime frequency, different timing','Equal-plasticity forgetting comparison'],
'B':[
'Related vs unrelated new classes','Select/exclude related-class client','Relationship strength',
'Gradient alignment classes','One-to-one related classes','Many-to-one relations','Related-class training dose',
'Repeated related-client selection','Mixed related/unrelated participants','Feature similarity intervention',
'Local GEM x relationship','Per-old-class spillover'],
'C':[
'Overlap magnitude','Joint versus separate participation','Shared-class dominance',
'Same labels with disjoint examples','Exact sample overlap','Shared class across shifted domains',
'Gradient interactions of overlapping clients','Redundancy versus complementary coverage',
'Overlap x local GEM','Longitudinal overlap selection'],
'D':[
'Same-client recurrence','Cross-client recurrence','Related but new-label recurrence',
'Shifted-domain recurrence','Recurrence delay','Recurrence timing','Select/exclude recurring client',
'Periodic recurrence','Mixed old/new recurrence','Inactive-client recurrence',
'Delayed recurrent-client participation','Recurrence under GEM'],
'E':[
'Relatedness x overlap','Relatedness x recurrence','Overlap x recurrence',
'Triple relationship interaction','Mixed helpful and harmful client updates','Asynchronous old/new class relationships',
'Rare-class recurrence','Persistent vs temporary correlations','Relatedness strength x participation',
'Relatedness x aggregation weights'],
'F':[
'Uniform random','Fixed subset','Round-robin','Least recently selected','Class coverage maximizing',
'Old-class coverage','New-class coverage','Balanced old and new coverage','Maximum label diversity',
'Maximum label overlap','Maximum feature similarity','Minimum feature similarity',
'Gradient alignment selection','Gradient conflict selection','Forgetting-aware selection',
'Recurrence-aware selection','Fairness-constrained selection','Randomized correlation-aware selection',
'History-aware adaptive selection','Oracle exhaustive selection'],
'G':[
'Local SGD','Local GEM','Local A-GEM','Experience replay','ER-ACE','DER and DER++','EWC',
'Synaptic intelligence','MAS','Local distillation','Replay vs no replay','Memory capacity',
'Memory selection algorithms','GEM per-group constraints','Proximal regularization plus GEM',
'Heterogeneous local algorithms'],
'H':[
'Active-only synchronization','All-client synchronization','Sync when rejoining','Optimizer state after broadcast',
'Private memory persistence','Synchronization frequency','Long inactive gap','Local personalization'],
'I':[
'Sample-weighted FedAvg','Uniform aggregation','Class-balanced aggregation','Correlation-aware aggregation',
'Old-retention aggregation','Server momentum/adaptive optimizer','Gradient conflict correction',
'Participation-probability correction'],
'J':[
'IID vs non-IID','Label distribution skew','Client sample-count skew','Feature distribution skew',
'Class imbalance','Dependent versus independent client streams','Unequal arrival rates',
'Inactive client data retention','Recurrence timing distribution','Spurious correlations',
'Label noise','Changing correlation structure'],
'K':[
'Class-incremental learning','Task-incremental learning','Domain-incremental learning','Task-free learning',
'Fixed versus expanding classifier','Known versus hidden experience boundaries',
'Asynchronous versus synchronized transitions','Closed-world versus open-world class arrival'],
'L':[
'Leave-one-client-out aggregation','Add-one-client aggregation','Pairwise client interactions',
'Exhaustive participating subset enumeration','Pre/post local historical loss','Pre/post aggregation loss',
'Gradient interference analysis','Representation drift','Correlation intervention',
'Historical contribution attribution'],
'M':[
'Dataset generalization','Model architecture','Client population size','Local optimization intensity',
'Randomness robustness','Communication efficiency','Computational efficiency','Privacy/resource restrictions'],
'N':[
'Correlation-aware participation','Class-participation balancing','Recurrence-aware scheduling',
'GEM-informed participation','Adaptive participation rate','Joint selection and aggregation'],
}
CATALOGUE={f'{fam}{i:02d}':name for fam,names in FAMILIES.items() for i,name in enumerate(names,1)}
assert len(CATALOGUE)==152

# Only these IDs have a corresponding concrete intervention in the current
# simulator. All others remain conspicuously NOT IMPLEMENTED. Some are proxy
# baselines, not full reproductions of the research method in the catalogue.
IMPLEMENTED={
'A01','A02','A05','A08',
'B01','B02',
'C01','C02',
'D01','D02','D05','D07',
'E09',
'F01','F02','F03','F04','F05','F06','F07','F09','F10','F16',
'G01','G02','G03','G04','G07','G11','G12',
'H01','H02','H03','H05',
'I01','I02',
'J01','J02',
'K01',
'M01','M02','M03','M04',
'N01','N03',
'A10','A11','B11','C08','C09','D12','E01','E02','E04','E06','E10',
'F08','F17','F18','G05','G06','G09','G10','G13','G15','G16',
'H06','I06','J08','L01','L02','M05','N02'
}
# Partial/approximate coverage is deliberately distinguishable from a complete
# block reproduction; these blocks have functioning simple comparison arms.
PARTIAL={'A05','A08','B01','B02','C01','C02','D01','D02','D05','D07','E09','F09','F10','F16',
         'G07','G11','G12','H03','H05','J02','M01','M03','M04','N01','N03',
         'A11','B11','C08','C09','D12','E01','E02','E04','E06','E10',
         'F08','F17','F18','G05','G06','G09','G10','G13','G15','G16',
         'H06','I06','J08','L02','M05','N02'}

def catalogue_report():
    return [{'id':key,'name':name,'status':('partial_implementation' if key in PARTIAL else 'implemented')
              if key in IMPLEMENTED else 'not_implemented'} for key,name in CATALOGUE.items()]

# Expanded November 2026 implementation. Every new block is PARTIAL until
# exact reference fidelity, causal controls, and CIFAR validations are verified.
EXPANDED_PARTIAL={'F15', 'A03', 'B08', 'J11', 'K06', 'I08', 'G14', 'H07', 'J12', 'D06', 'L07', 'J10', 'M07', 'C06', 'I07', 'N04', 'E07', 'B03', 'J04', 'C05', 'I04', 'A09', 'G08', 'C03', 'L05', 'B09', 'J06', 'J09', 'B07', 'D04', 'D09', 'L03', 'L06', 'J03', 'L10', 'M06', 'N05', 'L04', 'B05', 'C04', 'F14', 'F20', 'L09', 'C07', 'K07', 'C10', 'F11', 'D10', 'D03', 'A07', 'F13', 'I03', 'K03', 'E03', 'H04', 'E05', 'D08', 'A04', 'A06', 'J05', 'K02', 'K04', 'L08', 'F19', 'B04', 'D11', 'J07', 'B12', 'I05', 'F12', 'E08'}
IMPLEMENTED.update(EXPANDED_PARTIAL)
PARTIAL.update(EXPANDED_PARTIAL)
