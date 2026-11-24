# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 70.23% on 26 observed labels.
Validation accuracy: 0.7184615384615385; seen labels never predicted: 0.
Server-accepted classes: 23; pending private classes: 3.
Accepted-only TEST accuracy: 0.7578260869565218.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **84.5%** (0: 79%, 1: 90%) [PENDING upload] | **79.5%** (5: 69%, 6: 90%) | **77.5%** (10: 72%, 11: 83%) | **77.0%** (15: 60%, 16: 94%) [PENDING upload] | **68.0%** (20: 83%, 21: 53%) [PENDING upload] |
| Client 1 | **78.0%** (1: 90%, 2: 66%) | **63.0%** (6: 90%, 7: 36%) | **75.5%** (11: 83%, 12: 68%) | **88.5%** (16: 94%, 17: 83%) [PENDING upload] | **52.0%** (21: 53%, 22: 51%) [PENDING upload] |
| Client 2 | **57.0%** (2: 66%, 3: 48%) | **55.0%** (7: 36%, 8: 74%) | **73.0%** (12: 68%, 13: 78%) | **74.5%** (17: 83%, 18: 66%) | **66.5%** (22: 51%, 23: 82%) |
| Client 3 | **70.5%** (3: 48%, 4: 93%) | **65.5%** (8: 74%, 9: 57%) | **61.5%** (13: 78%, 14: 45%) | **72.5%** (18: 66%, 19: 79%) | **58.5%** (23: 82%, 24: 35%) |
| Client 4 | **81.0%** (4: 93%, 5: 69%) | **64.5%** (9: 57%, 10: 72%) | **52.5%** (14: 45%, 15: 60%) | **81.0%** (19: 79%, 20: 83%) | **63.5%** (24: 35%, 25: 92%) |
