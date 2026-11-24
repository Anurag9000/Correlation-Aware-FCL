# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.31% on 26 observed labels.
Validation accuracy: 0.5223076923076924; seen labels never predicted: 2.
Server-accepted classes: 23; pending private classes: 3.
Accepted-only TEST accuracy: 0.5482608695652174.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **48.0%** (0: 1%, 1: 95%) [PENDING upload] | **71.0%** (5: 64%, 6: 78%) | **15.5%** (10: 27%, 11: 4%) | **60.5%** (15: 26%, 16: 95%) [PENDING upload] | **99.0%** (20: 98%, 21: 100%) [PENDING upload] |
| Client 1 | **59.5%** (1: 95%, 2: 24%) | **63.0%** (6: 78%, 7: 48%) | **45.0%** (11: 4%, 12: 86%) | **92.0%** (16: 95%, 17: 89%) [PENDING upload] | **50.0%** (21: 100%, 22: 0%) [PENDING upload] |
| Client 2 | **42.0%** (2: 24%, 3: 60%) | **33.0%** (7: 48%, 8: 18%) | **43.0%** (12: 86%, 13: 0%) | **77.0%** (17: 89%, 18: 65%) | **50.0%** (22: 0%, 23: 100%) |
| Client 3 | **44.5%** (3: 60%, 4: 29%) | **47.0%** (8: 18%, 9: 76%) | **43.5%** (13: 0%, 14: 87%) | **33.0%** (18: 65%, 19: 1%) | **50.0%** (23: 100%, 24: 0%) |
| Client 4 | **46.5%** (4: 29%, 5: 64%) | **51.5%** (9: 76%, 10: 27%) | **56.5%** (14: 87%, 15: 26%) | **49.5%** (19: 1%, 20: 98%) | **44.5%** (24: 0%, 25: 89%) |
