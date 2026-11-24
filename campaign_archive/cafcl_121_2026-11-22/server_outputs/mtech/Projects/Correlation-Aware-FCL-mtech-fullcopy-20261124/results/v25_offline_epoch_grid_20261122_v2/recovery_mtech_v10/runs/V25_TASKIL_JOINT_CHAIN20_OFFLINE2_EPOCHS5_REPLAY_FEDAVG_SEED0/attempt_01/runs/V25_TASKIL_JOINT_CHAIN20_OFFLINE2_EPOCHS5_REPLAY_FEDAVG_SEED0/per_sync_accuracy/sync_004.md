# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 67.52% on 21 observed labels.
Validation accuracy: 0.6942857142857143; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.7135.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **67.0%** (0: 74%, 1: 60%) | **64.0%** (5: 82%, 6: 46%) | **79.5%** (10: 100%, 11: 59%) | **86.5%** (15: 97%, 16: 76%) |
| Client 1 | **77.5%** (1: 60%, 2: 95%) | **62.0%** (6: 46%, 7: 78%) | **70.5%** (11: 59%, 12: 82%) | **38.0%** (16: 76%, 17: 0%) [PENDING upload] |
| Client 2 | **97.5%** (2: 95%, 3: 100%) | **87.5%** (7: 78%, 8: 97%) | **87.5%** (12: 82%, 13: 93%) | **28.0%** (17: 0%, 18: 56%) [PENDING upload] |
| Client 3 | **57.5%** (3: 100%, 4: 15%) | **52.0%** (8: 97%, 9: 7%) | **64.0%** (13: 93%, 14: 35%) | **67.0%** (18: 56%, 19: 78%) |
| Client 4 | **48.5%** (4: 15%, 5: 82%) | **53.5%** (9: 7%, 10: 100%) | **66.0%** (14: 35%, 15: 97%) | **83.0%** (19: 78%, 20: 88%) |
