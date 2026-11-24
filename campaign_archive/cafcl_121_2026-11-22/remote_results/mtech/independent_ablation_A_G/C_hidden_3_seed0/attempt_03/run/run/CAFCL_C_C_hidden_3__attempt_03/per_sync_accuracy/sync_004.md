# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 70.85% on 40 observed labels.
Validation accuracy: 0.695; seen labels never predicted: 1.
Server-accepted classes: 38; pending private classes: 2.
Accepted-only TEST accuracy: 0.7194736842105263.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **79.5%** (8: 97%, 9: 62%) | **69.0%** (14: 48%, 15: 90%) | **70.5%** (26: 47%, 27: 94%) | **71.0%** (34: 93%, 35: 49%) |
| Client 1 | **55.0%** (6: 12%, 7: 98%) | **57.5%** (18: 20%, 19: 95%) | **46.0%** (28: 5%, 29: 87%) | **31.5%** (30: 60%, 31: 3%) |
| Client 2 | **81.5%** (4: 96%, 5: 67%) | **85.5%** (16: 71%, 17: 100%) | **84.5%** (22: 73%, 23: 96%) | **69.0%** (32: 40%, 33: 98%) |
| Client 3 | **85.0%** (0: 74%, 1: 96%) | **75.0%** (10: 79%, 11: 71%) | **91.5%** (20: 83%, 21: 100%) | **78.5%** (38: 86%, 39: 71%) |
| Client 4 | **84.0%** (2: 76%, 3: 92%) | **67.0%** (12: 84%, 13: 50%) | **85.5%** (24: 73%, 25: 98%) | **50.0%** (36: 100%, 37: 0%) [PENDING upload] |
