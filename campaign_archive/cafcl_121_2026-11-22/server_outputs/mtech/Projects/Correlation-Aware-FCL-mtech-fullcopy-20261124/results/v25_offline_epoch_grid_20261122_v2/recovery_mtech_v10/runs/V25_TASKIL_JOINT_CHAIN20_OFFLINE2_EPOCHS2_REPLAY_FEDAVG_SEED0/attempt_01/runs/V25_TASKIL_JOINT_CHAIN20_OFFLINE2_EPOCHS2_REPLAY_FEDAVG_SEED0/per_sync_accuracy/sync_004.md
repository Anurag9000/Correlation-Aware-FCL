# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 65.43% on 21 observed labels.
Validation accuracy: 0.6647619047619048; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.7145.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **73.5%** (0: 97%, 1: 50%) | **62.0%** (5: 75%, 6: 49%) | **73.5%** (10: 76%, 11: 71%) | **83.0%** (15: 66%, 16: 100%) |
| Client 1 | **71.5%** (1: 50%, 2: 93%) | **70.5%** (6: 49%, 7: 92%) | **72.5%** (11: 71%, 12: 74%) | **50.0%** (16: 100%, 17: 0%) [PENDING upload] |
| Client 2 | **96.5%** (2: 93%, 3: 100%) | **79.0%** (7: 92%, 8: 66%) | **71.5%** (12: 74%, 13: 69%) | **19.5%** (17: 0%, 18: 39%) [PENDING upload] |
| Client 3 | **50.5%** (3: 100%, 4: 1%) | **45.0%** (8: 66%, 9: 24%) | **65.5%** (13: 69%, 14: 62%) | **61.5%** (18: 39%, 19: 84%) |
| Client 4 | **38.0%** (4: 1%, 5: 75%) | **50.0%** (9: 24%, 10: 76%) | **64.0%** (14: 62%, 15: 66%) | **85.0%** (19: 84%, 20: 86%) |
