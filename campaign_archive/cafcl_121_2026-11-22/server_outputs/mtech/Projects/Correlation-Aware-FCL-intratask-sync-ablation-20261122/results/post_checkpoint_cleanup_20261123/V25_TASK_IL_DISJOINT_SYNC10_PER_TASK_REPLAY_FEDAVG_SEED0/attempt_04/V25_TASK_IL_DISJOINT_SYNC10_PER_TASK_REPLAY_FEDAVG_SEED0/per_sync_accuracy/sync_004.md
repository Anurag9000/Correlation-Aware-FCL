# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.90% on 10 observed labels.
Validation accuracy: 0.776; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.779.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **89.0%** (0: 87%, 1: 91%) | **89.0%** (0: 87%, 1: 91%) | **89.0%** (0: 87%, 1: 91%) | **89.0%** (0: 87%, 1: 91%) |
| Client 1 | **87.5%** (2: 91%, 3: 84%) | **87.5%** (2: 91%, 3: 84%) | **87.5%** (2: 91%, 3: 84%) | **87.5%** (2: 91%, 3: 84%) |
| Client 2 | **78.0%** (4: 74%, 5: 82%) | **78.0%** (4: 74%, 5: 82%) | **78.0%** (4: 74%, 5: 82%) | **78.0%** (4: 74%, 5: 82%) |
| Client 3 | **65.0%** (6: 75%, 7: 55%) | **65.0%** (6: 75%, 7: 55%) | **65.0%** (6: 75%, 7: 55%) | **65.0%** (6: 75%, 7: 55%) |
| Client 4 | **70.0%** (8: 94%, 9: 46%) | **70.0%** (8: 94%, 9: 46%) | **70.0%** (8: 94%, 9: 46%) | **70.0%** (8: 94%, 9: 46%) |
