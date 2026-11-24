# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 84.92% on 24 observed labels.
Validation accuracy: 0.8675; seen labels never predicted: 0.
Server-accepted classes: 24; pending private classes: 0.
Accepted-only TEST accuracy: 0.8491666666666666.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **87.0%** (0: 92%, 1: 82%) | **82.0%** (4: 95%, 5: 69%) | **89.5%** (8: 93%, 9: 86%) | **79.5%** (12: 94%, 13: 65%) | **92.0%** (16: 86%, 17: 98%) | **95.0%** (20: 90%, 21: 100%) |
| Client 1 | **85.0%** (2: 72%, 3: 98%) | **80.5%** (6: 83%, 7: 78%) | **75.5%** (10: 86%, 11: 65%) | **84.0%** (14: 75%, 15: 93%) | **84.0%** (18: 91%, 19: 77%) | **85.0%** (22: 74%, 23: 96%) |
