# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 65.70% on 30 observed labels.
Validation accuracy: 0.6506666666666666; seen labels never predicted: 0.
Server-accepted classes: 28; pending private classes: 2.
Accepted-only TEST accuracy: 0.6675.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **57.0%** (8: 99%, 9: 15%) | **54.0%** (14: 8%, 15: 100%) | **67.0%** (26: 37%, 27: 97%) |
| Client 1 | **51.0%** (6: 2%, 7: 100%) | **69.0%** (18: 76%, 19: 62%) | **61.0%** (28: 83%, 29: 39%) |
| Client 2 | **67.0%** (4: 98%, 5: 36%) | **69.5%** (16: 39%, 17: 100%) | **78.5%** (22: 73%, 23: 84%) |
| Client 3 | **82.0%** (0: 79%, 1: 85%) | **52.5%** (10: 100%, 11: 5%) | **92.0%** (20: 84%, 21: 100%) |
| Client 4 | **79.0%** (2: 60%, 3: 98%) | **55.0%** (12: 100%, 13: 10%) | **51.0%** (24: 98%, 25: 4%) [PENDING upload] |
