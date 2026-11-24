# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 76.58% on 40 observed labels.
Validation accuracy: 0.78; seen labels never predicted: 2.
Server-accepted classes: 32; pending private classes: 8.
Accepted-only TEST accuracy: 0.8296875.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **85.0%** (8: 94%, 9: 76%) | **73.0%** (14: 60%, 15: 86%) | **80.0%** (26: 65%, 27: 95%) | **77.5%** (34: 83%, 35: 72%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **50.0%** (18: 0%, 19: 100%) [PENDING upload] | **53.5%** (28: 42%, 29: 65%) [PENDING upload] | **50.5%** (30: 100%, 31: 1%) [PENDING upload] |
| Client 2 | **86.0%** (4: 90%, 5: 82%) | **90.5%** (16: 82%, 17: 99%) | **87.0%** (22: 80%, 23: 94%) | **73.5%** (32: 52%, 33: 95%) |
| Client 3 | **88.0%** (0: 91%, 1: 85%) | **77.5%** (10: 84%, 11: 71%) | **94.5%** (20: 92%, 21: 97%) | **84.0%** (38: 83%, 39: 85%) |
| Client 4 | **84.5%** (2: 84%, 3: 85%) | **71.0%** (12: 84%, 13: 58%) | **88.0%** (24: 81%, 25: 95%) | **87.5%** (36: 83%, 37: 92%) |
