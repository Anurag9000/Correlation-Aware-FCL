# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 74.43% on 30 observed labels.
Validation accuracy: 0.7493333333333333; seen labels never predicted: 1.
Server-accepted classes: 24; pending private classes: 6.
Accepted-only TEST accuracy: 0.80375.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **81.5%** (8: 96%, 9: 67%) | **69.0%** (14: 53%, 15: 85%) | **71.0%** (26: 49%, 27: 93%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **60.5%** (18: 37%, 19: 84%) [PENDING upload] | **41.5%** (28: 25%, 29: 58%) [PENDING upload] |
| Client 2 | **87.5%** (4: 92%, 5: 83%) | **89.5%** (16: 81%, 17: 98%) | **82.5%** (22: 72%, 23: 93%) |
| Client 3 | **88.0%** (0: 82%, 1: 94%) | **68.5%** (10: 93%, 11: 44%) | **91.5%** (20: 84%, 21: 99%) |
| Client 4 | **85.0%** (2: 79%, 3: 91%) | **65.5%** (12: 88%, 13: 43%) | **85.0%** (24: 73%, 25: 97%) |
