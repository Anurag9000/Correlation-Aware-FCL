# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.97% on 30 observed labels.
Validation accuracy: 0.6146666666666667; seen labels never predicted: 0.
Server-accepted classes: 28; pending private classes: 2.
Accepted-only TEST accuracy: 0.6167857142857143.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **53.0%** (0: 6%, 1: 100%) | **49.0%** (10: 98%, 11: 0%) | **79.5%** (20: 59%, 21: 100%) |
| Client 1 | **61.0%** (2: 22%, 3: 100%) | **60.5%** (12: 92%, 13: 29%) | **51.0%** (22: 14%, 23: 88%) [PENDING upload] |
| Client 2 | **63.0%** (4: 96%, 5: 30%) | **69.0%** (14: 62%, 15: 76%) | **65.5%** (24: 37%, 25: 94%) |
| Client 3 | **51.0%** (6: 3%, 7: 99%) | **82.5%** (16: 70%, 17: 95%) | **54.5%** (26: 9%, 27: 100%) |
| Client 4 | **51.0%** (8: 99%, 9: 3%) | **63.5%** (18: 82%, 19: 45%) | **60.5%** (28: 60%, 29: 61%) |
