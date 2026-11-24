# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 67.65% on 40 observed labels.
Validation accuracy: 0.6795; seen labels never predicted: 0.
Server-accepted classes: 38; pending private classes: 2.
Accepted-only TEST accuracy: 0.6857894736842105.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **73.5%** (0: 52%, 1: 95%) | **51.0%** (10: 97%, 11: 5%) | **92.0%** (20: 84%, 21: 100%) | **50.0%** (30: 1%, 31: 99%) [PENDING upload] |
| Client 1 | **55.5%** (2: 11%, 3: 100%) | **68.5%** (12: 87%, 13: 50%) | **84.0%** (22: 77%, 23: 91%) | **80.5%** (32: 71%, 33: 90%) |
| Client 2 | **72.5%** (4: 95%, 5: 50%) | **53.5%** (14: 7%, 15: 100%) | **65.0%** (24: 31%, 25: 99%) | **53.5%** (34: 100%, 35: 7%) |
| Client 3 | **66.0%** (6: 37%, 7: 95%) | **87.0%** (16: 85%, 17: 89%) | **66.5%** (26: 34%, 27: 99%) | **60.0%** (36: 21%, 37: 99%) |
| Client 4 | **61.0%** (8: 99%, 9: 23%) | **78.5%** (18: 59%, 19: 98%) | **66.5%** (28: 65%, 29: 68%) | **68.0%** (38: 91%, 39: 45%) |
