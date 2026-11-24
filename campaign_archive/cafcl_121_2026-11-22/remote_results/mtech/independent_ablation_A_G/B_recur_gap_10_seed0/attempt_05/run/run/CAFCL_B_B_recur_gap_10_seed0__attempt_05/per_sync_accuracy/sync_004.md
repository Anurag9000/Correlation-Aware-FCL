# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 82.03% on 40 observed labels.
Validation accuracy: 0.816; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.82025.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **90.0%** (0: 91%, 1: 89%) | **72.0%** (10: 97%, 11: 47%) | **93.5%** (20: 88%, 21: 99%) | **91.0%** (30: 92%, 31: 90%) |
| Client 1 | **89.0%** (2: 81%, 3: 97%) | **70.5%** (12: 94%, 13: 47%) | **86.0%** (22: 74%, 23: 98%) | **80.5%** (32: 70%, 33: 91%) |
| Client 2 | **83.0%** (4: 94%, 5: 72%) | **73.0%** (14: 51%, 15: 95%) | **87.0%** (24: 86%, 25: 88%) | **72.0%** (34: 99%, 35: 45%) |
| Client 3 | **79.5%** (6: 75%, 7: 84%) | **91.5%** (16: 86%, 17: 97%) | **79.0%** (26: 59%, 27: 99%) | **76.0%** (36: 55%, 37: 97%) |
| Client 4 | **82.0%** (8: 89%, 9: 75%) | **83.0%** (18: 83%, 19: 83%) | **80.5%** (28: 77%, 29: 84%) | **81.5%** (38: 81%, 39: 82%) |
