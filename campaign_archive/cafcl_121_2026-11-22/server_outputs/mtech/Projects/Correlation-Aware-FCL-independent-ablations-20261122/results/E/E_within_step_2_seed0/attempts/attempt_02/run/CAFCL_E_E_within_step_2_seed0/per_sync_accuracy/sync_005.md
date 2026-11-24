# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 85.15% on 40 observed labels.
Validation accuracy: 0.866; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.8515.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **87.5%** (2: 81%, 3: 94%) | **83.5%** (8: 94%, 9: 73%) | **81.0%** (18: 78%, 19: 84%) | **86.5%** (28: 89%, 29: 84%) | **80.5%** (32: 72%, 33: 89%) |
| Client 1 | **90.5%** (0: 94%, 1: 87%) | **79.0%** (10: 83%, 11: 75%) | **95.0%** (16: 92%, 17: 98%) | **95.5%** (30: 97%, 31: 94%) | **80.5%** (32: 72%, 33: 89%) |
| Client 2 | **79.5%** (6: 80%, 7: 79%) | **83.5%** (8: 94%, 9: 73%) | **84.0%** (22: 73%, 23: 95%) | **90.0%** (24: 90%, 25: 90%) | **90.5%** (36: 84%, 37: 97%) |
| Client 3 | **90.5%** (0: 94%, 1: 87%) | **82.5%** (14: 75%, 15: 90%) | **94.5%** (20: 93%, 21: 96%) | **76.0%** (26: 58%, 27: 94%) | **81.5%** (38: 77%, 39: 86%) |
| Client 4 | **89.5%** (4: 92%, 5: 87%) | **74.5%** (12: 94%, 13: 55%) | **95.0%** (16: 92%, 17: 98%) | **90.0%** (24: 90%, 25: 90%) | **81.5%** (34: 72%, 35: 91%) |
