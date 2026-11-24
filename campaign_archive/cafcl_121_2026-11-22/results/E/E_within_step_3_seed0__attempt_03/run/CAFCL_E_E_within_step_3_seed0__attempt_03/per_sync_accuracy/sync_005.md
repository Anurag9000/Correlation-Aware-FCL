# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 85.53% on 30 observed labels.
Validation accuracy: 0.8673333333333333; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.8553333333333333.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **88.0%** (0: 83%, 1: 93%) | **82.5%** (6: 86%, 7: 79%) | **71.5%** (12: 91%, 13: 52%) | **93.0%** (20: 87%, 21: 99%) | **85.0%** (24: 83%, 25: 87%) |
| Client 1 | **88.0%** (0: 83%, 1: 93%) | **82.5%** (6: 86%, 7: 79%) | **71.5%** (12: 91%, 13: 52%) | **90.0%** (22: 85%, 23: 95%) | **85.0%** (24: 83%, 25: 87%) |
| Client 2 | **91.0%** (4: 92%, 5: 90%) | **82.5%** (6: 86%, 7: 79%) | **93.5%** (16: 87%, 17: 100%) | **79.0%** (18: 72%, 19: 86%) | **80.5%** (26: 63%, 27: 98%) |
| Client 3 | **88.0%** (0: 83%, 1: 93%) | **80.5%** (10: 85%, 11: 76%) | **83.5%** (14: 70%, 15: 97%) | **79.0%** (18: 72%, 19: 86%) | **85.0%** (28: 87%, 29: 83%) |
| Client 4 | **89.5%** (2: 85%, 3: 94%) | **90.5%** (8: 90%, 9: 91%) | **71.5%** (12: 91%, 13: 52%) | **79.0%** (18: 72%, 19: 86%) | **85.0%** (24: 83%, 25: 87%) |
