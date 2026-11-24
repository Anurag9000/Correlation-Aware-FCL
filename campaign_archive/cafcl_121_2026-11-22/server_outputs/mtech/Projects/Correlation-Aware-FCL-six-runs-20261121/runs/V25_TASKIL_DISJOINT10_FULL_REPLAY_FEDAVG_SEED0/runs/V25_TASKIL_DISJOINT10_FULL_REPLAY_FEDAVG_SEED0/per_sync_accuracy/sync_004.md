# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.70% on 40 observed labels.
Validation accuracy: 0.831; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.817.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **82.5%** (0: 75%, 1: 90%) | **62.5%** (10: 90%, 11: 35%) | **93.5%** (20: 93%, 21: 94%) | **91.0%** (30: 96%, 31: 86%) |
| Client 1 | **84.0%** (2: 70%, 3: 98%) | **72.5%** (12: 70%, 13: 75%) | **88.0%** (22: 82%, 23: 94%) | **85.0%** (32: 83%, 33: 87%) |
| Client 2 | **86.5%** (4: 91%, 5: 82%) | **71.5%** (14: 55%, 15: 88%) | **86.5%** (24: 83%, 25: 90%) | **78.0%** (34: 88%, 35: 68%) |
| Client 3 | **81.0%** (6: 69%, 7: 93%) | **89.0%** (16: 87%, 17: 91%) | **77.0%** (26: 58%, 27: 96%) | **83.5%** (36: 70%, 37: 97%) |
| Client 4 | **86.5%** (8: 94%, 9: 79%) | **82.0%** (18: 81%, 19: 83%) | **76.0%** (28: 72%, 29: 80%) | **77.5%** (38: 88%, 39: 67%) |
