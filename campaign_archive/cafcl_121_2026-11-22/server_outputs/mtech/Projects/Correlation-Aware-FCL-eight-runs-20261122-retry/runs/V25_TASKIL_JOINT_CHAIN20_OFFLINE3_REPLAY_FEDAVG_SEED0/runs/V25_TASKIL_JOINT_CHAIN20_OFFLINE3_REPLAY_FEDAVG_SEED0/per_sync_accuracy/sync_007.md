# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 70.69% on 36 observed labels.
Validation accuracy: 0.7266666666666667; seen labels never predicted: 0.
Server-accepted classes: 32; pending private classes: 4.
Accepted-only TEST accuracy: 0.8115625.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **82.0%** (0: 71%, 1: 93%) | **68.0%** (5: 74%, 6: 62%) | **72.5%** (10: 83%, 11: 62%) | **81.0%** (15: 71%, 16: 91%) | **94.0%** (20: 88%, 21: 100%) | **78.5%** (25: 68%, 26: 89%) | **94.0%** (30: 98%, 31: 90%) |
| Client 1 | **90.5%** (1: 93%, 2: 88%) | **64.5%** (6: 62%, 7: 67%) | **78.0%** (11: 62%, 12: 94%) | **93.0%** (16: 91%, 17: 95%) | **70.5%** (21: 100%, 22: 41%) | **44.5%** (26: 89%, 27: 0%) [PENDING upload] | **59.0%** (31: 90%, 32: 28%) [PENDING upload] |
| Client 2 | **85.5%** (2: 88%, 3: 83%) | **79.5%** (7: 67%, 8: 92%) | **81.5%** (12: 94%, 13: 69%) | **85.0%** (17: 95%, 18: 75%) | **65.5%** (22: 41%, 23: 90%) | **6.5%** (27: 0%, 28: 13%) [PENDING upload] | **29.0%** (32: 28%, 33: 30%) [PENDING upload] |
| Client 3 | **73.5%** (3: 83%, 4: 64%) | **81.5%** (8: 92%, 9: 71%) | **63.0%** (13: 69%, 14: 57%) | **79.0%** (18: 75%, 19: 83%) | **53.0%** (23: 90%, 24: 16%) | **48.5%** (28: 13%, 29: 84%) [PENDING upload] | **55.0%** (33: 30%, 34: 80%) [PENDING upload] |
| Client 4 | **69.0%** (4: 64%, 5: 74%) | **77.0%** (9: 71%, 10: 83%) | **64.0%** (14: 57%, 15: 71%) | **85.5%** (19: 83%, 20: 88%) | **42.0%** (24: 16%, 25: 68%) | **91.0%** (29: 84%, 30: 98%) | **82.5%** (34: 80%, 35: 85%) |
