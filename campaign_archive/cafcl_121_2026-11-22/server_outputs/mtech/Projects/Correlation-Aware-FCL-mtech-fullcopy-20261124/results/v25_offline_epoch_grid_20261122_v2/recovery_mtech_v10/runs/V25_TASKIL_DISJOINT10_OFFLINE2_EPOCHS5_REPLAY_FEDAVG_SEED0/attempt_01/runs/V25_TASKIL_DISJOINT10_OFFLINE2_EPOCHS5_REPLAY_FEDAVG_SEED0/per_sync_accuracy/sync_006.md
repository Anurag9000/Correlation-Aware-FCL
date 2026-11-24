# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.43% on 60 observed labels.
Validation accuracy: 0.75; seen labels never predicted: 0.
Server-accepted classes: 44; pending private classes: 16.
Accepted-only TEST accuracy: 0.836590909090909.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **83.0%** (0: 81%, 1: 85%) | **65.0%** (10: 82%, 11: 48%) | **90.5%** (20: 81%, 21: 100%) | **91.5%** (30: 96%, 31: 87%) | **53.5%** (40: 22%, 41: 85%) [PENDING upload] | **61.5%** (50: 30%, 51: 93%) [PENDING upload] |
| Client 1 | **84.5%** (2: 72%, 3: 97%) | **72.5%** (12: 87%, 13: 58%) | **83.0%** (22: 71%, 23: 95%) | **85.5%** (32: 87%, 33: 84%) | **74.5%** (42: 92%, 43: 57%) | **99.5%** (52: 100%, 53: 99%) |
| Client 2 | **53.0%** (4: 74%, 5: 32%) [PENDING upload] | **46.5%** (14: 88%, 15: 5%) [PENDING upload] | **45.0%** (24: 69%, 25: 21%) [PENDING upload] | **67.0%** (34: 86%, 35: 48%) [PENDING upload] | **49.5%** (44: 2%, 45: 97%) [PENDING upload] | **46.5%** (54: 91%, 55: 2%) [PENDING upload] |
| Client 3 | **76.5%** (6: 70%, 7: 83%) | **91.5%** (16: 91%, 17: 92%) | **73.5%** (26: 52%, 27: 95%) | **89.5%** (36: 87%, 37: 92%) | **93.5%** (46: 91%, 47: 96%) | **85.5%** (56: 78%, 57: 93%) |
| Client 4 | **81.0%** (8: 81%, 9: 81%) | **79.5%** (18: 91%, 19: 68%) | **81.5%** (28: 72%, 29: 91%) | **77.5%** (38: 86%, 39: 69%) | **94.0%** (48: 92%, 49: 96%) | **87.5%** (58: 87%, 59: 88%) |
