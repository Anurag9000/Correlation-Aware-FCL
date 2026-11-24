# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 70.10% on 60 observed labels.
Validation accuracy: 0.6956666666666667; seen labels never predicted: 0.
Server-accepted classes: 60; pending private classes: 0.
Accepted-only TEST accuracy: 0.701.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **78.5%** (0: 62%, 1: 95%) | **86.5%** (20: 73%, 21: 100%) | **60.5%** (40: 90%, 41: 31%) |
| Client 1 | **75.5%** (2: 51%, 3: 100%) | **77.0%** (22: 60%, 23: 94%) | **53.0%** (42: 100%, 43: 6%) |
| Client 2 | **78.0%** (4: 90%, 5: 66%) | **76.5%** (24: 63%, 25: 90%) | **62.5%** (44: 96%, 45: 29%) |
| Client 3 | **70.0%** (6: 55%, 7: 85%) | **68.5%** (26: 38%, 27: 99%) | **79.5%** (46: 70%, 47: 89%) |
| Client 4 | **65.5%** (8: 96%, 9: 35%) | **77.5%** (28: 82%, 29: 73%) | **57.5%** (48: 17%, 49: 98%) |
| Client 5 | **56.0%** (10: 89%, 11: 23%) | **89.5%** (30: 89%, 31: 90%) | **51.0%** (50: 99%, 51: 3%) |
| Client 6 | **57.5%** (12: 97%, 13: 18%) | **76.0%** (32: 61%, 33: 91%) | **88.0%** (52: 100%, 53: 76%) |
| Client 7 | **66.0%** (14: 33%, 15: 99%) | **61.5%** (34: 95%, 35: 28%) | **66.5%** (54: 33%, 55: 100%) |
| Client 8 | **82.5%** (16: 68%, 17: 97%) | **70.5%** (36: 42%, 37: 99%) | **57.0%** (56: 99%, 57: 15%) |
| Client 9 | **77.0%** (18: 82%, 19: 72%) | **72.5%** (38: 90%, 39: 55%) | **65.0%** (58: 35%, 59: 95%) |
