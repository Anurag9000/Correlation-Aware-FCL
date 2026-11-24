# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.54% on 70 observed labels.
Validation accuracy: 0.7985714285714286; seen labels never predicted: 0.
Server-accepted classes: 56; pending private classes: 14.
Accepted-only TEST accuracy: 0.8642857142857143.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **86.0%** (8: 92%, 9: 80%) | **78.0%** (14: 66%, 15: 90%) | **82.0%** (26: 84%, 27: 80%) | **82.5%** (34: 79%, 35: 86%) | **95.5%** (48: 94%, 49: 97%) | **89.5%** (56: 93%, 57: 86%) | **97.0%** (62: 95%, 63: 99%) |
| Client 1 | **49.5%** (6: 0%, 7: 99%) [PENDING upload] | **66.5%** (18: 41%, 19: 92%) [PENDING upload] | **50.0%** (28: 40%, 29: 60%) [PENDING upload] | **26.5%** (30: 52%, 31: 1%) [PENDING upload] | **64.5%** (42: 67%, 43: 62%) [PENDING upload] | **32.5%** (54: 8%, 55: 57%) [PENDING upload] | **39.5%** (68: 37%, 69: 42%) [PENDING upload] |
| Client 2 | **89.5%** (4: 92%, 5: 87%) | **96.0%** (16: 96%, 17: 96%) | **85.5%** (22: 77%, 23: 94%) | **83.0%** (32: 84%, 33: 82%) | **93.5%** (46: 94%, 47: 93%) | **90.5%** (58: 86%, 59: 95%) | **91.5%** (60: 96%, 61: 87%) |
| Client 3 | **88.5%** (0: 95%, 1: 82%) | **79.0%** (10: 71%, 11: 87%) | **95.0%** (20: 91%, 21: 99%) | **87.0%** (38: 87%, 39: 87%) | **72.5%** (44: 80%, 45: 65%) | **76.5%** (50: 88%, 51: 65%) | **87.5%** (66: 78%, 67: 97%) |
| Client 4 | **88.5%** (2: 90%, 3: 87%) | **73.5%** (12: 79%, 13: 68%) | **88.0%** (24: 93%, 25: 83%) | **91.0%** (36: 93%, 37: 89%) | **91.0%** (40: 93%, 41: 89%) | **99.5%** (52: 100%, 53: 99%) | **62.5%** (64: 90%, 65: 35%) |
