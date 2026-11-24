# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 70.17% on 60 observed labels.
Validation accuracy: 0.704; seen labels never predicted: 1.
Server-accepted classes: 44; pending private classes: 16.
Accepted-only TEST accuracy: 0.7704545454545455.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **78.5%** (0: 58%, 1: 99%) | **63.0%** (10: 74%, 11: 52%) | **88.5%** (20: 82%, 21: 95%) | **89.5%** (30: 86%, 31: 93%) | **55.5%** (40: 33%, 41: 78%) [PENDING upload] | **61.0%** (50: 25%, 51: 97%) [PENDING upload] |
| Client 1 | **83.5%** (2: 70%, 3: 97%) | **68.5%** (12: 85%, 13: 52%) | **79.5%** (22: 81%, 23: 78%) | **70.5%** (32: 52%, 33: 89%) | **78.0%** (42: 92%, 43: 64%) | **99.5%** (52: 100%, 53: 99%) |
| Client 2 | **37.0%** (4: 32%, 5: 42%) [PENDING upload] | **54.5%** (14: 99%, 15: 10%) [PENDING upload] | **52.0%** (24: 94%, 25: 10%) [PENDING upload] | **51.0%** (34: 100%, 35: 2%) [PENDING upload] | **50.0%** (44: 0%, 45: 100%) [PENDING upload] | **49.0%** (54: 94%, 55: 4%) [PENDING upload] |
| Client 3 | **65.5%** (6: 55%, 7: 76%) | **81.5%** (16: 68%, 17: 95%) | **72.0%** (26: 49%, 27: 95%) | **80.0%** (36: 68%, 37: 92%) | **84.5%** (46: 89%, 47: 80%) | **82.0%** (56: 98%, 57: 66%) |
| Client 4 | **64.0%** (8: 91%, 9: 37%) | **58.5%** (18: 25%, 19: 92%) | **70.5%** (28: 68%, 29: 73%) | **73.5%** (38: 84%, 39: 63%) | **83.5%** (48: 73%, 49: 94%) | **80.5%** (58: 71%, 59: 90%) |
