# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.80% on 70 observed labels.
Validation accuracy: 0.7462857142857143; seen labels never predicted: 2.
Server-accepted classes: 52; pending private classes: 18.
Accepted-only TEST accuracy: 0.8113461538461538.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **76.5%** (0: 65%, 1: 88%) | **52.5%** (10: 99%, 11: 6%) | **90.5%** (20: 85%, 21: 96%) | **92.0%** (30: 85%, 31: 99%) | **85.5%** (40: 95%, 41: 76%) | **66.0%** (50: 40%, 51: 92%) | **96.0%** (60: 97%, 61: 95%) |
| Client 1 | **88.0%** (2: 80%, 3: 96%) | **70.5%** (12: 71%, 13: 70%) | **89.5%** (22: 85%, 23: 94%) | **44.0%** (32: 86%, 33: 2%) [PENDING upload] | **68.5%** (42: 94%, 43: 43%) [PENDING upload] | **50.0%** (52: 100%, 53: 0%) [PENDING upload] | **52.0%** (62: 100%, 63: 4%) [PENDING upload] |
| Client 2 | **81.5%** (4: 96%, 5: 67%) | **74.0%** (14: 71%, 15: 77%) | **86.5%** (24: 92%, 25: 81%) | **61.5%** (34: 96%, 35: 27%) | **62.5%** (44: 95%, 45: 30%) | **37.0%** (54: 71%, 55: 3%) [PENDING upload] | **47.5%** (64: 20%, 65: 75%) [PENDING upload] |
| Client 3 | **73.5%** (6: 87%, 7: 60%) | **84.0%** (16: 84%, 17: 84%) | **78.5%** (26: 63%, 27: 94%) | **92.0%** (36: 92%, 37: 92%) | **50.0%** (46: 100%, 47: 0%) [PENDING upload] | **56.0%** (56: 40%, 57: 72%) [PENDING upload] | **68.5%** (66: 70%, 67: 67%) [PENDING upload] |
| Client 4 | **86.0%** (8: 88%, 9: 84%) | **84.5%** (18: 82%, 19: 87%) | **82.0%** (28: 78%, 29: 86%) | **81.5%** (38: 100%, 39: 63%) | **92.5%** (48: 92%, 49: 93%) | **91.0%** (58: 93%, 59: 89%) | **91.0%** (68: 92%, 69: 90%) |
