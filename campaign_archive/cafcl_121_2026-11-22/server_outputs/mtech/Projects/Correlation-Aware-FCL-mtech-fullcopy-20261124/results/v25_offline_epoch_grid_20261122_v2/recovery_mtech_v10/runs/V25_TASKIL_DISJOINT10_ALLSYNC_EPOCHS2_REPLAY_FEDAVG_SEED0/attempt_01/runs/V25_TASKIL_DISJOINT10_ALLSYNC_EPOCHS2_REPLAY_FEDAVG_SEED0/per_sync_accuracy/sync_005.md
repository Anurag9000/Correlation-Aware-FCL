# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.70% on 50 observed labels.
Validation accuracy: 0.7252; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.737.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **84.0%** (0: 73%, 1: 95%) | **63.0%** (10: 71%, 11: 55%) | **92.0%** (20: 86%, 21: 98%) | **86.5%** (30: 97%, 31: 76%) | **59.0%** (40: 84%, 41: 34%) |
| Client 1 | **86.5%** (2: 87%, 3: 86%) | **59.5%** (12: 89%, 13: 30%) | **74.5%** (22: 60%, 23: 89%) | **77.5%** (32: 69%, 33: 86%) | **63.5%** (42: 97%, 43: 30%) |
| Client 2 | **75.5%** (4: 78%, 5: 73%) | **70.0%** (14: 63%, 15: 77%) | **75.0%** (24: 76%, 25: 74%) | **76.0%** (34: 79%, 35: 73%) | **68.0%** (44: 77%, 45: 59%) |
| Client 3 | **55.5%** (6: 14%, 7: 97%) | **83.0%** (16: 72%, 17: 94%) | **76.5%** (26: 67%, 27: 86%) | **74.0%** (36: 62%, 37: 86%) | **80.5%** (46: 92%, 47: 69%) |
| Client 4 | **69.5%** (8: 89%, 9: 50%) | **70.5%** (18: 71%, 19: 70%) | **68.5%** (28: 83%, 29: 54%) | **75.0%** (38: 73%, 39: 77%) | **79.0%** (48: 63%, 49: 95%) |
