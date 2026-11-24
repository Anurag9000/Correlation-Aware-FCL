# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 76.97% on 30 observed labels.
Validation accuracy: 0.7706666666666667; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.7696666666666667.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **85.5%** (0: 74%, 1: 97%) | **61.0%** (10: 94%, 11: 28%) | **89.5%** (20: 84%, 21: 95%) |
| Client 1 | **80.5%** (2: 62%, 3: 99%) | **63.5%** (12: 95%, 13: 32%) | **86.0%** (22: 78%, 23: 94%) |
| Client 2 | **83.0%** (4: 93%, 5: 73%) | **69.5%** (14: 43%, 15: 96%) | **81.5%** (24: 77%, 25: 86%) |
| Client 3 | **74.5%** (6: 64%, 7: 85%) | **89.0%** (16: 79%, 17: 99%) | **68.5%** (26: 39%, 27: 98%) |
| Client 4 | **75.0%** (8: 93%, 9: 57%) | **74.5%** (18: 82%, 19: 67%) | **73.0%** (28: 54%, 29: 92%) |
