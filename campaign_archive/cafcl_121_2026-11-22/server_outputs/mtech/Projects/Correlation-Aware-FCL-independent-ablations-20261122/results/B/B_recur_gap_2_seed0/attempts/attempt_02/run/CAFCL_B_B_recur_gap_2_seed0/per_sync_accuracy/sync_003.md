# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.00% on 30 observed labels.
Validation accuracy: 0.778; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.78.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **84.5%** (0: 71%, 1: 98%) | **63.0%** (10: 90%, 11: 36%) | **93.5%** (20: 87%, 21: 100%) |
| Client 1 | **76.5%** (2: 54%, 3: 99%) | **62.5%** (12: 89%, 13: 36%) | **82.5%** (22: 72%, 23: 93%) |
| Client 2 | **83.0%** (4: 93%, 5: 73%) | **75.0%** (14: 60%, 15: 90%) | **84.5%** (24: 94%, 25: 75%) |
| Client 3 | **73.0%** (6: 55%, 7: 91%) | **90.0%** (16: 83%, 17: 97%) | **74.5%** (26: 52%, 27: 97%) |
| Client 4 | **76.5%** (8: 95%, 9: 58%) | **76.0%** (18: 81%, 19: 71%) | **75.0%** (28: 58%, 29: 92%) |
