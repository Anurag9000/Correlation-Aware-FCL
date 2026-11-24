# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 76.00% on 40 observed labels.
Validation accuracy: 0.766; seen labels never predicted: 1.
Server-accepted classes: 36; pending private classes: 4.
Accepted-only TEST accuracy: 0.7880555555555555.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **82.0%** (8: 88%, 9: 76%) | **73.5%** (14: 70%, 15: 77%) | **76.0%** (26: 55%, 27: 97%) | **76.0%** (34: 93%, 35: 59%) |
| Client 1 | **68.0%** (6: 53%, 7: 83%) | **71.5%** (18: 51%, 19: 92%) | **75.0%** (28: 77%, 29: 73%) | **93.0%** (30: 99%, 31: 87%) |
| Client 2 | **82.5%** (4: 95%, 5: 70%) | **91.5%** (16: 84%, 17: 99%) | **81.5%** (22: 67%, 23: 96%) | **72.5%** (32: 50%, 33: 95%) |
| Client 3 | **86.0%** (0: 79%, 1: 93%) | **74.0%** (10: 93%, 11: 55%) | **93.0%** (20: 88%, 21: 98%) | **76.5%** (38: 94%, 39: 59%) |
| Client 4 | **82.5%** (2: 72%, 3: 93%) | **63.5%** (12: 87%, 13: 40%) | **51.5%** (24: 100%, 25: 3%) [PENDING upload] | **50.0%** (36: 100%, 37: 0%) [PENDING upload] |
