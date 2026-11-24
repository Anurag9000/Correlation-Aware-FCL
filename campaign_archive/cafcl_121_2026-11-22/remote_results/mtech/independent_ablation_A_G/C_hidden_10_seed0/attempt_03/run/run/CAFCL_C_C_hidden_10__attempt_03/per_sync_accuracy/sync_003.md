# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 76.13% on 30 observed labels.
Validation accuracy: 0.768; seen labels never predicted: 1.
Server-accepted classes: 24; pending private classes: 6.
Accepted-only TEST accuracy: 0.8104166666666667.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **80.5%** (8: 95%, 9: 66%) | **73.5%** (14: 62%, 15: 85%) | **71.0%** (26: 50%, 27: 92%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **61.0%** (18: 29%, 19: 93%) [PENDING upload] | **58.5%** (28: 64%, 29: 53%) [PENDING upload] |
| Client 2 | **83.5%** (4: 92%, 5: 75%) | **92.5%** (16: 86%, 17: 99%) | **82.0%** (22: 72%, 23: 92%) |
| Client 3 | **85.5%** (0: 89%, 1: 82%) | **72.0%** (10: 87%, 11: 57%) | **93.0%** (20: 88%, 21: 98%) |
| Client 4 | **83.0%** (2: 74%, 3: 92%) | **71.5%** (12: 88%, 13: 55%) | **84.5%** (24: 72%, 25: 97%) |
