# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 58.00% on 30 observed labels.
Validation accuracy: 0.578; seen labels never predicted: 2.
Server-accepted classes: 28; pending private classes: 2.
Accepted-only TEST accuracy: 0.595.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **51.5%** (8: 100%, 9: 3%) | **54.5%** (14: 10%, 15: 99%) | **52.0%** (26: 4%, 27: 100%) |
| Client 1 | **51.0%** (6: 4%, 7: 98%) | **51.5%** (18: 3%, 19: 100%) | **62.0%** (28: 59%, 29: 65%) |
| Client 2 | **74.0%** (4: 86%, 5: 62%) | **62.5%** (16: 25%, 17: 100%) | **37.0%** (22: 55%, 23: 19%) [PENDING upload] |
| Client 3 | **57.0%** (0: 14%, 1: 100%) | **50.0%** (10: 100%, 11: 0%) | **94.5%** (20: 90%, 21: 99%) |
| Client 4 | **72.5%** (2: 45%, 3: 100%) | **50.0%** (12: 99%, 13: 1%) | **50.0%** (24: 0%, 25: 100%) |
