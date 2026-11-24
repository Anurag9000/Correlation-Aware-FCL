# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 63.27% on 30 observed labels.
Validation accuracy: 0.6513333333333333; seen labels never predicted: 2.
Server-accepted classes: 28; pending private classes: 2.
Accepted-only TEST accuracy: 0.6414285714285715.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **54.5%** (8: 100%, 9: 9%) | **70.0%** (14: 79%, 15: 61%) | **59.0%** (26: 19%, 27: 99%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) | **52.0%** (18: 9%, 19: 95%) | **49.0%** (28: 98%, 29: 0%) |
| Client 2 | **74.0%** (4: 90%, 5: 58%) | **83.5%** (16: 93%, 17: 74%) | **67.0%** (22: 58%, 23: 76%) |
| Client 3 | **62.0%** (0: 26%, 1: 98%) | **50.0%** (10: 100%, 11: 0%) | **92.5%** (20: 85%, 21: 100%) |
| Client 4 | **70.0%** (2: 41%, 3: 99%) | **64.5%** (12: 63%, 13: 66%) | **51.0%** (24: 97%, 25: 5%) [PENDING upload] |
