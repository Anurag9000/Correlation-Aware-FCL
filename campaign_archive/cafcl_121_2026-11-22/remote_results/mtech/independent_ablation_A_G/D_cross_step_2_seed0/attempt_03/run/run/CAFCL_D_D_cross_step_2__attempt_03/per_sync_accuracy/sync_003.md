# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.09% on 22 observed labels.
Validation accuracy: 0.76; seen labels never predicted: 0.
Server-accepted classes: 22; pending private classes: 0.
Accepted-only TEST accuracy: 0.7509090909090909.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **73.0%** (0: 50%, 1: 96%) | **62.0%** (8: 99%, 9: 25%) | **75.0%** (16: 50%, 17: 100%) |
| Client 1 | **87.5%** (2: 78%, 3: 97%) | **64.0%** (10: 72%, 11: 56%) | **62.0%** (8: 99%, 9: 25%) |
| Client 2 | **82.0%** (4: 87%, 5: 77%) | **63.0%** (12: 89%, 13: 37%) | **76.5%** (18: 87%, 19: 66%) |
| Client 3 | **74.5%** (6: 57%, 7: 92%) | **75.5%** (14: 59%, 15: 92%) | **93.0%** (20: 87%, 21: 99%) |
| Client 4 | **62.0%** (8: 99%, 9: 25%) | **82.0%** (4: 87%, 5: 77%) | **63.0%** (12: 89%, 13: 37%) |
