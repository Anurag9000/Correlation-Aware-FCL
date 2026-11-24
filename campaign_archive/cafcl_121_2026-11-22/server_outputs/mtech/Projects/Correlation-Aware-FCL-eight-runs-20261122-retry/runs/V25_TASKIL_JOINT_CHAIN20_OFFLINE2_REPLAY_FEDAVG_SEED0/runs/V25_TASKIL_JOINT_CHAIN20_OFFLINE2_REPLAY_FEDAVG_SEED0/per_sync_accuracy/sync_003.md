# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 74.81% on 16 observed labels.
Validation accuracy: 0.76125; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.765625.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **77.5%** (0: 85%, 1: 70%) | **66.5%** (5: 77%, 6: 56%) | **75.5%** (10: 99%, 11: 52%) |
| Client 1 | **82.5%** (1: 70%, 2: 95%) | **65.5%** (6: 56%, 7: 75%) | **70.0%** (11: 52%, 12: 88%) |
| Client 2 | **96.0%** (2: 95%, 3: 97%) | **86.5%** (7: 75%, 8: 98%) | **92.5%** (12: 88%, 13: 97%) |
| Client 3 | **77.0%** (3: 97%, 4: 57%) | **54.0%** (8: 98%, 9: 10%) | **78.0%** (13: 97%, 14: 59%) |
| Client 4 | **67.0%** (4: 57%, 5: 77%) | **54.5%** (9: 10%, 10: 99%) | **70.5%** (14: 59%, 15: 82%) |
