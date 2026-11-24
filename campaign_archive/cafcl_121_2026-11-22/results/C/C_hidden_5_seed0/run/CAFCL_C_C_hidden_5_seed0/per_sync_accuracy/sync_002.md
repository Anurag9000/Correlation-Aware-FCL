# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 62.30% on 20 observed labels.
Validation accuracy: 0.614; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.64125.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **50.5%** (8: 100%, 9: 1%) | **62.0%** (14: 44%, 15: 80%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **60.0%** (18: 30%, 19: 90%) [PENDING upload] |
| Client 2 | **73.5%** (4: 74%, 5: 73%) | **73.0%** (16: 48%, 17: 98%) |
| Client 3 | **65.5%** (0: 33%, 1: 98%) | **50.0%** (10: 100%, 11: 0%) |
| Client 4 | **74.0%** (2: 48%, 3: 100%) | **64.5%** (12: 84%, 13: 45%) |
