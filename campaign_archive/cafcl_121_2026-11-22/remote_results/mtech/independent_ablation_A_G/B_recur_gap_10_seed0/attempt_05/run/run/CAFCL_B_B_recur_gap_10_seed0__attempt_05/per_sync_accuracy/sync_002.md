# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 62.40% on 20 observed labels.
Validation accuracy: 0.62; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.624.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **75.5%** (0: 52%, 1: 99%) | **55.0%** (10: 98%, 11: 12%) |
| Client 1 | **55.5%** (2: 11%, 3: 100%) | **51.0%** (12: 2%, 13: 100%) |
| Client 2 | **63.5%** (4: 99%, 5: 28%) | **62.0%** (14: 25%, 15: 99%) |
| Client 3 | **64.0%** (6: 43%, 7: 85%) | **82.0%** (16: 81%, 17: 83%) |
| Client 4 | **52.0%** (8: 99%, 9: 5%) | **63.5%** (18: 88%, 19: 39%) |
