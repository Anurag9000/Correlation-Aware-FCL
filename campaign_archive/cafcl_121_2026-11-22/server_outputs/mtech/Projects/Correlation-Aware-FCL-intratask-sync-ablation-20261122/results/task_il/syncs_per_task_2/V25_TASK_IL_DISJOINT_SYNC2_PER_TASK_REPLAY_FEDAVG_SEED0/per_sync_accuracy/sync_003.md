# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 72.05% on 20 observed labels.
Validation accuracy: 0.72; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.7205.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **83.5%** (0: 73%, 1: 94%) | **83.5%** (0: 73%, 1: 94%) | **64.0%** (10: 70%, 11: 58%) |
| Client 1 | **82.5%** (2: 65%, 3: 100%) | **82.5%** (2: 65%, 3: 100%) | **59.5%** (12: 88%, 13: 31%) |
| Client 2 | **80.5%** (4: 95%, 5: 66%) | **80.5%** (4: 95%, 5: 66%) | **56.0%** (14: 14%, 15: 98%) |
| Client 3 | **74.5%** (6: 68%, 7: 81%) | **74.5%** (6: 68%, 7: 81%) | **82.5%** (16: 71%, 17: 94%) |
| Client 4 | **73.0%** (8: 97%, 9: 49%) | **73.0%** (8: 97%, 9: 49%) | **64.5%** (18: 78%, 19: 51%) |
