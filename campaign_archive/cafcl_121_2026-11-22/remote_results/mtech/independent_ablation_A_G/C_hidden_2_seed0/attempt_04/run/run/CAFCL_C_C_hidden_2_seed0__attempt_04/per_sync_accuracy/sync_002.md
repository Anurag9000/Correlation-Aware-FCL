# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.45% on 20 observed labels.
Validation accuracy: 0.584; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.629375.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **52.0%** (8: 100%, 9: 4%) | **60.5%** (14: 80%, 15: 41%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **51.0%** (18: 17%, 19: 85%) [PENDING upload] |
| Client 2 | **67.5%** (4: 97%, 5: 38%) | **67.5%** (16: 35%, 17: 100%) |
| Client 3 | **79.0%** (0: 65%, 1: 93%) | **50.0%** (10: 100%, 11: 0%) |
| Client 4 | **74.5%** (2: 49%, 3: 100%) | **52.5%** (12: 99%, 13: 6%) |
