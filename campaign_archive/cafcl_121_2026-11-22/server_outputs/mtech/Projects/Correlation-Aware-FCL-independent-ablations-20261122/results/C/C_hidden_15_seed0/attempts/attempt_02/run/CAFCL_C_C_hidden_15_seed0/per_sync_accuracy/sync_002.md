# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.70% on 20 observed labels.
Validation accuracy: 0.601; seen labels never predicted: 2.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.64625.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **51.5%** (8: 100%, 9: 3%) | **54.0%** (14: 97%, 15: 11%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **50.0%** (18: 0%, 19: 100%) [PENDING upload] |
| Client 2 | **72.5%** (4: 85%, 5: 60%) | **72.0%** (16: 44%, 17: 100%) |
| Client 3 | **71.0%** (0: 44%, 1: 98%) | **53.5%** (10: 99%, 11: 8%) |
| Client 4 | **78.5%** (2: 58%, 3: 99%) | **64.0%** (12: 85%, 13: 43%) |
