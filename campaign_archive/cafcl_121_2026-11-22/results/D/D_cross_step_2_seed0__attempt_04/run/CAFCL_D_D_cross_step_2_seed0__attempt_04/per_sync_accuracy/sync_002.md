# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 58.50% on 16 observed labels.
Validation accuracy: 0.58125; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.585.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **64.5%** (0: 30%, 1: 99%) | **51.5%** (8: 99%, 9: 4%) |
| Client 1 | **73.5%** (2: 47%, 3: 100%) | **51.5%** (10: 87%, 11: 16%) |
| Client 2 | **71.0%** (4: 90%, 5: 52%) | **51.0%** (12: 94%, 13: 8%) |
| Client 3 | **54.5%** (6: 9%, 7: 100%) | **50.5%** (14: 1%, 15: 100%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **71.0%** (4: 90%, 5: 52%) |
