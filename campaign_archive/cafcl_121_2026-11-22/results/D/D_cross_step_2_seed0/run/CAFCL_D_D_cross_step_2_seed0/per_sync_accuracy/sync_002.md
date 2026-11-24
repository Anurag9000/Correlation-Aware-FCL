# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.44% on 16 observed labels.
Validation accuracy: 0.59875; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.594375.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **58.0%** (0: 16%, 1: 100%) | **51.5%** (8: 99%, 9: 4%) |
| Client 1 | **64.0%** (2: 28%, 3: 100%) | **52.0%** (10: 97%, 11: 7%) |
| Client 2 | **71.5%** (4: 90%, 5: 53%) | **50.0%** (12: 97%, 13: 3%) |
| Client 3 | **57.5%** (6: 22%, 7: 93%) | **71.0%** (14: 45%, 15: 97%) |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **71.5%** (4: 90%, 5: 53%) |
