# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 58.05% on 20 observed labels.
Validation accuracy: 0.576; seen labels never predicted: 3.
Server-accepted classes: 16; pending private classes: 4.
Accepted-only TEST accuracy: 0.600625.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **50.5%** (8: 100%, 9: 1%) | **50.5%** (14: 96%, 15: 5%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **50.0%** (18: 0%, 19: 100%) [PENDING upload] |
| Client 2 | **73.0%** (4: 88%, 5: 58%) | **84.0%** (16: 84%, 17: 84%) |
| Client 3 | **51.5%** (0: 3%, 1: 100%) | **50.0%** (10: 100%, 11: 0%) |
| Client 4 | **72.0%** (2: 46%, 3: 98%) | **49.0%** (12: 1%, 13: 97%) |
