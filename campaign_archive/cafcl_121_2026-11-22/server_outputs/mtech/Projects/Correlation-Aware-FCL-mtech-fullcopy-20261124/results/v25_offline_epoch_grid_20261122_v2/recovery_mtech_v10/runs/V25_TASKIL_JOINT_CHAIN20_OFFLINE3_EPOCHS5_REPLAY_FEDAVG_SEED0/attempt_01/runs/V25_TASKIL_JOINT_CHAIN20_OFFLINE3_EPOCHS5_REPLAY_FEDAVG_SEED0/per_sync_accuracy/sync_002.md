# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 55.09% on 11 observed labels.
Validation accuracy: 0.5381818181818182; seen labels never predicted: 0.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.6044444444444445.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **73.0%** (0: 75%, 1: 71%) [PENDING upload] | **35.5%** (5: 34%, 6: 37%) [PENDING upload] |
| Client 1 | **83.0%** (1: 71%, 2: 95%) | **68.5%** (6: 37%, 7: 100%) [PENDING upload] |
| Client 2 | **47.5%** (2: 95%, 3: 0%) | **50.0%** (7: 100%, 8: 0%) |
| Client 3 | **47.5%** (3: 0%, 4: 95%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **64.5%** (4: 95%, 5: 34%) | **49.5%** (9: 0%, 10: 99%) |
