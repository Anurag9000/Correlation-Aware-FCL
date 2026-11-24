# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 55.82% on 11 observed labels.
Validation accuracy: 0.5254545454545455; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.5581818181818182.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **34.0%** (0: 62%, 1: 6%) | **51.5%** (5: 100%, 6: 3%) |
| Client 1 | **34.0%** (1: 6%, 2: 62%) | **50.5%** (6: 3%, 7: 98%) |
| Client 2 | **81.0%** (2: 62%, 3: 100%) | **98.0%** (7: 98%, 8: 98%) |
| Client 3 | **50.0%** (3: 100%, 4: 0%) | **50.5%** (8: 98%, 9: 3%) |
| Client 4 | **50.0%** (4: 0%, 5: 100%) | **42.5%** (9: 3%, 10: 82%) |
