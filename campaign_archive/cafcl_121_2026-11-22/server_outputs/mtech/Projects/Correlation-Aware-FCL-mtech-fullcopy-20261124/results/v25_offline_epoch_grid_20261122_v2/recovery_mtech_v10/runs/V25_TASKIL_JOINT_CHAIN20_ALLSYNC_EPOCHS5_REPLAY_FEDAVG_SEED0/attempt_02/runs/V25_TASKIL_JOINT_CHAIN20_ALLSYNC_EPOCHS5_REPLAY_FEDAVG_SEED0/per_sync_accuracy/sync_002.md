# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 58.00% on 11 observed labels.
Validation accuracy: 0.5254545454545455; seen labels never predicted: 0.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.58.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **72.5%** (0: 73%, 1: 72%) | **54.5%** (5: 88%, 6: 21%) |
| Client 1 | **71.0%** (1: 72%, 2: 70%) | **31.5%** (6: 21%, 7: 42%) |
| Client 2 | **58.5%** (2: 70%, 3: 47%) | **71.0%** (7: 42%, 8: 100%) |
| Client 3 | **64.0%** (3: 47%, 4: 81%) | **55.0%** (8: 100%, 9: 10%) |
| Client 4 | **84.5%** (4: 81%, 5: 88%) | **22.0%** (9: 10%, 10: 34%) |
