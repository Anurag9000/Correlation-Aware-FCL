# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 49.00% on 11 observed labels.
Validation accuracy: 0.48727272727272725; seen labels never predicted: 1.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.6488888888888888.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **47.5%** (0: 28%, 1: 67%) [PENDING upload] | **41.0%** (5: 55%, 6: 27%) [PENDING upload] |
| Client 1 | **66.0%** (1: 67%, 2: 65%) | **62.0%** (6: 27%, 7: 97%) [PENDING upload] |
| Client 2 | **33.0%** (2: 65%, 3: 1%) | **48.5%** (7: 97%, 8: 0%) |
| Client 3 | **46.5%** (3: 1%, 4: 92%) | **3.5%** (8: 0%, 9: 7%) |
| Client 4 | **73.5%** (4: 92%, 5: 55%) | **53.5%** (9: 7%, 10: 100%) |
