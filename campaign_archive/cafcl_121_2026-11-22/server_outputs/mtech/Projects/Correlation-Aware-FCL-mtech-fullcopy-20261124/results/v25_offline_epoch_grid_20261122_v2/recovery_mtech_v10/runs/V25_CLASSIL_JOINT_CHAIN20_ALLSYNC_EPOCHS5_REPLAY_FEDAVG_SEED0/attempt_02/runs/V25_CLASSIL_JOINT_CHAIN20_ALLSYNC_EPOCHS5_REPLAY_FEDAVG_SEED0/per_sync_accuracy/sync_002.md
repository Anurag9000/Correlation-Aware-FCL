# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.00% on 11 observed labels.
Validation accuracy: 0.19090909090909092; seen labels never predicted: 5.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.19.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) | **37.5%** (5: 63%, 6: 12%) |
| Client 1 | **3.5%** (1: 0%, 2: 7%) | **18.5%** (6: 12%, 7: 25%) |
| Client 2 | **19.5%** (2: 7%, 3: 32%) | **12.5%** (7: 25%, 8: 0%) |
| Client 3 | **51.0%** (3: 32%, 4: 70%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **66.5%** (4: 70%, 5: 63%) | **0.0%** (9: 0%, 10: 0%) |
