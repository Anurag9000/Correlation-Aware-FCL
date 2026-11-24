# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 18.27% on 11 observed labels.
Validation accuracy: 0.18181818181818182; seen labels never predicted: 5.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.22333333333333333.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **31.0%** (0: 0%, 1: 62%) [PENDING upload] | **0.5%** (5: 1%, 6: 0%) [PENDING upload] |
| Client 1 | **44.5%** (1: 62%, 2: 27%) | **36.5%** (6: 0%, 7: 73%) [PENDING upload] |
| Client 2 | **13.5%** (2: 27%, 3: 0%) | **36.5%** (7: 73%, 8: 0%) |
| Client 3 | **13.0%** (3: 0%, 4: 26%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **13.5%** (4: 26%, 5: 1%) | **6.0%** (9: 0%, 10: 12%) |
