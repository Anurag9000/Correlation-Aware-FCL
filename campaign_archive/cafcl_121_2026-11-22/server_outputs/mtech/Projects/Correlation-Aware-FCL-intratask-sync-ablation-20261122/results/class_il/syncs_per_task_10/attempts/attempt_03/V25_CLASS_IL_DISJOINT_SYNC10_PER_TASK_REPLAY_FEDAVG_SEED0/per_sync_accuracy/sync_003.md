# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 25.50% on 10 observed labels.
Validation accuracy: 0.218; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.255.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **13.0%** (0: 5%, 1: 21%) | **13.0%** (0: 5%, 1: 21%) | **13.0%** (0: 5%, 1: 21%) |
| Client 1 | **33.5%** (2: 16%, 3: 51%) | **33.5%** (2: 16%, 3: 51%) | **33.5%** (2: 16%, 3: 51%) |
| Client 2 | **26.0%** (4: 30%, 5: 22%) | **26.0%** (4: 30%, 5: 22%) | **26.0%** (4: 30%, 5: 22%) |
| Client 3 | **7.5%** (6: 12%, 7: 3%) | **7.5%** (6: 12%, 7: 3%) | **7.5%** (6: 12%, 7: 3%) |
| Client 4 | **47.5%** (8: 67%, 9: 28%) | **47.5%** (8: 67%, 9: 28%) | **47.5%** (8: 67%, 9: 28%) |
