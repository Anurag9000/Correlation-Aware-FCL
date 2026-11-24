# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 20.25% on 16 observed labels.
Validation accuracy: 0.20125; seen labels never predicted: 3.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.2025.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **42.0%** (0: 74%, 1: 10%) | **15.0%** (5: 17%, 6: 13%) | **10.5%** (10: 0%, 11: 21%) |
| Client 1 | **10.0%** (1: 10%, 2: 10%) | **6.5%** (6: 13%, 7: 0%) | **12.5%** (11: 21%, 12: 4%) |
| Client 2 | **5.0%** (2: 10%, 3: 0%) | **22.0%** (7: 0%, 8: 44%) | **9.5%** (12: 4%, 13: 15%) |
| Client 3 | **21.0%** (3: 0%, 4: 42%) | **24.5%** (8: 44%, 9: 5%) | **32.0%** (13: 15%, 14: 49%) |
| Client 4 | **29.5%** (4: 42%, 5: 17%) | **2.5%** (9: 5%, 10: 0%) | **34.5%** (14: 49%, 15: 20%) |
