# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 20.56% on 16 observed labels.
Validation accuracy: 0.19375; seen labels never predicted: 4.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.205625.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **35.5%** (0: 71%, 1: 0%) | **11.0%** (5: 15%, 6: 7%) | **9.0%** (10: 0%, 11: 18%) |
| Client 1 | **2.0%** (1: 0%, 2: 4%) | **3.5%** (6: 7%, 7: 0%) | **15.0%** (11: 18%, 12: 12%) |
| Client 2 | **2.0%** (2: 4%, 3: 0%) | **14.0%** (7: 0%, 8: 28%) | **21.0%** (12: 12%, 13: 30%) |
| Client 3 | **28.0%** (3: 0%, 4: 56%) | **21.5%** (8: 28%, 9: 15%) | **48.5%** (13: 30%, 14: 67%) |
| Client 4 | **35.5%** (4: 56%, 5: 15%) | **7.5%** (9: 15%, 10: 0%) | **36.5%** (14: 67%, 15: 6%) |
