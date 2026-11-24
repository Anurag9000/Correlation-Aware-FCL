# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 29.62% on 16 observed labels.
Validation accuracy: 0.27; seen labels never predicted: 1.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.29625.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **42.5%** (0: 46%, 1: 39%) | **39.0%** (5: 46%, 6: 32%) | **6.5%** (10: 0%, 11: 13%) |
| Client 1 | **42.0%** (1: 39%, 2: 45%) | **49.0%** (6: 32%, 7: 66%) | **15.5%** (11: 13%, 12: 18%) |
| Client 2 | **33.0%** (2: 45%, 3: 21%) | **63.0%** (7: 66%, 8: 60%) | **15.0%** (12: 18%, 13: 12%) |
| Client 3 | **35.5%** (3: 21%, 4: 50%) | **35.5%** (8: 60%, 9: 11%) | **13.5%** (13: 12%, 14: 15%) |
| Client 4 | **48.0%** (4: 50%, 5: 46%) | **5.5%** (9: 11%, 10: 0%) | **7.5%** (14: 15%, 15: 0%) |
