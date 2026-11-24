# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 15.12% on 16 observed labels.
Validation accuracy: 0.175; seen labels never predicted: 5.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.15125.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **54.5%** (0: 100%, 1: 9%) | **0.0%** (5: 0%, 6: 0%) | **2.0%** (10: 4%, 11: 0%) |
| Client 1 | **6.5%** (1: 9%, 2: 4%) | **0.5%** (6: 0%, 7: 1%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **5.5%** (2: 4%, 3: 7%) | **16.0%** (7: 1%, 8: 31%) | **2.0%** (12: 0%, 13: 4%) |
| Client 3 | **18.5%** (3: 7%, 4: 30%) | **23.0%** (8: 31%, 9: 15%) | **20.5%** (13: 4%, 14: 37%) |
| Client 4 | **15.0%** (4: 30%, 5: 0%) | **9.5%** (9: 15%, 10: 4%) | **18.5%** (14: 37%, 15: 0%) |
