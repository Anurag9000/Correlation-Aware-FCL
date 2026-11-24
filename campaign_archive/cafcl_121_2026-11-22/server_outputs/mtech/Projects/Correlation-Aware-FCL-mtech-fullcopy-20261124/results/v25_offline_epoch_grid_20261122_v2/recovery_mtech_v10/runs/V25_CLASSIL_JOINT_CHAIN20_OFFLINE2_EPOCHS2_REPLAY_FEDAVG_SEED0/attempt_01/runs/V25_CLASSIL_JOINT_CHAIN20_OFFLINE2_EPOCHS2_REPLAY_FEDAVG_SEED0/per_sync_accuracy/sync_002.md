# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.09% on 11 observed labels.
Validation accuracy: 0.2018181818181818; seen labels never predicted: 3.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.2309090909090909.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **58.0%** (0: 55%, 1: 61%) | **13.5%** (5: 27%, 6: 0%) |
| Client 1 | **37.0%** (1: 61%, 2: 13%) | **2.0%** (6: 0%, 7: 4%) |
| Client 2 | **6.5%** (2: 13%, 3: 0%) | **2.0%** (7: 4%, 8: 0%) |
| Client 3 | **32.0%** (3: 0%, 4: 64%) | **15.0%** (8: 0%, 9: 30%) |
| Client 4 | **45.5%** (4: 64%, 5: 27%) | **15.0%** (9: 30%, 10: 0%) |
