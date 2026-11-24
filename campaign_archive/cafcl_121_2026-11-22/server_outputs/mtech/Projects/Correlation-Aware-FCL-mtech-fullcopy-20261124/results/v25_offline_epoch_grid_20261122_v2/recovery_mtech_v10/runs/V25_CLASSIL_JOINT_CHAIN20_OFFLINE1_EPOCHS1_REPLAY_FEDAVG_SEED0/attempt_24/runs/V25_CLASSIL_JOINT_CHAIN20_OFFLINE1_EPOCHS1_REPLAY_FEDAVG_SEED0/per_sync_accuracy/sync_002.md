# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.09% on 11 observed labels.
Validation accuracy: 0.2109090909090909; seen labels never predicted: 2.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.2309090909090909.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **56.5%** (0: 85%, 1: 28%) | **0.5%** (5: 0%, 6: 1%) |
| Client 1 | **21.0%** (1: 28%, 2: 14%) | **6.5%** (6: 1%, 7: 12%) |
| Client 2 | **18.5%** (2: 14%, 3: 23%) | **6.0%** (7: 12%, 8: 0%) |
| Client 3 | **51.0%** (3: 23%, 4: 79%) | **4.5%** (8: 0%, 9: 9%) |
| Client 4 | **39.5%** (4: 79%, 5: 0%) | **6.0%** (9: 9%, 10: 3%) |
