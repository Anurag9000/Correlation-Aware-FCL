# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.18% on 11 observed labels.
Validation accuracy: 0.21454545454545454; seen labels never predicted: 3.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.22181818181818183.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **39.0%** (0: 64%, 1: 14%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **44.0%** (1: 14%, 2: 74%) | **5.0%** (6: 0%, 7: 10%) |
| Client 2 | **37.0%** (2: 74%, 3: 0%) | **5.0%** (7: 10%, 8: 0%) |
| Client 3 | **36.0%** (3: 0%, 4: 72%) | **4.0%** (8: 0%, 9: 8%) |
| Client 4 | **36.0%** (4: 72%, 5: 0%) | **5.0%** (9: 8%, 10: 2%) |
