# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 15.90% on 10 observed labels.
Validation accuracy: 0.154; seen labels never predicted: 2.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.159.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **5.0%** (0: 10%, 1: 0%) | **5.0%** (0: 10%, 1: 0%) |
| Client 1 | **8.0%** (2: 4%, 3: 12%) | **8.0%** (2: 4%, 3: 12%) |
| Client 2 | **43.0%** (4: 86%, 5: 0%) | **43.0%** (4: 86%, 5: 0%) |
| Client 3 | **2.5%** (6: 3%, 7: 2%) | **2.5%** (6: 3%, 7: 2%) |
| Client 4 | **21.0%** (8: 42%, 9: 0%) | **21.0%** (8: 42%, 9: 0%) |
