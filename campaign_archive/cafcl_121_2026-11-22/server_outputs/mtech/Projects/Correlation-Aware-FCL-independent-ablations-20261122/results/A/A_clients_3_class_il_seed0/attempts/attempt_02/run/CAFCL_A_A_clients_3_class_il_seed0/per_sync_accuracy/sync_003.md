# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 33.72% on 18 observed labels.
Validation accuracy: 0.34; seen labels never predicted: 0.
Server-accepted classes: 18; pending private classes: 0.
Accepted-only TEST accuracy: 0.3372222222222222.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **55.5%** (0: 58%, 1: 53%) | **56.0%** (6: 55%, 7: 57%) | **4.5%** (12: 4%, 13: 5%) |
| Client 1 | **49.5%** (2: 40%, 3: 59%) | **63.0%** (8: 80%, 9: 46%) | **2.5%** (14: 4%, 15: 1%) |
| Client 2 | **47.5%** (4: 43%, 5: 52%) | **20.5%** (10: 28%, 11: 13%) | **4.5%** (16: 1%, 17: 8%) |
