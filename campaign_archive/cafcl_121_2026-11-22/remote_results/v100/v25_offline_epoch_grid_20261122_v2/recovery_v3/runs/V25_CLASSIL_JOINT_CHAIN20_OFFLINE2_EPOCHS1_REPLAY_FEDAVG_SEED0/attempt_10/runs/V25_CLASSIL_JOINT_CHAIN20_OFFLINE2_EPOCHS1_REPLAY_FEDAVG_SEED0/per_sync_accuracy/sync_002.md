# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.36% on 11 observed labels.
Validation accuracy: 0.22545454545454546; seen labels never predicted: 4.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.23363636363636364.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **53.5%** (0: 67%, 1: 40%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **30.5%** (1: 40%, 2: 21%) | **20.0%** (6: 0%, 7: 40%) |
| Client 2 | **10.5%** (2: 21%, 3: 0%) | **20.0%** (7: 40%, 8: 0%) |
| Client 3 | **28.0%** (3: 0%, 4: 56%) | **6.5%** (8: 0%, 9: 13%) |
| Client 4 | **28.0%** (4: 56%, 5: 0%) | **16.5%** (9: 13%, 10: 20%) |
