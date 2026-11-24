# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 20.88% on 16 observed labels.
Validation accuracy: 0.19625; seen labels never predicted: 3.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.20875.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **45.5%** (0: 88%, 1: 3%) | **11.5%** (5: 12%, 6: 11%) | **3.0%** (10: 0%, 11: 6%) |
| Client 1 | **18.0%** (1: 3%, 2: 33%) | **5.5%** (6: 11%, 7: 0%) | **5.0%** (11: 6%, 12: 4%) |
| Client 2 | **16.5%** (2: 33%, 3: 0%) | **5.0%** (7: 0%, 8: 10%) | **6.5%** (12: 4%, 13: 9%) |
| Client 3 | **20.0%** (3: 0%, 4: 40%) | **21.0%** (8: 10%, 9: 32%) | **31.0%** (13: 9%, 14: 53%) |
| Client 4 | **26.0%** (4: 40%, 5: 12%) | **16.0%** (9: 32%, 10: 0%) | **43.0%** (14: 53%, 15: 33%) |
