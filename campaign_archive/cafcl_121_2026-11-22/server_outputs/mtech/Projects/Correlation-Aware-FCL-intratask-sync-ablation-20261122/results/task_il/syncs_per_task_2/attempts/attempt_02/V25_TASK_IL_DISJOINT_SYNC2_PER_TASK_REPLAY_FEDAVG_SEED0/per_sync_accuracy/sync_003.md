# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 69.20% on 20 observed labels.
Validation accuracy: 0.684; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.692.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **76.5%** (0: 62%, 1: 91%) | **76.5%** (0: 62%, 1: 91%) | **61.5%** (10: 56%, 11: 67%) |
| Client 1 | **81.5%** (2: 64%, 3: 99%) | **81.5%** (2: 64%, 3: 99%) | **53.0%** (12: 97%, 13: 9%) |
| Client 2 | **81.0%** (4: 90%, 5: 72%) | **81.0%** (4: 90%, 5: 72%) | **55.5%** (14: 14%, 15: 97%) |
| Client 3 | **72.5%** (6: 56%, 7: 89%) | **72.5%** (6: 56%, 7: 89%) | **83.0%** (16: 78%, 17: 88%) |
| Client 4 | **67.0%** (8: 97%, 9: 37%) | **67.0%** (8: 97%, 9: 37%) | **60.5%** (18: 97%, 19: 24%) |
