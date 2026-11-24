# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.64% on 11 observed labels.
Validation accuracy: 0.19272727272727272; seen labels never predicted: 2.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.22636363636363635.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **44.5%** (0: 63%, 1: 26%) | **3.5%** (5: 4%, 6: 3%) |
| Client 1 | **23.5%** (1: 26%, 2: 21%) | **8.5%** (6: 3%, 7: 14%) |
| Client 2 | **28.5%** (2: 21%, 3: 36%) | **7.0%** (7: 14%, 8: 0%) |
| Client 3 | **56.5%** (3: 36%, 4: 77%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **40.5%** (4: 77%, 5: 4%) | **2.5%** (9: 0%, 10: 5%) |
