# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 25.64% on 11 observed labels.
Validation accuracy: 0.2309090909090909; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.25636363636363635.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **52.5%** (0: 59%, 1: 46%) | **6.0%** (5: 9%, 6: 3%) |
| Client 1 | **41.0%** (1: 46%, 2: 36%) | **5.5%** (6: 3%, 7: 8%) |
| Client 2 | **52.0%** (2: 36%, 3: 68%) | **4.0%** (7: 8%, 8: 0%) |
| Client 3 | **53.5%** (3: 68%, 4: 39%) | **5.0%** (8: 0%, 9: 10%) |
| Client 4 | **24.0%** (4: 39%, 5: 9%) | **7.0%** (9: 10%, 10: 4%) |
