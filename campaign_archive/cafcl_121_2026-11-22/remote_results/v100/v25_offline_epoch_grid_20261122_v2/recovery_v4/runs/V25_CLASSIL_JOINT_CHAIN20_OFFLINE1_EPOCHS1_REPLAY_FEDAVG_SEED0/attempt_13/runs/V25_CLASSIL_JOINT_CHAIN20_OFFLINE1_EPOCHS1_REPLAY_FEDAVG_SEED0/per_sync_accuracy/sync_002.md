# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 24.64% on 11 observed labels.
Validation accuracy: 0.2309090909090909; seen labels never predicted: 2.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.24636363636363637.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **41.0%** (0: 65%, 1: 17%) | **5.0%** (5: 0%, 6: 10%) |
| Client 1 | **36.0%** (1: 17%, 2: 55%) | **7.5%** (6: 10%, 7: 5%) |
| Client 2 | **50.0%** (2: 55%, 3: 45%) | **2.5%** (7: 5%, 8: 0%) |
| Client 3 | **55.0%** (3: 45%, 4: 65%) | **4.0%** (8: 0%, 9: 8%) |
| Client 4 | **32.5%** (4: 65%, 5: 0%) | **4.5%** (9: 8%, 10: 1%) |
