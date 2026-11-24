# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 14.76% on 21 observed labels.
Validation accuracy: 0.14095238095238094; seen labels never predicted: 6.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.14761904761904762.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **69.0%** (0: 99%, 1: 39%) | **4.0%** (5: 8%, 6: 0%) | **1.5%** (10: 2%, 11: 1%) | **0.5%** (15: 0%, 16: 1%) |
| Client 1 | **24.5%** (1: 39%, 2: 10%) | **1.5%** (6: 0%, 7: 3%) | **0.5%** (11: 1%, 12: 0%) | **28.5%** (16: 1%, 17: 56%) |
| Client 2 | **5.0%** (2: 10%, 3: 0%) | **17.0%** (7: 3%, 8: 31%) | **0.0%** (12: 0%, 13: 0%) | **45.0%** (17: 56%, 18: 34%) |
| Client 3 | **8.0%** (3: 0%, 4: 16%) | **18.5%** (8: 31%, 9: 6%) | **0.0%** (13: 0%, 14: 0%) | **18.5%** (18: 34%, 19: 3%) |
| Client 4 | **12.0%** (4: 16%, 5: 8%) | **4.0%** (9: 6%, 10: 2%) | **0.0%** (14: 0%, 15: 0%) | **2.0%** (19: 3%, 20: 1%) |
