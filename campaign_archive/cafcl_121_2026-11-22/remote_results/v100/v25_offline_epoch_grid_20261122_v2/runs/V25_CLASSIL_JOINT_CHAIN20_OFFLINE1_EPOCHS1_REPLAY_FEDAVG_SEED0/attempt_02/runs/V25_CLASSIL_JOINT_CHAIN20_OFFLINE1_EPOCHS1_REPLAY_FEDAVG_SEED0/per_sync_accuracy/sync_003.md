# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 14.56% on 16 observed labels.
Validation accuracy: 0.165; seen labels never predicted: 4.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.145625.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **54.0%** (0: 100%, 1: 8%) | **1.0%** (5: 0%, 6: 2%) | **11.0%** (10: 22%, 11: 0%) |
| Client 1 | **6.0%** (1: 8%, 2: 4%) | **2.0%** (6: 2%, 7: 2%) | **0.0%** (11: 0%, 12: 0%) |
| Client 2 | **3.0%** (2: 4%, 3: 2%) | **5.0%** (7: 2%, 8: 8%) | **10.0%** (12: 0%, 13: 20%) |
| Client 3 | **20.5%** (3: 2%, 4: 39%) | **4.0%** (8: 8%, 9: 0%) | **23.0%** (13: 20%, 14: 26%) |
| Client 4 | **19.5%** (4: 39%, 5: 0%) | **11.0%** (9: 0%, 10: 22%) | **13.0%** (14: 26%, 15: 0%) |
