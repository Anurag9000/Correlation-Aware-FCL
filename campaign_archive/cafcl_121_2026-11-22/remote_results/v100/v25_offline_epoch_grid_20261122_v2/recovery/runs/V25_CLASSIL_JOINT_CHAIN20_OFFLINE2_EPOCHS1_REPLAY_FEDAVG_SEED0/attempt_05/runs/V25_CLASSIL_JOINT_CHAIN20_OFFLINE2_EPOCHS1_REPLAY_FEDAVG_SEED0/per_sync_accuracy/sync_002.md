# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.36% on 11 observed labels.
Validation accuracy: 0.20363636363636364; seen labels never predicted: 4.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.21363636363636362.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **49.0%** (0: 70%, 1: 28%) | **0.0%** (5: 0%, 6: 0%) |
| Client 1 | **38.0%** (1: 28%, 2: 48%) | **3.0%** (6: 0%, 7: 6%) |
| Client 2 | **24.0%** (2: 48%, 3: 0%) | **3.0%** (7: 6%, 8: 0%) |
| Client 3 | **31.5%** (3: 0%, 4: 63%) | **1.0%** (8: 0%, 9: 2%) |
| Client 4 | **31.5%** (4: 63%, 5: 0%) | **10.0%** (9: 2%, 10: 18%) |
