# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.75% on 16 observed labels.
Validation accuracy: 0.8075; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.7875.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **89.5%** (0: 90%, 1: 89%) | **85.0%** (4: 91%, 5: 79%) | **71.5%** (8: 92%, 9: 51%) | **70.5%** (12: 88%, 13: 53%) |
| Client 1 | **89.5%** (0: 90%, 1: 89%) | **85.0%** (4: 91%, 5: 79%) | **71.5%** (8: 92%, 9: 51%) | **76.0%** (14: 87%, 15: 65%) |
| Client 2 | **90.5%** (2: 83%, 3: 98%) | **85.0%** (4: 91%, 5: 79%) | **66.5%** (10: 88%, 11: 45%) | **70.5%** (12: 88%, 13: 53%) |
| Client 3 | **89.5%** (0: 90%, 1: 89%) | **80.5%** (6: 76%, 7: 85%) | **71.5%** (8: 92%, 9: 51%) | **70.5%** (12: 88%, 13: 53%) |
| Client 4 | **89.5%** (0: 90%, 1: 89%) | **85.0%** (4: 91%, 5: 79%) | **71.5%** (8: 92%, 9: 51%) | **70.5%** (12: 88%, 13: 53%) |
