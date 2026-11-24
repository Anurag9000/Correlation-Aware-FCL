# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 74.19% on 16 observed labels.
Validation accuracy: 0.75875; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.741875.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **89.0%** (0: 85%, 1: 93%) | **85.0%** (2: 72%, 3: 98%) | **57.5%** (12: 91%, 13: 24%) | **73.0%** (10: 82%, 11: 64%) |
| Client 1 | **85.0%** (2: 72%, 3: 98%) | **82.5%** (4: 88%, 5: 77%) | **89.0%** (0: 85%, 1: 93%) | **57.5%** (12: 91%, 13: 24%) |
| Client 2 | **82.5%** (4: 88%, 5: 77%) | **72.5%** (8: 92%, 9: 53%) | **73.0%** (10: 82%, 11: 64%) | **82.5%** (4: 88%, 5: 77%) |
| Client 3 | **70.5%** (6: 70%, 7: 71%) | **73.0%** (10: 82%, 11: 64%) | **72.5%** (8: 92%, 9: 53%) | **63.5%** (14: 55%, 15: 72%) |
| Client 4 | **72.5%** (8: 92%, 9: 53%) | **89.0%** (0: 85%, 1: 93%) | **82.5%** (4: 88%, 5: 77%) | **72.5%** (8: 92%, 9: 53%) |
