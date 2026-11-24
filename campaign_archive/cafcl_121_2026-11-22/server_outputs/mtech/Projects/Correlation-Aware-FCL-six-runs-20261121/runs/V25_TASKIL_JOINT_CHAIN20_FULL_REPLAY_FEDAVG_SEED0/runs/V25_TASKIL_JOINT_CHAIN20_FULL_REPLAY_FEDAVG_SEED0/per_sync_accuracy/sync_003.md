# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.69% on 16 observed labels.
Validation accuracy: 0.62125; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.606875.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **81.0%** (0: 72%, 1: 90%) | **40.5%** (5: 41%, 6: 40%) | **57.0%** (10: 96%, 11: 18%) |
| Client 1 | **59.5%** (1: 90%, 2: 29%) | **66.0%** (6: 40%, 7: 92%) | **57.0%** (11: 18%, 12: 96%) |
| Client 2 | **54.5%** (2: 29%, 3: 80%) | **93.5%** (7: 92%, 8: 95%) | **51.5%** (12: 96%, 13: 7%) |
| Client 3 | **87.0%** (3: 80%, 4: 94%) | **51.0%** (8: 95%, 9: 7%) | **44.0%** (13: 7%, 14: 81%) |
| Client 4 | **67.5%** (4: 94%, 5: 41%) | **51.5%** (9: 7%, 10: 96%) | **57.0%** (14: 81%, 15: 33%) |
