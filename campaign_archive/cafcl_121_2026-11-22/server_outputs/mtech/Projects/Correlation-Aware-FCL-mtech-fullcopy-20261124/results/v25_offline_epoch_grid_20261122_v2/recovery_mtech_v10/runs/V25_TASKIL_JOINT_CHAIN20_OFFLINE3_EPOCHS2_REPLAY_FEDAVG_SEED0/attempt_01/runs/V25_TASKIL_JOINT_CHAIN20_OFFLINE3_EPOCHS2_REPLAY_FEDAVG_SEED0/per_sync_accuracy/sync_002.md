# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 46.36% on 11 observed labels.
Validation accuracy: 0.47454545454545455; seen labels never predicted: 1.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.5955555555555555.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **43.5%** (0: 12%, 1: 75%) [PENDING upload] | **34.5%** (5: 26%, 6: 43%) [PENDING upload] |
| Client 1 | **80.5%** (1: 75%, 2: 86%) | **59.0%** (6: 43%, 7: 75%) [PENDING upload] |
| Client 2 | **43.5%** (2: 86%, 3: 1%) | **51.0%** (7: 75%, 8: 27%) |
| Client 3 | **33.5%** (3: 1%, 4: 66%) | **63.0%** (8: 27%, 9: 99%) |
| Client 4 | **46.0%** (4: 66%, 5: 26%) | **49.5%** (9: 99%, 10: 0%) |
