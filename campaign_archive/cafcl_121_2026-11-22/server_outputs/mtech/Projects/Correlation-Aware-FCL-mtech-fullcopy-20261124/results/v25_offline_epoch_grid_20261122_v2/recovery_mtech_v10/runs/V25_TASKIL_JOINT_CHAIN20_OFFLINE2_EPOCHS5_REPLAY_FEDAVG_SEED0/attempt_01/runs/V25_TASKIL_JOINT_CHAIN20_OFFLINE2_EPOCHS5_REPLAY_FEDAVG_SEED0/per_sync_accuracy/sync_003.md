# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 67.38% on 16 observed labels.
Validation accuracy: 0.685; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.71125.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **70.0%** (0: 79%, 1: 61%) | **54.5%** (5: 88%, 6: 21%) | **51.0%** (10: 97%, 11: 5%) |
| Client 1 | **76.0%** (1: 61%, 2: 91%) | **55.5%** (6: 21%, 7: 90%) | **51.5%** (11: 5%, 12: 98%) |
| Client 2 | **92.5%** (2: 91%, 3: 94%) | **94.0%** (7: 90%, 8: 98%) | **76.5%** (12: 98%, 13: 55%) |
| Client 3 | **69.5%** (3: 94%, 4: 45%) | **52.5%** (8: 98%, 9: 7%) | **54.5%** (13: 55%, 14: 54%) |
| Client 4 | **66.5%** (4: 45%, 5: 88%) | **52.0%** (9: 7%, 10: 97%) | **74.5%** (14: 54%, 15: 95%) |
