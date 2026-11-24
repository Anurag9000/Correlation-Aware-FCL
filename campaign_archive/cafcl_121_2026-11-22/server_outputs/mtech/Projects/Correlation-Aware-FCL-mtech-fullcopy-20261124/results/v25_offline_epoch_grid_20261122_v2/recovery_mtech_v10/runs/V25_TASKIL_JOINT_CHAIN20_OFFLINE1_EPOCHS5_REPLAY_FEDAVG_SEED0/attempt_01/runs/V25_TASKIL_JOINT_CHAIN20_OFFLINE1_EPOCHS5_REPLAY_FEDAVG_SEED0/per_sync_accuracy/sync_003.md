# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.50% on 16 observed labels.
Validation accuracy: 0.59625; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.59875.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **74.5%** (0: 77%, 1: 72%) | **34.0%** (5: 53%, 6: 15%) | **53.5%** (10: 89%, 11: 18%) |
| Client 1 | **68.5%** (1: 72%, 2: 65%) | **52.5%** (6: 15%, 7: 90%) | **55.0%** (11: 18%, 12: 92%) |
| Client 2 | **58.0%** (2: 65%, 3: 51%) | **93.5%** (7: 90%, 8: 97%) | **56.0%** (12: 92%, 13: 20%) |
| Client 3 | **66.5%** (3: 51%, 4: 82%) | **49.5%** (8: 97%, 9: 2%) | **30.0%** (13: 20%, 14: 40%) |
| Client 4 | **67.5%** (4: 82%, 5: 53%) | **45.5%** (9: 2%, 10: 89%) | **64.5%** (14: 40%, 15: 89%) |
