# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 63.56% on 16 observed labels.
Validation accuracy: 0.6375; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.635625.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **74.5%** (0: 71%, 1: 78%) | **57.5%** (5: 81%, 6: 34%) | **53.0%** (10: 100%, 11: 6%) |
| Client 1 | **83.5%** (1: 78%, 2: 89%) | **38.0%** (6: 34%, 7: 42%) | **51.5%** (11: 6%, 12: 97%) |
| Client 2 | **76.5%** (2: 89%, 3: 64%) | **70.5%** (7: 42%, 8: 99%) | **56.5%** (12: 97%, 13: 16%) |
| Client 3 | **74.5%** (3: 64%, 4: 85%) | **59.5%** (8: 99%, 9: 20%) | **30.5%** (13: 16%, 14: 45%) |
| Client 4 | **83.0%** (4: 85%, 5: 81%) | **60.0%** (9: 20%, 10: 100%) | **67.5%** (14: 45%, 15: 90%) |
