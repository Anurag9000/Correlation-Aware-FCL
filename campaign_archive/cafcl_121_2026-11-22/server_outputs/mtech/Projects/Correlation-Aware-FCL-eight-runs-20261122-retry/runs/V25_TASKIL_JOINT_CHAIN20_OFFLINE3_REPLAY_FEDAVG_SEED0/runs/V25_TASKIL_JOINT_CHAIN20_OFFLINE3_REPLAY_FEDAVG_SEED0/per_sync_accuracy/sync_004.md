# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 62.71% on 21 observed labels.
Validation accuracy: 0.6438095238095238; seen labels never predicted: 0.
Server-accepted classes: 18; pending private classes: 3.
Accepted-only TEST accuracy: 0.6972222222222222.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **52.0%** (0: 15%, 1: 89%) [PENDING upload] | **49.0%** (5: 51%, 6: 47%) | **65.5%** (10: 95%, 11: 36%) | **60.0%** (15: 33%, 16: 87%) [PENDING upload] |
| Client 1 | **63.5%** (1: 89%, 2: 38%) | **50.0%** (6: 47%, 7: 53%) | **63.0%** (11: 36%, 12: 90%) | **72.0%** (16: 87%, 17: 57%) [PENDING upload] |
| Client 2 | **54.5%** (2: 38%, 3: 71%) | **74.5%** (7: 53%, 8: 96%) | **90.0%** (12: 90%, 13: 90%) | **50.5%** (17: 57%, 18: 44%) [PENDING upload] |
| Client 3 | **84.5%** (3: 71%, 4: 98%) | **53.0%** (8: 96%, 9: 10%) | **69.5%** (13: 90%, 14: 49%) | **68.5%** (18: 44%, 19: 93%) |
| Client 4 | **74.5%** (4: 98%, 5: 51%) | **52.5%** (9: 10%, 10: 95%) | **41.0%** (14: 49%, 15: 33%) | **84.0%** (19: 93%, 20: 75%) |
