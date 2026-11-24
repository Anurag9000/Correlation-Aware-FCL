# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 67.86% on 21 observed labels.
Validation accuracy: 0.660952380952381; seen labels never predicted: 0.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.6785714285714286.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **72.0%** (0: 68%, 1: 76%) | **53.5%** (5: 82%, 6: 25%) | **56.5%** (10: 72%, 11: 41%) | **69.0%** (15: 99%, 16: 39%) |
| Client 1 | **75.5%** (1: 76%, 2: 75%) | **37.0%** (6: 25%, 7: 49%) | **67.0%** (11: 41%, 12: 93%) | **65.0%** (16: 39%, 17: 91%) |
| Client 2 | **81.5%** (2: 75%, 3: 88%) | **71.0%** (7: 49%, 8: 93%) | **85.5%** (12: 93%, 13: 78%) | **73.0%** (17: 91%, 18: 55%) |
| Client 3 | **69.5%** (3: 88%, 4: 51%) | **60.5%** (8: 93%, 9: 28%) | **61.5%** (13: 78%, 14: 45%) | **74.5%** (18: 55%, 19: 94%) |
| Client 4 | **66.5%** (4: 51%, 5: 82%) | **50.0%** (9: 28%, 10: 72%) | **72.0%** (14: 45%, 15: 99%) | **88.5%** (19: 94%, 20: 83%) |
