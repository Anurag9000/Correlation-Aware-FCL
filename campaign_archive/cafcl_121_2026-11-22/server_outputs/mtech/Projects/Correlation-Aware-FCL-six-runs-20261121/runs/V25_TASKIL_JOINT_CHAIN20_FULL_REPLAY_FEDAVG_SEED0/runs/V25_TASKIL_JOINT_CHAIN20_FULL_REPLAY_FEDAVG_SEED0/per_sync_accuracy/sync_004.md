# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 76.90% on 21 observed labels.
Validation accuracy: 0.7742857142857142; seen labels never predicted: 0.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.7690476190476191.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **76.5%** (0: 79%, 1: 74%) | **76.0%** (5: 80%, 6: 72%) | **77.5%** (10: 77%, 11: 78%) | **85.5%** (15: 89%, 16: 82%) |
| Client 1 | **77.5%** (1: 74%, 2: 81%) | **62.0%** (6: 72%, 7: 52%) | **85.0%** (11: 78%, 12: 92%) | **86.5%** (16: 82%, 17: 91%) |
| Client 2 | **84.0%** (2: 81%, 3: 87%) | **73.0%** (7: 52%, 8: 94%) | **75.0%** (12: 92%, 13: 58%) | **89.0%** (17: 91%, 18: 87%) |
| Client 3 | **81.5%** (3: 87%, 4: 76%) | **74.0%** (8: 94%, 9: 54%) | **54.5%** (13: 58%, 14: 51%) | **80.5%** (18: 87%, 19: 74%) |
| Client 4 | **78.0%** (4: 76%, 5: 80%) | **65.5%** (9: 54%, 10: 77%) | **70.0%** (14: 51%, 15: 89%) | **80.5%** (19: 74%, 20: 87%) |
