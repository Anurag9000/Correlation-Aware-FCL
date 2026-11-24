# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 76.23% on 26 observed labels.
Validation accuracy: 0.7638461538461538; seen labels never predicted: 0.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.7623076923076924.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **80.5%** (0: 81%, 1: 80%) | **58.0%** (5: 77%, 6: 39%) | **70.0%** (10: 85%, 11: 55%) | **66.0%** (15: 69%, 16: 63%) | **92.5%** (20: 89%, 21: 96%) |
| Client 1 | **78.5%** (1: 80%, 2: 77%) | **60.0%** (6: 39%, 7: 81%) | **71.0%** (11: 55%, 12: 87%) | **78.5%** (16: 63%, 17: 94%) | **72.5%** (21: 96%, 22: 49%) |
| Client 2 | **68.5%** (2: 77%, 3: 60%) | **88.5%** (7: 81%, 8: 96%) | **68.5%** (12: 87%, 13: 50%) | **89.5%** (17: 94%, 18: 85%) | **71.0%** (22: 49%, 23: 93%) |
| Client 3 | **73.5%** (3: 60%, 4: 87%) | **73.5%** (8: 96%, 9: 51%) | **64.5%** (13: 50%, 14: 79%) | **83.5%** (18: 85%, 19: 82%) | **89.5%** (23: 93%, 24: 86%) |
| Client 4 | **82.0%** (4: 87%, 5: 77%) | **68.0%** (9: 51%, 10: 85%) | **74.0%** (14: 79%, 15: 69%) | **85.5%** (19: 82%, 20: 89%) | **88.5%** (24: 86%, 25: 91%) |
