# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 68.54% on 26 observed labels.
Validation accuracy: 0.703076923076923; seen labels never predicted: 0.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.715.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **80.0%** (0: 92%, 1: 68%) | **68.0%** (5: 89%, 6: 47%) | **63.0%** (10: 39%, 11: 87%) | **79.0%** (15: 95%, 16: 63%) | **87.5%** (20: 92%, 21: 83%) |
| Client 1 | **73.5%** (1: 68%, 2: 79%) | **66.0%** (6: 47%, 7: 85%) | **84.5%** (11: 87%, 12: 82%) | **76.0%** (16: 63%, 17: 89%) | **56.5%** (21: 83%, 22: 30%) |
| Client 2 | **89.5%** (2: 79%, 3: 100%) | **84.0%** (7: 85%, 8: 83%) | **88.5%** (12: 82%, 13: 95%) | **77.0%** (17: 89%, 18: 65%) | **61.5%** (22: 30%, 23: 93%) |
| Client 3 | **51.0%** (3: 100%, 4: 2%) | **51.5%** (8: 83%, 9: 20%) | **70.5%** (13: 95%, 14: 46%) | **45.5%** (18: 65%, 19: 26%) | **84.5%** (23: 93%, 24: 76%) |
| Client 4 | **45.5%** (4: 2%, 5: 89%) | **29.5%** (9: 20%, 10: 39%) | **70.5%** (14: 46%, 15: 95%) | **59.0%** (19: 26%, 20: 92%) | **66.0%** (24: 76%, 25: 56%) |
