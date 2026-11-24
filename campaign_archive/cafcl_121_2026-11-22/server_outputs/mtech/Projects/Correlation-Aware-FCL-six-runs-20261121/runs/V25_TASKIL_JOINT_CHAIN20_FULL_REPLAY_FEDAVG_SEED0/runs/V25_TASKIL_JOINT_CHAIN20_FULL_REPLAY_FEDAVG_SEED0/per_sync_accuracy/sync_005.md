# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 82.58% on 26 observed labels.
Validation accuracy: 0.8276923076923077; seen labels never predicted: 0.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.8257692307692308.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **82.0%** (0: 78%, 1: 86%) | **76.5%** (5: 77%, 6: 76%) | **76.5%** (10: 81%, 11: 72%) | **80.5%** (15: 76%, 16: 85%) | **93.0%** (20: 90%, 21: 96%) |
| Client 1 | **80.0%** (1: 86%, 2: 74%) | **74.5%** (6: 76%, 7: 73%) | **81.0%** (11: 72%, 12: 90%) | **91.0%** (16: 85%, 17: 97%) | **85.0%** (21: 96%, 22: 74%) |
| Client 2 | **78.0%** (2: 74%, 3: 82%) | **83.0%** (7: 73%, 8: 93%) | **84.0%** (12: 90%, 13: 78%) | **91.0%** (17: 97%, 18: 85%) | **84.0%** (22: 74%, 23: 94%) |
| Client 3 | **82.0%** (3: 82%, 4: 82%) | **83.0%** (8: 93%, 9: 73%) | **78.5%** (13: 78%, 14: 79%) | **82.0%** (18: 85%, 19: 79%) | **89.5%** (23: 94%, 24: 85%) |
| Client 4 | **79.5%** (4: 82%, 5: 77%) | **77.0%** (9: 73%, 10: 81%) | **77.5%** (14: 79%, 15: 76%) | **84.5%** (19: 79%, 20: 90%) | **88.5%** (24: 85%, 25: 92%) |
