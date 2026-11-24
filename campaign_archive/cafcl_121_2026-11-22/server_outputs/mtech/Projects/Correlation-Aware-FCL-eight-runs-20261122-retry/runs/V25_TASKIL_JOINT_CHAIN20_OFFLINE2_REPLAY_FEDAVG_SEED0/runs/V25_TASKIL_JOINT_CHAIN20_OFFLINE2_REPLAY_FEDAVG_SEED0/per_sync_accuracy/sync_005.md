# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.00% on 26 observed labels.
Validation accuracy: 0.7776923076923077; seen labels never predicted: 0.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.8023076923076923.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **79.0%** (0: 76%, 1: 82%) | **70.5%** (5: 60%, 6: 81%) | **70.5%** (10: 93%, 11: 48%) | **80.5%** (15: 87%, 16: 74%) | **86.0%** (20: 81%, 21: 91%) |
| Client 1 | **85.5%** (1: 82%, 2: 89%) | **78.0%** (6: 81%, 7: 75%) | **72.0%** (11: 48%, 12: 96%) | **84.0%** (16: 74%, 17: 94%) | **64.0%** (21: 91%, 22: 37%) |
| Client 2 | **93.5%** (2: 89%, 3: 98%) | **86.0%** (7: 75%, 8: 97%) | **95.0%** (12: 96%, 13: 94%) | **86.0%** (17: 94%, 18: 78%) | **67.5%** (22: 37%, 23: 98%) |
| Client 3 | **69.5%** (3: 98%, 4: 41%) | **73.5%** (8: 97%, 9: 50%) | **79.5%** (13: 94%, 14: 65%) | **74.5%** (18: 78%, 19: 71%) | **90.5%** (23: 98%, 24: 83%) |
| Client 4 | **50.5%** (4: 41%, 5: 60%) | **71.5%** (9: 50%, 10: 93%) | **76.0%** (14: 65%, 15: 87%) | **76.0%** (19: 71%, 20: 81%) | **86.0%** (24: 83%, 25: 89%) |
