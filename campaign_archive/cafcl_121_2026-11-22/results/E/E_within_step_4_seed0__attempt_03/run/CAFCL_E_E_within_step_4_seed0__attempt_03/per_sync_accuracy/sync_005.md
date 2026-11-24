# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 82.90% on 20 observed labels.
Validation accuracy: 0.852; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.829.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **92.5%** (0: 94%, 1: 91%) | **85.5%** (4: 95%, 5: 76%) | **79.5%** (8: 91%, 9: 68%) | **73.0%** (12: 91%, 13: 55%) | **92.5%** (16: 86%, 17: 99%) |
| Client 1 | **92.5%** (0: 94%, 1: 91%) | **85.5%** (4: 95%, 5: 76%) | **79.5%** (8: 91%, 9: 68%) | **83.5%** (14: 73%, 15: 94%) | **92.5%** (16: 86%, 17: 99%) |
| Client 2 | **85.5%** (2: 78%, 3: 93%) | **85.5%** (4: 95%, 5: 76%) | **73.5%** (10: 70%, 11: 77%) | **73.0%** (12: 91%, 13: 55%) | **92.5%** (16: 86%, 17: 99%) |
| Client 3 | **92.5%** (0: 94%, 1: 91%) | **78.0%** (6: 83%, 7: 73%) | **79.5%** (8: 91%, 9: 68%) | **73.0%** (12: 91%, 13: 55%) | **85.5%** (18: 87%, 19: 84%) |
| Client 4 | **92.5%** (0: 94%, 1: 91%) | **85.5%** (4: 95%, 5: 76%) | **79.5%** (8: 91%, 9: 68%) | **73.0%** (12: 91%, 13: 55%) | **92.5%** (16: 86%, 17: 99%) |
