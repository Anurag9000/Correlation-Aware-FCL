# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.29% on 38 observed labels.
Validation accuracy: 0.8121052631578948; seen labels never predicted: 0.
Server-accepted classes: 38; pending private classes: 0.
Accepted-only TEST accuracy: 0.8028947368421052.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **88.5%** (0: 90%, 1: 87%) | **71.5%** (10: 94%, 11: 49%) | **93.5%** (20: 89%, 21: 98%) | **88.5%** (0: 90%, 1: 87%) |
| Client 1 | **85.0%** (2: 72%, 3: 98%) | **70.0%** (12: 91%, 13: 49%) | **88.0%** (22: 81%, 23: 95%) | **78.5%** (32: 65%, 33: 92%) |
| Client 2 | **83.5%** (4: 95%, 5: 72%) | **75.0%** (14: 58%, 15: 92%) | **85.0%** (24: 83%, 25: 87%) | **61.5%** (34: 100%, 35: 23%) |
| Client 3 | **81.5%** (6: 79%, 7: 84%) | **92.0%** (16: 86%, 17: 98%) | **75.5%** (26: 52%, 27: 99%) | **73.0%** (36: 48%, 37: 98%) |
| Client 4 | **84.0%** (8: 89%, 9: 79%) | **78.0%** (18: 79%, 19: 77%) | **78.0%** (28: 69%, 29: 87%) | **83.5%** (38: 89%, 39: 78%) |
