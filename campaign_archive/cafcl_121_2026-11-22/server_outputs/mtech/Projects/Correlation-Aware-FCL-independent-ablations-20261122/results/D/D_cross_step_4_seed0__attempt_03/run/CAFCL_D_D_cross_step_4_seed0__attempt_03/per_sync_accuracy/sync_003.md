# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.29% on 14 observed labels.
Validation accuracy: 0.7542857142857143; seen labels never predicted: 0.
Server-accepted classes: 14; pending private classes: 0.
Accepted-only TEST accuracy: 0.7328571428571429.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **86.5%** (0: 79%, 1: 94%) | **83.0%** (2: 68%, 3: 98%) | **54.0%** (12: 54%, 13: 54%) |
| Client 1 | **83.0%** (2: 68%, 3: 98%) | **80.0%** (4: 89%, 5: 71%) | **86.5%** (0: 79%, 1: 94%) |
| Client 2 | **80.0%** (4: 89%, 5: 71%) | **69.0%** (8: 94%, 9: 44%) | **70.5%** (10: 81%, 11: 60%) |
| Client 3 | **70.0%** (6: 69%, 7: 71%) | **70.5%** (10: 81%, 11: 60%) | **69.0%** (8: 94%, 9: 44%) |
| Client 4 | **69.0%** (8: 94%, 9: 44%) | **86.5%** (0: 79%, 1: 94%) | **80.0%** (4: 89%, 5: 71%) |
