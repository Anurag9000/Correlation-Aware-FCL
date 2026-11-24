# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 70.80% on 40 observed labels.
Validation accuracy: 0.7115; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.708.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **73.0%** (0: 49%, 1: 97%) | **56.0%** (10: 67%, 11: 45%) | **90.0%** (20: 80%, 21: 100%) | **81.5%** (30: 88%, 31: 75%) |
| Client 1 | **86.0%** (2: 79%, 3: 93%) | **57.5%** (12: 87%, 13: 28%) | **67.0%** (22: 45%, 23: 89%) | **79.5%** (32: 99%, 33: 60%) |
| Client 2 | **76.0%** (4: 71%, 5: 81%) | **70.0%** (14: 59%, 15: 81%) | **74.5%** (24: 55%, 25: 94%) | **66.5%** (34: 92%, 35: 41%) |
| Client 3 | **67.0%** (6: 51%, 7: 83%) | **67.0%** (16: 36%, 17: 98%) | **73.5%** (26: 53%, 27: 94%) | **63.5%** (36: 32%, 37: 95%) |
| Client 4 | **67.0%** (8: 92%, 9: 42%) | **60.0%** (18: 48%, 19: 72%) | **67.5%** (28: 79%, 29: 56%) | **73.0%** (38: 64%, 39: 82%) |
