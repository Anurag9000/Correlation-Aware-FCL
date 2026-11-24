# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.10% on 30 observed labels.
Validation accuracy: 0.7713333333333333; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.771.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **83.0%** (0: 68%, 1: 98%) | **59.5%** (10: 91%, 11: 28%) | **93.0%** (20: 86%, 21: 100%) |
| Client 1 | **86.0%** (2: 73%, 3: 99%) | **61.0%** (12: 96%, 13: 26%) | **88.5%** (22: 89%, 23: 88%) |
| Client 2 | **85.5%** (4: 92%, 5: 79%) | **72.5%** (14: 49%, 15: 96%) | **80.5%** (24: 85%, 25: 76%) |
| Client 3 | **68.5%** (6: 47%, 7: 90%) | **86.5%** (16: 76%, 17: 97%) | **68.0%** (26: 37%, 27: 99%) |
| Client 4 | **77.0%** (8: 96%, 9: 58%) | **74.5%** (18: 79%, 19: 70%) | **72.5%** (28: 59%, 29: 86%) |
