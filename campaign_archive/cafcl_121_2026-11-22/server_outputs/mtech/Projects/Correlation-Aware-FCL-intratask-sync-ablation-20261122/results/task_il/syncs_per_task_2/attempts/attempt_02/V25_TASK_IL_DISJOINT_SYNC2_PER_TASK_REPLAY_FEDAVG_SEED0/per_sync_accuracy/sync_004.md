# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.80% on 20 observed labels.
Validation accuracy: 0.787; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.778.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **83.0%** (0: 80%, 1: 86%) | **83.0%** (0: 80%, 1: 86%) | **67.0%** (10: 80%, 11: 54%) | **67.0%** (10: 80%, 11: 54%) |
| Client 1 | **79.0%** (2: 60%, 3: 98%) | **79.0%** (2: 60%, 3: 98%) | **67.0%** (12: 98%, 13: 36%) | **67.0%** (12: 98%, 13: 36%) |
| Client 2 | **81.0%** (4: 87%, 5: 75%) | **81.0%** (4: 87%, 5: 75%) | **79.0%** (14: 67%, 15: 91%) | **79.0%** (14: 67%, 15: 91%) |
| Client 3 | **74.0%** (6: 58%, 7: 90%) | **74.0%** (6: 58%, 7: 90%) | **91.0%** (16: 84%, 17: 98%) | **91.0%** (16: 84%, 17: 98%) |
| Client 4 | **76.5%** (8: 90%, 9: 63%) | **76.5%** (8: 90%, 9: 63%) | **80.5%** (18: 66%, 19: 95%) | **80.5%** (18: 66%, 19: 95%) |
