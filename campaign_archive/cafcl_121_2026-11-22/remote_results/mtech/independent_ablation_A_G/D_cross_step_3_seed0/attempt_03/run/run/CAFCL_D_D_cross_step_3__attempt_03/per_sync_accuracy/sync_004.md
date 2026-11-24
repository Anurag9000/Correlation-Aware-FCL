# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.86% on 22 observed labels.
Validation accuracy: 0.8245454545454546; seen labels never predicted: 0.
Server-accepted classes: 22; pending private classes: 0.
Accepted-only TEST accuracy: 0.8086363636363636.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **85.5%** (0: 90%, 1: 81%) | **69.0%** (10: 75%, 11: 63%) | **81.0%** (14: 74%, 15: 88%) | **69.0%** (10: 75%, 11: 63%) |
| Client 1 | **88.5%** (2: 90%, 3: 87%) | **83.5%** (4: 81%, 5: 86%) | **88.5%** (2: 90%, 3: 87%) | **92.5%** (16: 95%, 17: 90%) |
| Client 2 | **83.5%** (4: 81%, 5: 86%) | **79.0%** (8: 94%, 9: 64%) | **75.0%** (12: 89%, 13: 61%) | **61.0%** (18: 22%, 19: 100%) |
| Client 3 | **78.0%** (6: 65%, 7: 91%) | **75.0%** (12: 89%, 13: 61%) | **92.5%** (16: 95%, 17: 90%) | **96.5%** (20: 95%, 21: 98%) |
| Client 4 | **79.0%** (8: 94%, 9: 64%) | **88.5%** (2: 90%, 3: 87%) | **69.0%** (10: 75%, 11: 63%) | **88.5%** (2: 90%, 3: 87%) |
