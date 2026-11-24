# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.00% on 22 observed labels.
Validation accuracy: 0.7927272727272727; seen labels never predicted: 0.
Server-accepted classes: 22; pending private classes: 0.
Accepted-only TEST accuracy: 0.78.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **82.5%** (0: 73%, 1: 92%) | **72.0%** (8: 95%, 9: 49%) | **83.0%** (16: 67%, 17: 99%) |
| Client 1 | **88.0%** (2: 79%, 3: 97%) | **68.5%** (10: 77%, 11: 60%) | **72.0%** (8: 95%, 9: 49%) |
| Client 2 | **84.0%** (4: 87%, 5: 81%) | **60.0%** (12: 89%, 13: 31%) | **72.5%** (18: 86%, 19: 59%) |
| Client 3 | **77.5%** (6: 69%, 7: 86%) | **78.5%** (14: 69%, 15: 88%) | **91.5%** (20: 84%, 21: 99%) |
| Client 4 | **72.0%** (8: 95%, 9: 49%) | **84.0%** (4: 87%, 5: 81%) | **60.0%** (12: 89%, 13: 31%) |
