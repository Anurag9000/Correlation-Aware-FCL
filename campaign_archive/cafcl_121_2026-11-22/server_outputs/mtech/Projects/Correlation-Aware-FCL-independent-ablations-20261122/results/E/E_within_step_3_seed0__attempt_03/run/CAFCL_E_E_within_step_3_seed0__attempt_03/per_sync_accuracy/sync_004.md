# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 83.00% on 24 observed labels.
Validation accuracy: 0.8508333333333333; seen labels never predicted: 0.
Server-accepted classes: 24; pending private classes: 0.
Accepted-only TEST accuracy: 0.83.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **89.5%** (0: 87%, 1: 92%) | **78.0%** (6: 69%, 7: 87%) | **65.0%** (12: 92%, 13: 38%) | **91.5%** (20: 84%, 21: 99%) |
| Client 1 | **89.5%** (0: 87%, 1: 92%) | **78.0%** (6: 69%, 7: 87%) | **65.0%** (12: 92%, 13: 38%) | **82.0%** (22: 72%, 23: 92%) |
| Client 2 | **86.5%** (4: 98%, 5: 75%) | **78.0%** (6: 69%, 7: 87%) | **91.5%** (16: 84%, 17: 99%) | **75.0%** (18: 81%, 19: 69%) |
| Client 3 | **89.5%** (0: 87%, 1: 92%) | **76.5%** (10: 82%, 11: 71%) | **84.0%** (14: 82%, 15: 86%) | **75.0%** (18: 81%, 19: 69%) |
| Client 4 | **89.0%** (2: 81%, 3: 97%) | **87.5%** (8: 98%, 9: 77%) | **65.0%** (12: 92%, 13: 38%) | **75.0%** (18: 81%, 19: 69%) |
