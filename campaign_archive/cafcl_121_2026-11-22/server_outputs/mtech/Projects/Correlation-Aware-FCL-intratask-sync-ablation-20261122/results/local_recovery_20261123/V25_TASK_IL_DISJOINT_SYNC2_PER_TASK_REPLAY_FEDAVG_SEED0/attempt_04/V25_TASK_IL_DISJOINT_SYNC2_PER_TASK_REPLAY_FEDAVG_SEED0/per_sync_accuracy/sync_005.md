# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 80.67% on 30 observed labels.
Validation accuracy: 0.822; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.8066666666666666.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **83.0%** (0: 79%, 1: 87%) | **83.0%** (0: 79%, 1: 87%) | **65.5%** (10: 87%, 11: 44%) | **65.5%** (10: 87%, 11: 44%) | **94.5%** (20: 94%, 21: 95%) |
| Client 1 | **88.5%** (2: 89%, 3: 88%) | **88.5%** (2: 89%, 3: 88%) | **70.5%** (12: 98%, 13: 43%) | **70.5%** (12: 98%, 13: 43%) | **86.0%** (22: 78%, 23: 94%) |
| Client 2 | **85.5%** (4: 82%, 5: 89%) | **85.5%** (4: 82%, 5: 89%) | **76.0%** (14: 60%, 15: 92%) | **76.0%** (14: 60%, 15: 92%) | **85.5%** (24: 81%, 25: 90%) |
| Client 3 | **80.0%** (6: 79%, 7: 81%) | **80.0%** (6: 79%, 7: 81%) | **92.5%** (16: 90%, 17: 95%) | **92.5%** (16: 90%, 17: 95%) | **68.5%** (26: 88%, 27: 49%) |
| Client 4 | **81.5%** (8: 75%, 9: 88%) | **81.5%** (8: 75%, 9: 88%) | **78.5%** (18: 84%, 19: 73%) | **78.5%** (18: 84%, 19: 73%) | **74.0%** (28: 68%, 29: 80%) |
