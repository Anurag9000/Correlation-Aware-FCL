# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 83.74% on 34 observed labels.
Validation accuracy: 0.84; seen labels never predicted: 0.
Server-accepted classes: 34; pending private classes: 0.
Accepted-only TEST accuracy: 0.8373529411764706.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **79.0%** (0: 64%, 1: 94%) | **86.5%** (8: 94%, 9: 79%) | **94.5%** (16: 91%, 17: 98%) | **79.5%** (22: 61%, 23: 98%) | **94.0%** (20: 88%, 21: 100%) |
| Client 1 | **89.5%** (2: 91%, 3: 88%) | **70.0%** (10: 86%, 11: 54%) | **86.5%** (8: 94%, 9: 79%) | **94.5%** (16: 91%, 17: 98%) | **77.5%** (28: 58%, 29: 97%) |
| Client 2 | **88.5%** (4: 94%, 5: 83%) | **74.0%** (12: 90%, 13: 58%) | **84.5%** (18: 87%, 19: 82%) | **94.0%** (20: 88%, 21: 100%) | **94.0%** (30: 96%, 31: 92%) |
| Client 3 | **81.0%** (6: 70%, 7: 92%) | **82.0%** (14: 71%, 15: 93%) | **94.0%** (20: 88%, 21: 100%) | **85.5%** (24: 91%, 25: 80%) | **79.5%** (22: 61%, 23: 98%) |
| Client 4 | **86.5%** (8: 94%, 9: 79%) | **88.5%** (4: 94%, 5: 83%) | **74.0%** (12: 90%, 13: 58%) | **80.0%** (26: 70%, 27: 90%) | **83.5%** (32: 84%, 33: 83%) |
