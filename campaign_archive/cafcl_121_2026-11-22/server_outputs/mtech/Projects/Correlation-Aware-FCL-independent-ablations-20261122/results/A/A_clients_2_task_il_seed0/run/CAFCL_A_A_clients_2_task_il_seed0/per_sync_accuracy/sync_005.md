# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 85.10% on 20 observed labels.
Validation accuracy: 0.859; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.851.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **86.5%** (0: 89%, 1: 84%) | **88.0%** (4: 96%, 5: 80%) | **88.5%** (8: 90%, 9: 87%) | **82.5%** (12: 88%, 13: 77%) | **94.5%** (16: 91%, 17: 98%) |
| Client 1 | **91.5%** (2: 86%, 3: 97%) | **81.0%** (6: 74%, 7: 88%) | **72.5%** (10: 60%, 11: 85%) | **82.5%** (14: 76%, 15: 89%) | **83.5%** (18: 81%, 19: 86%) |
