# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.09% on 22 observed labels.
Validation accuracy: 0.7890909090909091; seen labels never predicted: 0.
Server-accepted classes: 22; pending private classes: 0.
Accepted-only TEST accuracy: 0.7809090909090909.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **81.5%** (0: 78%, 1: 85%) | **60.0%** (10: 85%, 11: 35%) | **70.5%** (14: 50%, 15: 91%) | **60.0%** (10: 85%, 11: 35%) |
| Client 1 | **89.5%** (2: 82%, 3: 97%) | **85.0%** (4: 88%, 5: 82%) | **89.5%** (2: 82%, 3: 97%) | **94.0%** (16: 90%, 17: 98%) |
| Client 2 | **85.0%** (4: 88%, 5: 82%) | **77.0%** (8: 94%, 9: 60%) | **74.0%** (12: 94%, 13: 54%) | **56.0%** (18: 12%, 19: 100%) |
| Client 3 | **78.0%** (6: 68%, 7: 88%) | **74.0%** (12: 94%, 13: 54%) | **94.0%** (16: 90%, 17: 98%) | **93.5%** (20: 89%, 21: 98%) |
| Client 4 | **77.0%** (8: 94%, 9: 60%) | **89.5%** (2: 82%, 3: 97%) | **60.0%** (10: 85%, 11: 35%) | **89.5%** (2: 82%, 3: 97%) |
