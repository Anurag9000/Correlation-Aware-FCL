# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.33% on 30 observed labels.
Validation accuracy: 0.7413333333333333; seen labels never predicted: 1.
Server-accepted classes: 24; pending private classes: 6.
Accepted-only TEST accuracy: 0.7958333333333333.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **83.5%** (8: 96%, 9: 71%) | **74.5%** (14: 61%, 15: 88%) | **72.0%** (26: 47%, 27: 97%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **49.0%** (18: 25%, 19: 73%) [PENDING upload] | **46.0%** (28: 56%, 29: 36%) [PENDING upload] |
| Client 2 | **78.5%** (4: 94%, 5: 63%) | **92.0%** (16: 84%, 17: 100%) | **78.0%** (22: 64%, 23: 92%) |
| Client 3 | **87.0%** (0: 91%, 1: 83%) | **72.0%** (10: 78%, 11: 66%) | **93.5%** (20: 87%, 21: 100%) |
| Client 4 | **81.5%** (2: 72%, 3: 91%) | **64.0%** (12: 87%, 13: 41%) | **78.5%** (24: 57%, 25: 100%) |
