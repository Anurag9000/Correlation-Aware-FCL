# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.62% on 40 observed labels.
Validation accuracy: 0.7725; seen labels never predicted: 1.
Server-accepted classes: 32; pending private classes: 8.
Accepted-only TEST accuracy: 0.8265625.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **86.0%** (8: 94%, 9: 78%) | **75.5%** (14: 67%, 15: 84%) | **79.0%** (26: 67%, 27: 91%) | **77.5%** (34: 85%, 35: 70%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **63.0%** (18: 28%, 19: 98%) [PENDING upload] | **48.5%** (28: 53%, 29: 44%) [PENDING upload] | **28.5%** (30: 57%, 31: 0%) [PENDING upload] |
| Client 2 | **86.0%** (4: 91%, 5: 81%) | **93.0%** (16: 88%, 17: 98%) | **85.5%** (22: 76%, 23: 95%) | **76.0%** (32: 58%, 33: 94%) |
| Client 3 | **87.0%** (0: 94%, 1: 80%) | **76.0%** (10: 90%, 11: 62%) | **92.5%** (20: 90%, 21: 95%) | **81.0%** (38: 73%, 39: 89%) |
| Client 4 | **86.0%** (2: 80%, 3: 92%) | **66.5%** (12: 84%, 13: 49%) | **88.0%** (24: 92%, 25: 84%) | **87.0%** (36: 82%, 37: 92%) |
