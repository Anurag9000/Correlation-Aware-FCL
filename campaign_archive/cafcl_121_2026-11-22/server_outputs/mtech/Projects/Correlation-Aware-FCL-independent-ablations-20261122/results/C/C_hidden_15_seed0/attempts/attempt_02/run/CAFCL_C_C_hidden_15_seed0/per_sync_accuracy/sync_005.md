# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.18% on 50 observed labels.
Validation accuracy: 0.7796; seen labels never predicted: 1.
Server-accepted classes: 40; pending private classes: 10.
Accepted-only TEST accuracy: 0.8385.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **88.5%** (8: 91%, 9: 86%) | **75.5%** (14: 66%, 15: 85%) | **81.0%** (26: 69%, 27: 93%) | **81.0%** (34: 78%, 35: 84%) | **93.0%** (48: 88%, 49: 98%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **71.0%** (18: 61%, 19: 81%) [PENDING upload] | **49.5%** (28: 54%, 29: 45%) [PENDING upload] | **22.0%** (30: 43%, 31: 1%) [PENDING upload] | **60.0%** (42: 54%, 43: 66%) [PENDING upload] |
| Client 2 | **86.0%** (4: 95%, 5: 77%) | **93.5%** (16: 90%, 17: 97%) | **86.5%** (22: 77%, 23: 96%) | **83.0%** (32: 80%, 33: 86%) | **92.5%** (46: 87%, 47: 98%) |
| Client 3 | **86.5%** (0: 81%, 1: 92%) | **78.5%** (10: 88%, 11: 69%) | **93.0%** (20: 88%, 21: 98%) | **86.0%** (38: 86%, 39: 86%) | **59.0%** (44: 97%, 45: 21%) |
| Client 4 | **88.0%** (2: 81%, 3: 95%) | **74.5%** (12: 87%, 13: 62%) | **87.0%** (24: 92%, 25: 82%) | **89.0%** (36: 90%, 37: 88%) | **75.0%** (40: 91%, 41: 59%) |
