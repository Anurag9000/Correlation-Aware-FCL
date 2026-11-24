# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 83.15% on 26 observed labels.
Validation accuracy: 0.8284615384615385; seen labels never predicted: 0.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.8315384615384616.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **83.5%** (0: 77%, 1: 90%) | **65.5%** (10: 76%, 11: 55%) | **79.0%** (14: 65%, 15: 93%) | **65.5%** (10: 76%, 11: 55%) | **84.5%** (22: 80%, 23: 89%) |
| Client 1 | **89.0%** (2: 82%, 3: 96%) | **87.0%** (4: 88%, 5: 86%) | **89.0%** (2: 82%, 3: 96%) | **91.5%** (16: 85%, 17: 98%) | **65.5%** (10: 76%, 11: 55%) |
| Client 2 | **87.0%** (4: 88%, 5: 86%) | **83.0%** (8: 89%, 9: 77%) | **76.5%** (12: 87%, 13: 66%) | **82.5%** (18: 78%, 19: 87%) | **89.0%** (2: 82%, 3: 96%) |
| Client 3 | **78.5%** (6: 67%, 7: 90%) | **76.5%** (12: 87%, 13: 66%) | **91.5%** (16: 85%, 17: 98%) | **93.5%** (20: 87%, 21: 100%) | **87.0%** (24: 91%, 25: 83%) |
| Client 4 | **83.0%** (8: 89%, 9: 77%) | **89.0%** (2: 82%, 3: 96%) | **65.5%** (10: 76%, 11: 55%) | **89.0%** (2: 82%, 3: 96%) | **93.5%** (20: 87%, 21: 100%) |
