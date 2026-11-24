# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 82.11% on 18 observed labels.
Validation accuracy: 0.8511111111111112; seen labels never predicted: 0.
Server-accepted classes: 18; pending private classes: 0.
Accepted-only TEST accuracy: 0.8211111111111111.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **89.0%** (0: 95%, 1: 83%) | **89.5%** (2: 85%, 3: 94%) | **69.5%** (12: 88%, 13: 51%) | **77.0%** (10: 80%, 11: 74%) | **77.5%** (8: 89%, 9: 66%) |
| Client 1 | **89.5%** (2: 85%, 3: 94%) | **83.5%** (4: 76%, 5: 91%) | **89.0%** (0: 95%, 1: 83%) | **69.5%** (12: 88%, 13: 51%) | **79.5%** (14: 80%, 15: 79%) |
| Client 2 | **83.5%** (4: 76%, 5: 91%) | **77.5%** (8: 89%, 9: 66%) | **77.0%** (10: 80%, 11: 74%) | **83.5%** (4: 76%, 5: 91%) | **69.5%** (12: 88%, 13: 51%) |
| Client 3 | **77.5%** (6: 77%, 7: 78%) | **77.0%** (10: 80%, 11: 74%) | **77.5%** (8: 89%, 9: 66%) | **79.5%** (14: 80%, 15: 79%) | **83.5%** (4: 76%, 5: 91%) |
| Client 4 | **77.5%** (8: 89%, 9: 66%) | **89.0%** (0: 95%, 1: 83%) | **83.5%** (4: 76%, 5: 91%) | **77.5%** (8: 89%, 9: 66%) | **96.0%** (16: 95%, 17: 97%) |
