# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 84.28% on 32 observed labels.
Validation accuracy: 0.854375; seen labels never predicted: 0.
Server-accepted classes: 32; pending private classes: 0.
Accepted-only TEST accuracy: 0.8428125.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **86.0%** (2: 73%, 3: 99%) | **85.0%** (8: 95%, 9: 75%) | **81.5%** (18: 77%, 19: 86%) | **78.5%** (28: 67%, 29: 90%) |
| Client 1 | **89.5%** (0: 96%, 1: 83%) | **76.5%** (10: 81%, 11: 72%) | **95.5%** (16: 92%, 17: 99%) | **96.5%** (30: 97%, 31: 96%) |
| Client 2 | **76.5%** (6: 72%, 7: 81%) | **85.0%** (8: 95%, 9: 75%) | **88.0%** (22: 82%, 23: 94%) | **86.0%** (24: 89%, 25: 83%) |
| Client 3 | **89.5%** (0: 96%, 1: 83%) | **83.0%** (14: 71%, 15: 95%) | **93.5%** (20: 89%, 21: 98%) | **72.5%** (26: 51%, 27: 94%) |
| Client 4 | **85.5%** (4: 98%, 5: 73%) | **74.5%** (12: 91%, 13: 58%) | **95.5%** (16: 92%, 17: 99%) | **86.0%** (24: 89%, 25: 83%) |
