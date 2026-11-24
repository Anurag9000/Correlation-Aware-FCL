# Server sync 10 — global model v10

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 88.78% on 40 observed labels.
Validation accuracy: 0.8865; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.88775.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 | U9 | U10 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **89.0%** (0: 87%, 1: 91%) | **90.5%** (4: 93%, 5: 88%) | **90.5%** (8: 95%, 9: 86%) | **83.0%** (12: 88%, 13: 78%) | **94.5%** (16: 91%, 17: 98%) | **96.0%** (20: 96%, 21: 96%) | **91.5%** (24: 94%, 25: 89%) | **90.0%** (28: 92%, 29: 88%) | **91.5%** (32: 94%, 33: 89%) | **93.0%** (36: 89%, 37: 97%) |
| Client 1 | **92.5%** (2: 94%, 3: 91%) | **79.5%** (6: 71%, 7: 88%) | **78.0%** (10: 78%, 11: 78%) | **86.0%** (14: 85%, 15: 87%) | **82.5%** (18: 85%, 19: 80%) | **90.0%** (22: 84%, 23: 96%) | **83.5%** (26: 79%, 27: 88%) | **96.0%** (30: 98%, 31: 94%) | **85.5%** (34: 81%, 35: 90%) | **92.5%** (38: 94%, 39: 91%) |
