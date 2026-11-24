# Server sync 11 — global model v11

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 89.11% on 44 observed labels.
Validation accuracy: 0.8695454545454545; seen labels never predicted: 0.
Server-accepted classes: 44; pending private classes: 0.
Accepted-only TEST accuracy: 0.8911363636363636.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 | U9 | U10 | U11 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **90.5%** (0: 89%, 1: 92%) | **91.5%** (4: 95%, 5: 88%) | **89.5%** (8: 91%, 9: 88%) | **81.5%** (12: 78%, 13: 85%) | **94.0%** (16: 93%, 17: 95%) | **96.5%** (20: 94%, 21: 99%) | **91.0%** (24: 89%, 25: 93%) | **90.0%** (28: 93%, 29: 87%) | **91.5%** (32: 94%, 33: 89%) | **91.0%** (36: 85%, 37: 97%) | **90.0%** (40: 82%, 41: 98%) |
| Client 1 | **94.5%** (2: 93%, 3: 96%) | **80.5%** (6: 74%, 7: 87%) | **79.0%** (10: 80%, 11: 78%) | **90.0%** (14: 91%, 15: 89%) | **86.5%** (18: 88%, 19: 85%) | **93.0%** (22: 93%, 23: 93%) | **86.0%** (26: 84%, 27: 88%) | **96.5%** (30: 99%, 31: 94%) | **86.5%** (34: 94%, 35: 79%) | **88.0%** (38: 88%, 39: 88%) | **83.0%** (42: 80%, 43: 86%) |
