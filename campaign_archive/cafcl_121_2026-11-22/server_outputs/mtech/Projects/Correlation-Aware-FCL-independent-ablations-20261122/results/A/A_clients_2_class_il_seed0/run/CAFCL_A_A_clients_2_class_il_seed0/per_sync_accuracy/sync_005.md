# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 43.20% on 20 observed labels.
Validation accuracy: 0.423; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.432.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **67.5%** (0: 75%, 1: 60%) | **47.0%** (4: 32%, 5: 62%) | **57.0%** (8: 57%, 9: 57%) | **57.5%** (12: 53%, 13: 62%) | **17.0%** (16: 16%, 17: 18%) |
| Client 1 | **37.0%** (2: 30%, 3: 44%) | **49.0%** (6: 45%, 7: 53%) | **40.5%** (10: 25%, 11: 56%) | **46.5%** (14: 29%, 15: 64%) | **13.0%** (18: 18%, 19: 8%) |
