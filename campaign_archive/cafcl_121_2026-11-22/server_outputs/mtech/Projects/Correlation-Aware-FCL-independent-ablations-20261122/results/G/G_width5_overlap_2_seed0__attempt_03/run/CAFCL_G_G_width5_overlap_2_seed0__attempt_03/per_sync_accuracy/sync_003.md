# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 67.86% on 51 observed labels.
Validation accuracy: 0.6843137254901961; seen labels never predicted: 0.
Server-accepted classes: 51; pending private classes: 0.
Accepted-only TEST accuracy: 0.6786274509803921.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **64.0%** (4: 41%, 7: 91%, 15: 43%, 58: 65%, 63: 80%) | **77.4%** (1: 56%, 21: 90%, 64: 80%, 93: 69%, 94: 92%) | **35.2%** (22: 36%, 34: 22%, 88: 5%, 97: 95%, 98: 18%) |
| Client 1 | **71.6%** (16: 38%, 69: 59%, 80: 95%, 86: 83%, 95: 83%) | **44.4%** (10: 20%, 19: 45%, 45: 22%, 72: 52%, 79: 83%) | **65.2%** (28: 54%, 35: 74%, 36: 50%, 52: 88%, 85: 60%) |
| Client 2 | **55.2%** (4: 41%, 15: 43%, 40: 50%, 69: 59%, 95: 83%) | **51.8%** (10: 20%, 45: 22%, 55: 81%, 61: 67%, 90: 69%) | **52.0%** (34: 22%, 36: 50%, 52: 88%, 78: 95%, 88: 5%) |
| Client 3 | **67.4%** (12: 79%, 16: 38%, 18: 73%, 26: 88%, 69: 59%) | **54.6%** (1: 56%, 45: 22%, 64: 80%, 82: 64%, 92: 51%) | **77.0%** (28: 54%, 30: 94%, 60: 92%, 65: 85%, 85: 60%) |
| Client 4 | **73.0%** (16: 38%, 69: 59%, 71: 91%, 91: 87%, 96: 90%) | **61.4%** (44: 84%, 45: 22%, 53: 85%, 72: 52%, 82: 64%) | **57.6%** (3: 91%, 22: 36%, 51: 64%, 76: 92%, 88: 5%) |
