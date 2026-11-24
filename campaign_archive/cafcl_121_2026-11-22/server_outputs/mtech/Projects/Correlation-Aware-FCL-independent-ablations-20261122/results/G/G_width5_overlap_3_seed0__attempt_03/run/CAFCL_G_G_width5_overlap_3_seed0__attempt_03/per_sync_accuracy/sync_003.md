# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.26% on 39 observed labels.
Validation accuracy: 0.6107692307692307; seen labels never predicted: 0.
Server-accepted classes: 39; pending private classes: 0.
Accepted-only TEST accuracy: 0.6125641025641025.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **56.0%** (4: 4%, 7: 77%, 15: 53%, 58: 54%, 63: 92%) | **41.2%** (10: 0%, 26: 68%, 55: 15%, 71: 91%, 90: 32%) | **44.6%** (1: 57%, 44: 97%, 53: 41%, 64: 5%, 92: 23%) |
| Client 1 | **54.0%** (4: 4%, 40: 18%, 69: 74%, 80: 92%, 95: 82%) | **39.6%** (10: 0%, 45: 71%, 55: 15%, 61: 80%, 90: 32%) | **50.8%** (21: 81%, 22: 34%, 64: 5%, 93: 71%, 94: 63%) |
| Client 2 | **42.2%** (4: 4%, 15: 53%, 40: 18%, 58: 54%, 95: 82%) | **46.0%** (10: 0%, 19: 34%, 79: 96%, 82: 68%, 90: 32%) | **61.2%** (3: 74%, 64: 5%, 76: 89%, 88: 75%, 94: 63%) |
| Client 3 | **39.0%** (4: 4%, 16: 52%, 40: 18%, 86: 39%, 95: 82%) | **68.0%** (26: 68%, 55: 15%, 71: 91%, 91: 75%, 96: 91%) | **44.2%** (21: 81%, 53: 41%, 64: 5%, 92: 23%, 93: 71%) |
| Client 4 | **53.4%** (4: 4%, 12: 81%, 16: 52%, 18: 91%, 86: 39%) | **33.2%** (10: 0%, 19: 34%, 55: 15%, 72: 85%, 90: 32%) | **48.0%** (3: 74%, 22: 34%, 51: 64%, 64: 5%, 94: 63%) |
