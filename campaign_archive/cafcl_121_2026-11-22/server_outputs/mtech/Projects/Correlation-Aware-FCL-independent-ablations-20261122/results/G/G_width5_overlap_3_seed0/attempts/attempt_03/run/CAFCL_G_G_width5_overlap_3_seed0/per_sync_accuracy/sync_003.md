# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.15% on 39 observed labels.
Validation accuracy: 0.6015384615384616; seen labels never predicted: 0.
Server-accepted classes: 39; pending private classes: 0.
Accepted-only TEST accuracy: 0.6115384615384616.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **54.6%** (4: 5%, 7: 70%, 15: 45%, 58: 61%, 63: 92%) | **42.8%** (10: 0%, 26: 62%, 55: 19%, 71: 92%, 90: 41%) | **49.2%** (1: 64%, 44: 94%, 53: 51%, 64: 8%, 92: 29%) |
| Client 1 | **52.8%** (4: 5%, 40: 13%, 69: 72%, 80: 90%, 95: 84%) | **42.6%** (10: 0%, 45: 76%, 55: 19%, 61: 77%, 90: 41%) | **49.8%** (21: 85%, 22: 30%, 64: 8%, 93: 72%, 94: 54%) |
| Client 2 | **41.6%** (4: 5%, 15: 45%, 40: 13%, 58: 61%, 95: 84%) | **51.0%** (10: 0%, 19: 41%, 79: 95%, 82: 78%, 90: 41%) | **60.0%** (3: 76%, 64: 8%, 76: 94%, 88: 68%, 94: 54%) |
| Client 3 | **37.2%** (4: 5%, 16: 57%, 40: 13%, 86: 27%, 95: 84%) | **66.2%** (26: 62%, 55: 19%, 71: 92%, 91: 65%, 96: 93%) | **49.0%** (21: 85%, 53: 51%, 64: 8%, 92: 29%, 93: 72%) |
| Client 4 | **52.2%** (4: 5%, 12: 83%, 16: 57%, 18: 89%, 86: 27%) | **36.2%** (10: 0%, 19: 41%, 55: 19%, 72: 80%, 90: 41%) | **44.2%** (3: 76%, 22: 30%, 51: 53%, 64: 8%, 94: 54%) |
