# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 45.79% on 34 observed labels.
Validation accuracy: 0.45529411764705885; seen labels never predicted: 0.
Server-accepted classes: 34; pending private classes: 0.
Accepted-only TEST accuracy: 0.45794117647058824.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **36.4%** (4: 17%, 7: 36%, 15: 16%, 58: 18%, 63: 95%) | **35.8%** (1: 2%, 21: 10%, 64: 73%, 93: 84%, 94: 10%) |
| Client 1 | **44.4%** (16: 4%, 69: 29%, 80: 97%, 86: 6%, 95: 86%) | **36.4%** (10: 2%, 19: 57%, 45: 0%, 72: 37%, 79: 86%) |
| Client 2 | **34.6%** (4: 17%, 15: 16%, 40: 25%, 69: 29%, 95: 86%) | **33.4%** (10: 2%, 45: 0%, 55: 97%, 61: 43%, 90: 25%) |
| Client 3 | **51.6%** (12: 78%, 16: 4%, 18: 85%, 26: 62%, 69: 29%) | **27.6%** (1: 2%, 45: 0%, 64: 73%, 82: 0%, 92: 63%) |
| Client 4 | **52.4%** (16: 4%, 69: 29%, 71: 83%, 91: 70%, 96: 76%) | **24.4%** (44: 84%, 45: 0%, 53: 1%, 72: 37%, 82: 0%) |
