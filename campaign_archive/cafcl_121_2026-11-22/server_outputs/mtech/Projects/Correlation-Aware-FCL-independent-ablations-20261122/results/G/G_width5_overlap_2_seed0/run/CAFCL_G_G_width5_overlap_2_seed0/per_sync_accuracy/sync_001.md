# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 32.82% on 17 observed labels.
Validation accuracy: 0.30941176470588233; seen labels never predicted: 8.
Server-accepted classes: 17; pending private classes: 0.
Accepted-only TEST accuracy: 0.32823529411764707.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **19.0%** (4: 0%, 7: 94%, 15: 0%, 58: 0%, 63: 1%) |
| Client 1 | **20.0%** (16: 0%, 69: 0%, 80: 100%, 86: 0%, 95: 0%) |
| Client 2 | **20.0%** (4: 0%, 15: 0%, 40: 100%, 69: 0%, 95: 0%) |
| Client 3 | **32.2%** (12: 0%, 16: 0%, 18: 82%, 26: 79%, 69: 0%) |
| Client 4 | **20.4%** (16: 0%, 69: 0%, 71: 0%, 91: 100%, 96: 2%) |
