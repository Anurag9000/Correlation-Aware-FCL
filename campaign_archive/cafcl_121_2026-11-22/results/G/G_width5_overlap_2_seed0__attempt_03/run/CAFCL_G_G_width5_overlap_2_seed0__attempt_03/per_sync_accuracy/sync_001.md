# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 27.24% on 17 observed labels.
Validation accuracy: 0.2635294117647059; seen labels never predicted: 9.
Server-accepted classes: 17; pending private classes: 0.
Accepted-only TEST accuracy: 0.2723529411764706.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **29.2%** (4: 59%, 7: 87%, 15: 0%, 58: 0%, 63: 0%) |
| Client 1 | **20.0%** (16: 0%, 69: 0%, 80: 100%, 86: 0%, 95: 0%) |
| Client 2 | **12.8%** (4: 59%, 15: 0%, 40: 5%, 69: 0%, 95: 0%) |
| Client 3 | **22.4%** (12: 0%, 16: 0%, 18: 12%, 26: 100%, 69: 0%) |
| Client 4 | **20.0%** (16: 0%, 69: 0%, 71: 0%, 91: 100%, 96: 0%) |
