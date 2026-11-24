# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 22.84% on 25 observed labels.
Validation accuracy: 0.2248; seen labels never predicted: 10.
Server-accepted classes: 15; pending private classes: 10.
Accepted-only TEST accuracy: 0.24933333333333332.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **25.8%** (0: 0%, 1: 0%, 2: 15%, 3: 32%, 4: 82%) |
| Client 1 | **29.0%** (5: 18%, 6: 2%, 7: 30%, 8: 95%, 9: 0%) |
| Client 2 | **17.2%** (10: 3%, 11: 13%, 12: 15%, 13: 0%, 14: 55%) [PENDING upload] |
| Client 3 | **22.2%** (15: 0%, 16: 15%, 17: 96%, 18: 0%, 19: 0%) [PENDING upload] |
| Client 4 | **20.0%** (20: 0%, 21: 0%, 22: 0%, 23: 100%, 24: 0%) |
