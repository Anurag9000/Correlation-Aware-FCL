# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 26.33% on 9 observed labels.
Validation accuracy: 0.27111111111111114; seen labels never predicted: 3.
Server-accepted classes: 9; pending private classes: 0.
Accepted-only TEST accuracy: 0.2633333333333333.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **23.2%** (4: 0%, 7: 0%, 15: 17%, 58: 0%, 63: 99%) |
| Client 1 | **4.2%** (4: 0%, 7: 0%, 40: 0%, 58: 0%, 95: 21%) |
| Client 2 | **3.4%** (4: 0%, 7: 0%, 15: 17%, 40: 0%, 58: 0%) |
| Client 3 | **4.2%** (4: 0%, 40: 0%, 58: 0%, 69: 0%, 95: 21%) |
| Client 4 | **24.2%** (4: 0%, 40: 0%, 69: 0%, 80: 100%, 95: 21%) |
