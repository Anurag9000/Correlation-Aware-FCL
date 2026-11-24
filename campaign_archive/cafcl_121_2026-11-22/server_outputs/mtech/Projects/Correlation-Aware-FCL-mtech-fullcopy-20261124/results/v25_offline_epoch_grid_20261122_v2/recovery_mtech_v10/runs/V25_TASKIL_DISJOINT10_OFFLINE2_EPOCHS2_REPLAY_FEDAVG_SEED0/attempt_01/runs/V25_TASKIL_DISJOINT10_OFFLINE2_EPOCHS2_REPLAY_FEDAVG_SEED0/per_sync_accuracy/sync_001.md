# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.40% on 10 observed labels.
Validation accuracy: 0.62; seen labels never predicted: 1.
Server-accepted classes: 6; pending private classes: 4.
Accepted-only TEST accuracy: 0.675.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **72.5%** (0: 76%, 1: 69%) |
| Client 1 | **76.0%** (2: 88%, 3: 64%) |
| Client 2 | **44.5%** (4: 0%, 5: 89%) [PENDING upload] |
| Client 3 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] |
| Client 4 | **54.0%** (8: 98%, 9: 10%) |
