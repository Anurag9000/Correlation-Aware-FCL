# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.17% on 6 observed labels.
Validation accuracy: 0.5233333333333333; seen labels never predicted: 0.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.7325.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **55.5%** (0: 12%, 1: 99%) [PENDING upload] |
| Client 1 | **69.0%** (1: 99%, 2: 39%) |
| Client 2 | **19.5%** (2: 39%, 3: 0%) [PENDING upload] |
| Client 3 | **44.0%** (3: 0%, 4: 88%) [PENDING upload] |
| Client 4 | **81.5%** (4: 88%, 5: 75%) |
