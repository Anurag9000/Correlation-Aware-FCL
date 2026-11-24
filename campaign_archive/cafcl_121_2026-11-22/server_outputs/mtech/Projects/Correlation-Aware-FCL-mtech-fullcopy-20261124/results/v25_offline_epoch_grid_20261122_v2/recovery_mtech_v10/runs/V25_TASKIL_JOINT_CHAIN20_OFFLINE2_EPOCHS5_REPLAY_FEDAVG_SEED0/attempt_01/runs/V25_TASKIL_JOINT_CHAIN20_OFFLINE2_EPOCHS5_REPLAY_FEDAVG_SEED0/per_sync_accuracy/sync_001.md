# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 55.00% on 6 observed labels.
Validation accuracy: 0.58; seen labels never predicted: 0.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.708.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **75.0%** (0: 99%, 1: 51%) |
| Client 1 | **44.5%** (1: 51%, 2: 38%) |
| Client 2 | **19.0%** (2: 38%, 3: 0%) [PENDING upload] |
| Client 3 | **42.5%** (3: 0%, 4: 85%) [PENDING upload] |
| Client 4 | **71.0%** (4: 85%, 5: 57%) |
