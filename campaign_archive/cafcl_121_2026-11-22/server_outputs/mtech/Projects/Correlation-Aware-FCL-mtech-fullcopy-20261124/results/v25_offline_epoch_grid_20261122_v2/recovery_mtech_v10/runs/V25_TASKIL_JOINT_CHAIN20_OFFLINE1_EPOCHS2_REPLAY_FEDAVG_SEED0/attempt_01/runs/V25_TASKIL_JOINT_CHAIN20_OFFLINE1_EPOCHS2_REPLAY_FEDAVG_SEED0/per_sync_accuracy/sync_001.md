# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.17% on 6 observed labels.
Validation accuracy: 0.59; seen labels never predicted: 0.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.692.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **79.0%** (0: 89%, 1: 69%) |
| Client 1 | **78.5%** (1: 69%, 2: 88%) |
| Client 2 | **44.0%** (2: 88%, 3: 0%) |
| Client 3 | **4.5%** (3: 0%, 4: 9%) |
| Client 4 | **54.5%** (4: 9%, 5: 100%) [PENDING upload] |
