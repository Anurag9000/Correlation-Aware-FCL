# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 29.50% on 6 observed labels.
Validation accuracy: 0.25666666666666665; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.354.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **80.5%** (0: 72%, 1: 89%) |
| Client 1 | **47.5%** (1: 89%, 2: 6%) |
| Client 2 | **8.0%** (2: 6%, 3: 10%) |
| Client 3 | **5.0%** (3: 10%, 4: 0%) |
| Client 4 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] |
