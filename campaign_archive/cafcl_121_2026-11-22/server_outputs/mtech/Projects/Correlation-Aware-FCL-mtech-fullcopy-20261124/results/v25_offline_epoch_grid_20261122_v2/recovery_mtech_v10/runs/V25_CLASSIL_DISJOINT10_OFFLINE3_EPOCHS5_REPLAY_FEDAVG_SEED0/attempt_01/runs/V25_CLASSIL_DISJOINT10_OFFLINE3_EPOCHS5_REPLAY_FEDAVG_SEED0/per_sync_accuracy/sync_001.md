# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 10.30% on 10 observed labels.
Validation accuracy: 0.104; seen labels never predicted: 6.
Server-accepted classes: 4; pending private classes: 6.
Accepted-only TEST accuracy: 0.26.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **0.0%** (0: 0%, 1: 0%) [PENDING upload] |
| Client 1 | **48.5%** (2: 0%, 3: 97%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) [PENDING upload] |
| Client 4 | **3.0%** (8: 6%, 9: 0%) |
