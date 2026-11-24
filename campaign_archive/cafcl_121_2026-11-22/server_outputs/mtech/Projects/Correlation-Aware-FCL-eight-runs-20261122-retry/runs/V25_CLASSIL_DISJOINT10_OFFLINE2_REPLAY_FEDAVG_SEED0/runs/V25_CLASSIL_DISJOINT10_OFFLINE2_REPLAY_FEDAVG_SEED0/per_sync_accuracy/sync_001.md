# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 10.40% on 10 observed labels.
Validation accuracy: 0.102; seen labels never predicted: 8.
Server-accepted classes: 6; pending private classes: 4.
Accepted-only TEST accuracy: 0.17333333333333334.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) |
| Client 1 | **2.0%** (2: 0%, 3: 4%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) [PENDING upload] |
| Client 4 | **0.0%** (8: 0%, 9: 0%) |
