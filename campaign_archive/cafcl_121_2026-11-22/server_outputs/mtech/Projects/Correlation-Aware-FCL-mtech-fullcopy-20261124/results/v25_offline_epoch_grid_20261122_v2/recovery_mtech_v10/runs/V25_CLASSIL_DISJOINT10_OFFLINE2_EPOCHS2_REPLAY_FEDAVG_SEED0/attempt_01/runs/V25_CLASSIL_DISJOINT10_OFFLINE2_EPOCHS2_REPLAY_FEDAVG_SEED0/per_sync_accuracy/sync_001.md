# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 14.80% on 10 observed labels.
Validation accuracy: 0.16; seen labels never predicted: 7.
Server-accepted classes: 6; pending private classes: 4.
Accepted-only TEST accuracy: 0.24666666666666667.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **71.0%** (0: 67%, 1: 75%) |
| Client 1 | **3.0%** (2: 6%, 3: 0%) |
| Client 2 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] |
| Client 3 | **0.0%** (6: 0%, 7: 0%) [PENDING upload] |
| Client 4 | **0.0%** (8: 0%, 9: 0%) |
