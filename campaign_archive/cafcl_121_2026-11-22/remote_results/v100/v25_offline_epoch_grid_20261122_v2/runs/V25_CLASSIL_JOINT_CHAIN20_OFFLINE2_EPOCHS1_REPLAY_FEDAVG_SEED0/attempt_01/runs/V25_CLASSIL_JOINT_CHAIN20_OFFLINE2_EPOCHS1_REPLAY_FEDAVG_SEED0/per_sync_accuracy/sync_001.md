# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 34.83% on 6 observed labels.
Validation accuracy: 0.33666666666666667; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.418.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **78.5%** (0: 75%, 1: 82%) |
| Client 1 | **48.0%** (1: 82%, 2: 14%) |
| Client 2 | **7.0%** (2: 14%, 3: 0%) [PENDING upload] |
| Client 3 | **19.0%** (3: 0%, 4: 38%) [PENDING upload] |
| Client 4 | **19.0%** (4: 38%, 5: 0%) |
