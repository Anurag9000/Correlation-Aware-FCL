# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 53.83% on 6 observed labels.
Validation accuracy: 0.5133333333333333; seen labels never predicted: 0.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.646.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **58.5%** (0: 97%, 1: 20%) |
| Client 1 | **53.5%** (1: 20%, 2: 87%) |
| Client 2 | **43.5%** (2: 87%, 3: 0%) [PENDING upload] |
| Client 3 | **9.5%** (3: 0%, 4: 19%) [PENDING upload] |
| Client 4 | **59.5%** (4: 19%, 5: 100%) |
