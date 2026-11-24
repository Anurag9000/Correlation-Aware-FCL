# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.33% on 6 observed labels.
Validation accuracy: 0.26; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.34.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **78.5%** (0: 74%, 1: 83%) |
| Client 1 | **43.0%** (1: 83%, 2: 3%) |
| Client 2 | **6.5%** (2: 3%, 3: 10%) |
| Client 3 | **5.0%** (3: 10%, 4: 0%) |
| Client 4 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] |
