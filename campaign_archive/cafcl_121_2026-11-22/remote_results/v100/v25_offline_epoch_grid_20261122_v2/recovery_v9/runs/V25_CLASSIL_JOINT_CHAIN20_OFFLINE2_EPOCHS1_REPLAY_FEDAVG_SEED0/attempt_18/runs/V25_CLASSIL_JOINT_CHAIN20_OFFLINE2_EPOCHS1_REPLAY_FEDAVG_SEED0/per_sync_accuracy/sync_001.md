# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 38.00% on 6 observed labels.
Validation accuracy: 0.35; seen labels never predicted: 2.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.456.
Terminal flush: False. Selected clients: [0, 1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **77.5%** (0: 83%, 1: 72%) |
| Client 1 | **48.5%** (1: 72%, 2: 25%) |
| Client 2 | **12.5%** (2: 25%, 3: 0%) [PENDING upload] |
| Client 3 | **24.0%** (3: 0%, 4: 48%) [PENDING upload] |
| Client 4 | **24.0%** (4: 48%, 5: 0%) |
