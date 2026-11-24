# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 27.83% on 6 observed labels.
Validation accuracy: 0.26; seen labels never predicted: 2.
Server-accepted classes: 4; pending private classes: 2.
Accepted-only TEST accuracy: 0.4175.
Terminal flush: False. Selected clients: [1, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **45.0%** (0: 0%, 1: 90%) [PENDING upload] |
| Client 1 | **52.5%** (1: 90%, 2: 15%) |
| Client 2 | **7.5%** (2: 15%, 3: 0%) [PENDING upload] |
| Client 3 | **0.0%** (3: 0%, 4: 0%) [PENDING upload] |
| Client 4 | **31.0%** (4: 0%, 5: 62%) |
