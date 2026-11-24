# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 53.80% on 20 observed labels.
Validation accuracy: 0.525; seen labels never predicted: 2.
Server-accepted classes: 10; pending private classes: 10.
Accepted-only TEST accuracy: 0.553.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **66.0%** (0: 38%, 1: 94%) [PENDING upload] | **52.0%** (10: 91%, 11: 13%) [PENDING upload] |
| Client 1 | **68.5%** (2: 37%, 3: 100%) | **52.0%** (12: 98%, 13: 6%) [PENDING upload] |
| Client 2 | **56.5%** (4: 100%, 5: 13%) | **50.0%** (14: 0%, 15: 100%) |
| Client 3 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **41.5%** (16: 63%, 17: 20%) [PENDING upload] |
| Client 4 | **51.5%** (8: 99%, 9: 4%) | **50.0%** (18: 1%, 19: 99%) |
