# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 23.82% on 11 observed labels.
Validation accuracy: 0.19818181818181818; seen labels never predicted: 2.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.2381818181818182.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **54.5%** (0: 59%, 1: 50%) | **0.5%** (5: 1%, 6: 0%) |
| Client 1 | **27.0%** (1: 50%, 2: 4%) | **5.5%** (6: 0%, 7: 11%) |
| Client 2 | **39.5%** (2: 4%, 3: 75%) | **5.5%** (7: 11%, 8: 0%) |
| Client 3 | **64.0%** (3: 75%, 4: 53%) | **4.5%** (8: 0%, 9: 9%) |
| Client 4 | **27.0%** (4: 53%, 5: 1%) | **4.5%** (9: 9%, 10: 0%) |
