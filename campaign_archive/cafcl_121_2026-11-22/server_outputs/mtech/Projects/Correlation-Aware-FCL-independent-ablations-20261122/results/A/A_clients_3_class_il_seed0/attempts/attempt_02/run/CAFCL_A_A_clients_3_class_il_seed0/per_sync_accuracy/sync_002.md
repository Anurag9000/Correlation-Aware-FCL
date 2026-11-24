# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 21.83% on 12 observed labels.
Validation accuracy: 0.17; seen labels never predicted: 3.
Server-accepted classes: 12; pending private classes: 0.
Accepted-only TEST accuracy: 0.21833333333333332.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **22.0%** (0: 24%, 1: 20%) | **2.0%** (6: 0%, 7: 4%) |
| Client 1 | **37.5%** (2: 9%, 3: 66%) | **0.0%** (8: 0%, 9: 0%) |
| Client 2 | **64.5%** (4: 71%, 5: 58%) | **5.0%** (10: 3%, 11: 7%) |
