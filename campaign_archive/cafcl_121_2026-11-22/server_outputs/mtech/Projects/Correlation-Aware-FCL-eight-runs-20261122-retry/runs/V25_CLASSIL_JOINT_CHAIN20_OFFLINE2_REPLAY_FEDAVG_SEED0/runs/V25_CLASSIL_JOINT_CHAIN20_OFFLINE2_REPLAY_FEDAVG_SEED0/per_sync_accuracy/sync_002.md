# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.91% on 11 observed labels.
Validation accuracy: 0.15454545454545454; seen labels never predicted: 5.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.1990909090909091.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **1.5%** (0: 0%, 1: 3%) | **20.5%** (5: 41%, 6: 0%) |
| Client 1 | **24.5%** (1: 3%, 2: 46%) | **32.5%** (6: 0%, 7: 65%) |
| Client 2 | **23.0%** (2: 46%, 3: 0%) | **32.5%** (7: 65%, 8: 0%) |
| Client 3 | **31.5%** (3: 0%, 4: 63%) | **0.0%** (8: 0%, 9: 0%) |
| Client 4 | **52.0%** (4: 63%, 5: 41%) | **0.5%** (9: 0%, 10: 1%) |
