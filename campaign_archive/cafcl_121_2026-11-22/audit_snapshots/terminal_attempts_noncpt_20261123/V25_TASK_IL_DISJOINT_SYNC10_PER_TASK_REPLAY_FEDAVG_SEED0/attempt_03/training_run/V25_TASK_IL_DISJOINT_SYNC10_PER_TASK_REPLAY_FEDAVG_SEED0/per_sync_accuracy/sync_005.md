# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.20% on 10 observed labels.
Validation accuracy: 0.736; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.752.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **85.5%** (0: 75%, 1: 96%) | **85.5%** (0: 75%, 1: 96%) | **85.5%** (0: 75%, 1: 96%) | **85.5%** (0: 75%, 1: 96%) | **85.5%** (0: 75%, 1: 96%) |
| Client 1 | **81.5%** (2: 63%, 3: 100%) | **81.5%** (2: 63%, 3: 100%) | **81.5%** (2: 63%, 3: 100%) | **81.5%** (2: 63%, 3: 100%) | **81.5%** (2: 63%, 3: 100%) |
| Client 2 | **77.5%** (4: 82%, 5: 73%) | **77.5%** (4: 82%, 5: 73%) | **77.5%** (4: 82%, 5: 73%) | **77.5%** (4: 82%, 5: 73%) | **77.5%** (4: 82%, 5: 73%) |
| Client 3 | **68.0%** (6: 40%, 7: 96%) | **68.0%** (6: 40%, 7: 96%) | **68.0%** (6: 40%, 7: 96%) | **68.0%** (6: 40%, 7: 96%) | **68.0%** (6: 40%, 7: 96%) |
| Client 4 | **63.5%** (8: 98%, 9: 29%) | **63.5%** (8: 98%, 9: 29%) | **63.5%** (8: 98%, 9: 29%) | **63.5%** (8: 98%, 9: 29%) | **63.5%** (8: 98%, 9: 29%) |
