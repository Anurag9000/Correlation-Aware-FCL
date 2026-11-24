# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 54.45% on 11 observed labels.
Validation accuracy: 0.5472727272727272; seen labels never predicted: 1.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.5763636363636364.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **47.5%** (0: 65%, 1: 30%) | **16.5%** (5: 33%, 6: 0%) |
| Client 1 | **62.5%** (1: 30%, 2: 95%) | **50.0%** (6: 0%, 7: 100%) |
| Client 2 | **47.5%** (2: 95%, 3: 0%) | **83.5%** (7: 100%, 8: 67%) |
| Client 3 | **50.0%** (3: 0%, 4: 100%) | **39.5%** (8: 67%, 9: 12%) |
| Client 4 | **66.5%** (4: 100%, 5: 33%) | **54.5%** (9: 12%, 10: 97%) |
