# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.73% on 11 observed labels.
Validation accuracy: 0.19454545454545455; seen labels never predicted: 4.
Server-accepted classes: 9; pending private classes: 2.
Accepted-only TEST accuracy: 0.24333333333333335.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **38.0%** (0: 0%, 1: 76%) [PENDING upload] | **0.0%** (5: 0%, 6: 0%) [PENDING upload] |
| Client 1 | **63.0%** (1: 76%, 2: 50%) | **4.5%** (6: 0%, 7: 9%) [PENDING upload] |
| Client 2 | **25.0%** (2: 50%, 3: 0%) | **4.5%** (7: 9%, 8: 0%) |
| Client 3 | **27.0%** (3: 0%, 4: 54%) | **1.5%** (8: 0%, 9: 3%) |
| Client 4 | **27.0%** (4: 54%, 5: 0%) | **14.0%** (9: 3%, 10: 25%) |
