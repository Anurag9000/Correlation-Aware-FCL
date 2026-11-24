# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 58.75% on 20 observed labels.
Validation accuracy: 0.588; seen labels never predicted: 1.
Server-accepted classes: 10; pending private classes: 10.
Accepted-only TEST accuracy: 0.666.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **49.0%** (0: 1%, 1: 97%) [PENDING upload] | **57.0%** (10: 68%, 11: 46%) [PENDING upload] |
| Client 1 | **85.5%** (2: 77%, 3: 94%) | **49.0%** (12: 2%, 13: 96%) [PENDING upload] |
| Client 2 | **74.0%** (4: 81%, 5: 67%) | **50.5%** (14: 99%, 15: 2%) |
| Client 3 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **49.5%** (16: 99%, 17: 0%) [PENDING upload] |
| Client 4 | **66.0%** (8: 89%, 9: 43%) | **57.0%** (18: 89%, 19: 25%) |
