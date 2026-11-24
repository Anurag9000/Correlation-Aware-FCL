# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 47.31% on 16 observed labels.
Validation accuracy: 0.47625; seen labels never predicted: 0.
Server-accepted classes: 16; pending private classes: 0.
Accepted-only TEST accuracy: 0.478125.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **68.5%** (0: 42%, 1: 95%) | **20.5%** (5: 40%, 6: 1%) | **53.5%** (10: 96%, 11: 11%) |
| Client 1 | **50.5%** (1: 95%, 2: 6%) | **32.0%** (6: 1%, 7: 63%) | **55.5%** (11: 11%, 12: 100%) |
| Client 2 | **28.0%** (2: 6%, 3: 50%) | **81.5%** (7: 63%, 8: 100%) | **50.5%** (12: 100%, 13: 1%) |
| Client 3 | **73.5%** (3: 50%, 4: 97%) | **50.0%** (8: 100%, 9: 0%) | **13.0%** (13: 1%, 14: 25%) |
| Client 4 | **68.5%** (4: 97%, 5: 40%) | **48.0%** (9: 0%, 10: 96%) | **27.5%** (14: 25%, 15: 30%) |
