# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 42.19% on 16 observed labels.
Validation accuracy: 0.4175; seen labels never predicted: 1.
Server-accepted classes: 14; pending private classes: 2.
Accepted-only TEST accuracy: 0.5192857142857142.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **50.0%** (0: 9%, 1: 91%) [PENDING upload] | **23.5%** (5: 40%, 6: 7%) | **79.0%** (10: 100%, 11: 58%) |
| Client 1 | **60.0%** (1: 91%, 2: 29%) | **53.0%** (6: 7%, 7: 99%) | **29.0%** (11: 58%, 12: 0%) |
| Client 2 | **16.5%** (2: 29%, 3: 4%) | **54.5%** (7: 99%, 8: 10%) | **50.0%** (12: 0%, 13: 100%) |
| Client 3 | **50.5%** (3: 4%, 4: 97%) | **5.0%** (8: 10%, 9: 0%) | **50.0%** (13: 100%, 14: 0%) |
| Client 4 | **68.5%** (4: 97%, 5: 40%) | **50.0%** (9: 0%, 10: 100%) | **15.5%** (14: 0%, 15: 31%) [PENDING upload] |
