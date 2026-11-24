# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.35% on 40 observed labels.
Validation accuracy: 0.5255; seen labels never predicted: 7.
Server-accepted classes: 26; pending private classes: 14.
Accepted-only TEST accuracy: 0.5196153846153846.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **67.0%** (0: 40%, 1: 94%) [PENDING upload] | **54.0%** (10: 54%, 11: 54%) [PENDING upload] | **50.0%** (20: 100%, 21: 0%) [PENDING upload] | **50.0%** (30: 0%, 31: 100%) [PENDING upload] |
| Client 1 | **60.0%** (2: 20%, 3: 100%) | **49.5%** (12: 96%, 13: 3%) | **50.0%** (22: 100%, 23: 0%) | **67.0%** (32: 56%, 33: 78%) [PENDING upload] |
| Client 2 | **50.0%** (4: 100%, 5: 0%) | **50.0%** (14: 0%, 15: 100%) | **33.0%** (24: 32%, 25: 34%) [PENDING upload] | **50.5%** (34: 99%, 35: 2%) [PENDING upload] |
| Client 3 | **50.0%** (6: 0%, 7: 100%) | **53.5%** (16: 7%, 17: 100%) | **50.0%** (26: 0%, 27: 100%) | **52.0%** (36: 4%, 37: 100%) |
| Client 4 | **56.0%** (8: 99%, 9: 13%) | **51.5%** (18: 7%, 19: 96%) | **52.0%** (28: 6%, 29: 98%) | **51.0%** (38: 31%, 39: 71%) |
