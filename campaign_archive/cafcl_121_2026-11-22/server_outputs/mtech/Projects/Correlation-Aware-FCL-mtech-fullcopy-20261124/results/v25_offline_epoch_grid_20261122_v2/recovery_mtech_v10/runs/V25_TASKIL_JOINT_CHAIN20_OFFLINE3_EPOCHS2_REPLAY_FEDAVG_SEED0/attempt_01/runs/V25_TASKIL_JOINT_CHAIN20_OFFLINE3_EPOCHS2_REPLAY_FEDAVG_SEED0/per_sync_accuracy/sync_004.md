# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 43.67% on 21 observed labels.
Validation accuracy: 0.4523809523809524; seen labels never predicted: 1.
Server-accepted classes: 18; pending private classes: 3.
Accepted-only TEST accuracy: 0.47944444444444445.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **58.5%** (0: 21%, 1: 96%) [PENDING upload] | **58.0%** (5: 33%, 6: 83%) | **5.0%** (10: 4%, 11: 6%) | **49.5%** (15: 0%, 16: 99%) [PENDING upload] |
| Client 1 | **59.5%** (1: 96%, 2: 23%) | **62.5%** (6: 83%, 7: 42%) | **42.5%** (11: 6%, 12: 79%) | **73.0%** (16: 99%, 17: 47%) [PENDING upload] |
| Client 2 | **52.0%** (2: 23%, 3: 81%) | **33.5%** (7: 42%, 8: 25%) | **39.5%** (12: 79%, 13: 0%) | **35.5%** (17: 47%, 18: 24%) [PENDING upload] |
| Client 3 | **47.5%** (3: 81%, 4: 14%) | **19.5%** (8: 25%, 9: 14%) | **50.0%** (13: 0%, 14: 100%) | **27.5%** (18: 24%, 19: 31%) |
| Client 4 | **23.5%** (4: 14%, 5: 33%) | **9.0%** (9: 14%, 10: 4%) | **50.0%** (14: 100%, 15: 0%) | **63.0%** (19: 31%, 20: 95%) |
