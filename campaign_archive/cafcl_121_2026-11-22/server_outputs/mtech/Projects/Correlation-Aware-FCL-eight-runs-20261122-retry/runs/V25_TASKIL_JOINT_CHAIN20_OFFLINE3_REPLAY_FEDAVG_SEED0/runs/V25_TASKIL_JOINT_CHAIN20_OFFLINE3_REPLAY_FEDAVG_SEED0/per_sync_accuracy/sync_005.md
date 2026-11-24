# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.62% on 26 observed labels.
Validation accuracy: 0.62; seen labels never predicted: 0.
Server-accepted classes: 23; pending private classes: 3.
Accepted-only TEST accuracy: 0.6908695652173913.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **48.0%** (0: 11%, 1: 85%) [PENDING upload] | **56.0%** (5: 47%, 6: 65%) | **67.0%** (10: 87%, 11: 47%) | **92.5%** (15: 96%, 16: 89%) [PENDING upload] | **61.0%** (20: 87%, 21: 35%) [PENDING upload] |
| Client 1 | **58.5%** (1: 85%, 2: 32%) | **62.5%** (6: 65%, 7: 60%) | **70.0%** (11: 47%, 12: 93%) | **80.5%** (16: 89%, 17: 72%) [PENDING upload] | **37.0%** (21: 35%, 22: 39%) [PENDING upload] |
| Client 2 | **58.5%** (2: 32%, 3: 85%) | **77.0%** (7: 60%, 8: 94%) | **53.0%** (12: 93%, 13: 13%) | **85.0%** (17: 72%, 18: 98%) | **50.0%** (22: 39%, 23: 61%) |
| Client 3 | **92.0%** (3: 85%, 4: 99%) | **67.5%** (8: 94%, 9: 41%) | **23.0%** (13: 13%, 14: 33%) | **65.5%** (18: 98%, 19: 33%) | **34.5%** (23: 61%, 24: 8%) |
| Client 4 | **73.0%** (4: 99%, 5: 47%) | **64.0%** (9: 41%, 10: 87%) | **64.5%** (14: 33%, 15: 96%) | **60.0%** (19: 33%, 20: 87%) | **50.0%** (24: 8%, 25: 92%) |
