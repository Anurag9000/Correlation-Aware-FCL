# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 69.81% on 21 observed labels.
Validation accuracy: 0.6923809523809524; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 1.
Accepted-only TEST accuracy: 0.7145.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **79.0%** (0: 90%, 1: 68%) | **62.0%** (5: 40%, 6: 84%) | **81.0%** (10: 94%, 11: 68%) | **51.0%** (15: 70%, 16: 32%) |
| Client 1 | **80.0%** (1: 68%, 2: 92%) | **74.0%** (6: 84%, 7: 64%) | **78.0%** (11: 68%, 12: 88%) | **53.5%** (16: 32%, 17: 75%) [PENDING upload] |
| Client 2 | **96.0%** (2: 92%, 3: 100%) | **80.5%** (7: 64%, 8: 97%) | **74.5%** (12: 88%, 13: 61%) | **48.5%** (17: 75%, 18: 22%) [PENDING upload] |
| Client 3 | **70.0%** (3: 100%, 4: 40%) | **67.0%** (8: 97%, 9: 37%) | **75.5%** (13: 61%, 14: 90%) | **58.5%** (18: 22%, 19: 95%) |
| Client 4 | **40.0%** (4: 40%, 5: 40%) | **65.5%** (9: 37%, 10: 94%) | **80.0%** (14: 90%, 15: 70%) | **77.0%** (19: 95%, 20: 59%) |
