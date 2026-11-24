# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.16% on 31 observed labels.
Validation accuracy: 0.6180645161290322; seen labels never predicted: 0.
Server-accepted classes: 29; pending private classes: 2.
Accepted-only TEST accuracy: 0.6731034482758621.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **86.0%** (0: 97%, 1: 75%) | **57.5%** (5: 37%, 6: 78%) | **65.5%** (10: 75%, 11: 56%) | **56.0%** (15: 14%, 16: 98%) | **42.5%** (20: 81%, 21: 4%) | **97.0%** (25: 94%, 26: 100%) |
| Client 1 | **68.0%** (1: 75%, 2: 61%) | **68.5%** (6: 78%, 7: 59%) | **66.0%** (11: 56%, 12: 76%) | **91.0%** (16: 98%, 17: 84%) | **44.5%** (21: 4%, 22: 85%) | **50.0%** (26: 100%, 27: 0%) [PENDING upload] |
| Client 2 | **79.5%** (2: 61%, 3: 98%) | **73.0%** (7: 59%, 8: 87%) | **76.0%** (12: 76%, 13: 76%) | **75.0%** (17: 84%, 18: 66%) | **55.5%** (22: 85%, 23: 26%) | **5.0%** (27: 0%, 28: 10%) [PENDING upload] |
| Client 3 | **57.0%** (3: 98%, 4: 16%) | **57.0%** (8: 87%, 9: 27%) | **65.5%** (13: 76%, 14: 55%) | **79.0%** (18: 66%, 19: 92%) | **13.0%** (23: 26%, 24: 0%) | **48.5%** (28: 10%, 29: 87%) [PENDING upload] |
| Client 4 | **26.5%** (4: 16%, 5: 37%) | **51.0%** (9: 27%, 10: 75%) | **34.5%** (14: 55%, 15: 14%) | **86.5%** (19: 92%, 20: 81%) | **47.0%** (24: 0%, 25: 94%) | **84.5%** (29: 87%, 30: 82%) |
