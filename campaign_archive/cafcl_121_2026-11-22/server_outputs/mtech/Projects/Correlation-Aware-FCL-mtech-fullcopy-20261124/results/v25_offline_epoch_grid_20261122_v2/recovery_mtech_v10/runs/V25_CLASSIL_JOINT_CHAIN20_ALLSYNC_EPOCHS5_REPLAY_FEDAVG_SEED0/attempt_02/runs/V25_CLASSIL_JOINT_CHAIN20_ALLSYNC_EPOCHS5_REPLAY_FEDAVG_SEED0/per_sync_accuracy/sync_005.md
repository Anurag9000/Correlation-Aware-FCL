# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 30.62% on 26 observed labels.
Validation accuracy: 0.28307692307692306; seen labels never predicted: 2.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.30615384615384617.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **63.0%** (0: 70%, 1: 56%) | **32.0%** (5: 34%, 6: 30%) | **5.5%** (10: 3%, 11: 8%) | **14.0%** (15: 27%, 16: 1%) | **31.0%** (20: 61%, 21: 1%) |
| Client 1 | **42.0%** (1: 56%, 2: 28%) | **41.5%** (6: 30%, 7: 53%) | **29.0%** (11: 8%, 12: 50%) | **38.0%** (16: 1%, 17: 75%) | **0.5%** (21: 1%, 22: 0%) |
| Client 2 | **28.0%** (2: 28%, 3: 28%) | **45.5%** (7: 53%, 8: 38%) | **40.0%** (12: 50%, 13: 30%) | **68.5%** (17: 75%, 18: 62%) | **15.0%** (22: 0%, 23: 30%) |
| Client 3 | **34.5%** (3: 28%, 4: 41%) | **32.0%** (8: 38%, 9: 26%) | **16.5%** (13: 30%, 14: 3%) | **42.5%** (18: 62%, 19: 23%) | **24.0%** (23: 30%, 24: 18%) |
| Client 4 | **37.5%** (4: 41%, 5: 34%) | **14.5%** (9: 26%, 10: 3%) | **15.0%** (14: 3%, 15: 27%) | **42.0%** (19: 23%, 20: 61%) | **9.0%** (24: 18%, 25: 0%) |
