# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 39.45% on 20 observed labels.
Validation accuracy: 0.404; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.3945.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **56.5%** (0: 66%, 1: 47%) | **56.5%** (0: 66%, 1: 47%) | **14.5%** (10: 14%, 11: 15%) | **14.5%** (10: 14%, 11: 15%) |
| Client 1 | **32.0%** (2: 17%, 3: 47%) | **32.0%** (2: 17%, 3: 47%) | **64.5%** (12: 77%, 13: 52%) | **64.5%** (12: 77%, 13: 52%) |
| Client 2 | **42.0%** (4: 30%, 5: 54%) | **42.0%** (4: 30%, 5: 54%) | **22.5%** (14: 15%, 15: 30%) | **22.5%** (14: 15%, 15: 30%) |
| Client 3 | **42.0%** (6: 43%, 7: 41%) | **42.0%** (6: 43%, 7: 41%) | **44.0%** (16: 31%, 17: 57%) | **44.0%** (16: 31%, 17: 57%) |
| Client 4 | **35.5%** (8: 30%, 9: 41%) | **35.5%** (8: 30%, 9: 41%) | **41.0%** (18: 50%, 19: 32%) | **41.0%** (18: 50%, 19: 32%) |
