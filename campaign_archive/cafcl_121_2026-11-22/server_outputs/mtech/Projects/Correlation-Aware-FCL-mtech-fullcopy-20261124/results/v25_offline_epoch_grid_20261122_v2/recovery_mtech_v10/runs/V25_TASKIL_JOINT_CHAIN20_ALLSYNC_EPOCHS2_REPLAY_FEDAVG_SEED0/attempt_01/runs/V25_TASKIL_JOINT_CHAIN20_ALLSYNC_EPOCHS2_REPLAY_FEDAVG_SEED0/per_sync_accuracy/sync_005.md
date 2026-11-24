# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.81% on 26 observed labels.
Validation accuracy: 0.6046153846153847; seen labels never predicted: 0.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.5980769230769231.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **65.0%** (0: 62%, 1: 68%) | **66.0%** (5: 78%, 6: 54%) | **61.0%** (10: 82%, 11: 40%) | **59.0%** (15: 89%, 16: 29%) | **91.0%** (20: 87%, 21: 95%) |
| Client 1 | **67.0%** (1: 68%, 2: 66%) | **27.0%** (6: 54%, 7: 0%) | **65.5%** (11: 40%, 12: 91%) | **64.0%** (16: 29%, 17: 99%) | **50.5%** (21: 95%, 22: 6%) |
| Client 2 | **68.5%** (2: 66%, 3: 71%) | **49.5%** (7: 0%, 8: 99%) | **60.0%** (12: 91%, 13: 29%) | **79.0%** (17: 99%, 18: 59%) | **52.0%** (22: 6%, 23: 98%) |
| Client 3 | **64.5%** (3: 71%, 4: 58%) | **55.0%** (8: 99%, 9: 11%) | **35.5%** (13: 29%, 14: 42%) | **67.0%** (18: 59%, 19: 75%) | **50.5%** (23: 98%, 24: 3%) |
| Client 4 | **68.0%** (4: 58%, 5: 78%) | **46.5%** (9: 11%, 10: 82%) | **65.5%** (14: 42%, 15: 89%) | **81.0%** (19: 75%, 20: 87%) | **33.5%** (24: 3%, 25: 64%) |
