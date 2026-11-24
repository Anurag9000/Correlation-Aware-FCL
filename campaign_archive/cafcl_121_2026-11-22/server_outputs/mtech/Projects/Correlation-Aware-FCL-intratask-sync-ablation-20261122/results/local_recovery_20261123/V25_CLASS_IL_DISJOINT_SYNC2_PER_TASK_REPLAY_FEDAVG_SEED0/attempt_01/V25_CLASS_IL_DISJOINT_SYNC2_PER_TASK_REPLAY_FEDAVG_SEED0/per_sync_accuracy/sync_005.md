# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 29.27% on 30 observed labels.
Validation accuracy: 0.292; seen labels never predicted: 9.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.2926666666666667.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **69.0%** (0: 72%, 1: 66%) | **69.0%** (0: 72%, 1: 66%) | **13.5%** (10: 13%, 11: 14%) | **13.5%** (10: 13%, 11: 14%) | **0.0%** (20: 0%, 21: 0%) |
| Client 1 | **42.5%** (2: 25%, 3: 60%) | **42.5%** (2: 25%, 3: 60%) | **46.0%** (12: 59%, 13: 33%) | **46.0%** (12: 59%, 13: 33%) | **0.0%** (22: 0%, 23: 0%) |
| Client 2 | **57.5%** (4: 47%, 5: 68%) | **57.5%** (4: 47%, 5: 68%) | **31.0%** (14: 15%, 15: 47%) | **31.0%** (14: 15%, 15: 47%) | **0.0%** (24: 0%, 25: 0%) |
| Client 3 | **39.0%** (6: 39%, 7: 39%) | **39.0%** (6: 39%, 7: 39%) | **50.5%** (16: 27%, 17: 74%) | **50.5%** (16: 27%, 17: 74%) | **0.0%** (26: 0%, 27: 0%) |
| Client 4 | **54.5%** (8: 51%, 9: 58%) | **54.5%** (8: 51%, 9: 58%) | **35.5%** (18: 43%, 19: 28%) | **35.5%** (18: 43%, 19: 28%) | **0.0%** (28: 0%, 29: 0%) |
