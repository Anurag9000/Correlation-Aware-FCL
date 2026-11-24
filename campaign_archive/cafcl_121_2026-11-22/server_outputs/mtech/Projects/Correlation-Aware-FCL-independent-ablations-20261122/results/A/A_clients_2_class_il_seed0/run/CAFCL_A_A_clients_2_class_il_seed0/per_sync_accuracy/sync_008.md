# Server sync 8 — global model v8

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 44.94% on 32 observed labels.
Validation accuracy: 0.45625; seen labels never predicted: 0.
Server-accepted classes: 32; pending private classes: 0.
Accepted-only TEST accuracy: 0.449375.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **62.5%** (0: 75%, 1: 50%) | **34.5%** (4: 31%, 5: 38%) | **46.0%** (8: 46%, 9: 46%) | **43.5%** (12: 45%, 13: 42%) | **69.5%** (16: 56%, 17: 83%) | **66.5%** (20: 72%, 21: 61%) | **53.5%** (24: 71%, 25: 36%) | **12.5%** (28: 19%, 29: 6%) |
| Client 1 | **33.0%** (2: 43%, 3: 23%) | **41.5%** (6: 40%, 7: 43%) | **32.5%** (10: 22%, 11: 43%) | **52.0%** (14: 43%, 15: 61%) | **46.0%** (18: 49%, 19: 43%) | **62.0%** (22: 47%, 23: 77%) | **48.5%** (26: 41%, 27: 56%) | **15.0%** (30: 27%, 31: 3%) |
