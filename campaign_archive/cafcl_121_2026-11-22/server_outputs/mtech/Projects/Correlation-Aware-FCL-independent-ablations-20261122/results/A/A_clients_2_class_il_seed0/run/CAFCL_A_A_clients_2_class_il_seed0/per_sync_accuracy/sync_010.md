# Server sync 10 — global model v10

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.50% on 40 observed labels.
Validation accuracy: 0.413; seen labels never predicted: 0.
Server-accepted classes: 40; pending private classes: 0.
Accepted-only TEST accuracy: 0.415.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 | U9 | U10 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **54.5%** (0: 73%, 1: 36%) | **32.5%** (4: 21%, 5: 44%) | **44.0%** (8: 41%, 9: 47%) | **45.0%** (12: 44%, 13: 46%) | **66.0%** (16: 48%, 17: 84%) | **68.5%** (20: 76%, 21: 61%) | **53.0%** (24: 79%, 25: 27%) | **51.0%** (28: 55%, 29: 47%) | **47.5%** (32: 38%, 33: 57%) | **12.5%** (36: 11%, 37: 14%) |
| Client 1 | **13.0%** (2: 9%, 3: 17%) | **38.0%** (6: 43%, 7: 33%) | **29.5%** (10: 26%, 11: 33%) | **41.0%** (14: 33%, 15: 49%) | **36.0%** (18: 45%, 19: 27%) | **49.5%** (22: 38%, 23: 61%) | **31.0%** (26: 20%, 27: 42%) | **62.0%** (30: 78%, 31: 46%) | **51.5%** (34: 51%, 35: 52%) | **4.0%** (38: 3%, 39: 5%) |
