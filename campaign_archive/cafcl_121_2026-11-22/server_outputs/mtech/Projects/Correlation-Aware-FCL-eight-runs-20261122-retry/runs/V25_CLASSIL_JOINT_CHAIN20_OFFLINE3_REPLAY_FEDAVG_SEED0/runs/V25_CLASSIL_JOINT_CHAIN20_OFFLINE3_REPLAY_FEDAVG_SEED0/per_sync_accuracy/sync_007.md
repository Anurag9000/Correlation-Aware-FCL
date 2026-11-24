# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 34.19% on 36 observed labels.
Validation accuracy: 0.33055555555555555; seen labels never predicted: 4.
Server-accepted classes: 32; pending private classes: 4.
Accepted-only TEST accuracy: 0.3846875.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **61.5%** (0: 58%, 1: 65%) | **36.5%** (5: 22%, 6: 51%) | **12.0%** (10: 22%, 11: 2%) | **12.5%** (15: 25%, 16: 0%) | **63.5%** (20: 69%, 21: 58%) | **23.0%** (25: 40%, 26: 6%) | **58.5%** (30: 93%, 31: 24%) |
| Client 1 | **53.5%** (1: 65%, 2: 42%) | **48.0%** (6: 51%, 7: 45%) | **15.5%** (11: 2%, 12: 29%) | **37.5%** (16: 0%, 17: 75%) | **37.5%** (21: 58%, 22: 17%) | **3.0%** (26: 6%, 27: 0%) [PENDING upload] | **12.0%** (31: 24%, 32: 0%) [PENDING upload] |
| Client 2 | **46.0%** (2: 42%, 3: 50%) | **51.5%** (7: 45%, 8: 58%) | **29.5%** (12: 29%, 13: 30%) | **55.0%** (17: 75%, 18: 35%) | **46.0%** (22: 17%, 23: 75%) | **0.0%** (27: 0%, 28: 0%) [PENDING upload] | **0.0%** (32: 0%, 33: 0%) [PENDING upload] |
| Client 3 | **40.5%** (3: 50%, 4: 31%) | **60.5%** (8: 58%, 9: 63%) | **16.5%** (13: 30%, 14: 3%) | **23.5%** (18: 35%, 19: 12%) | **73.0%** (23: 75%, 24: 71%) | **10.0%** (28: 0%, 29: 20%) [PENDING upload] | **20.0%** (33: 0%, 34: 40%) [PENDING upload] |
| Client 4 | **26.5%** (4: 31%, 5: 22%) | **42.5%** (9: 63%, 10: 22%) | **14.0%** (14: 3%, 15: 25%) | **40.5%** (19: 12%, 20: 69%) | **55.5%** (24: 71%, 25: 40%) | **56.5%** (29: 20%, 30: 93%) | **20.0%** (34: 40%, 35: 0%) |
