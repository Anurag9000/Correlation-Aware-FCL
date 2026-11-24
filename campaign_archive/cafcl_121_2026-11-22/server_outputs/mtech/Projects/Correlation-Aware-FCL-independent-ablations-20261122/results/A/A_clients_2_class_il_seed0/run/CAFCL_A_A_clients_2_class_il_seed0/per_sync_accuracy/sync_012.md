# Server sync 12 — global model v12

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.73% on 48 observed labels.
Validation accuracy: 0.4325; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 0.
Accepted-only TEST accuracy: 0.41729166666666667.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 | U9 | U10 | U11 | U12 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **65.0%** (0: 69%, 1: 61%) | **24.5%** (4: 18%, 5: 31%) | **43.0%** (8: 38%, 9: 48%) | **50.5%** (12: 51%, 13: 50%) | **60.5%** (16: 46%, 17: 75%) | **66.5%** (20: 69%, 21: 64%) | **40.5%** (24: 61%, 25: 20%) | **38.5%** (28: 46%, 29: 31%) | **50.0%** (32: 41%, 33: 59%) | **46.5%** (36: 47%, 37: 46%) | **55.0%** (40: 35%, 41: 75%) | **7.5%** (44: 7%, 45: 8%) |
| Client 1 | **22.0%** (2: 32%, 3: 12%) | **37.5%** (6: 38%, 7: 37%) | **30.5%** (10: 21%, 11: 40%) | **32.0%** (14: 36%, 15: 28%) | **27.0%** (18: 32%, 19: 22%) | **49.0%** (22: 28%, 23: 70%) | **37.5%** (26: 39%, 27: 36%) | **61.0%** (30: 76%, 31: 46%) | **51.5%** (34: 52%, 35: 51%) | **36.5%** (38: 28%, 39: 45%) | **50.0%** (42: 37%, 43: 63%) | **19.0%** (46: 3%, 47: 35%) |
