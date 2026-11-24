# Server sync 8 — global model v8

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 41.54% on 48 observed labels.
Validation accuracy: 0.42791666666666667; seen labels never predicted: 5.
Server-accepted classes: 48; pending private classes: 0.
Accepted-only TEST accuracy: 0.41541666666666666.
Terminal flush: False. Selected clients: [0, 1, 2].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **56.0%** (0: 77%, 1: 35%) | **45.0%** (6: 53%, 7: 37%) | **56.5%** (12: 59%, 13: 54%) | **39.5%** (18: 42%, 19: 37%) | **50.5%** (24: 74%, 25: 27%) | **71.5%** (30: 85%, 31: 58%) | **34.5%** (36: 39%, 37: 30%) | **0.0%** (42: 0%, 43: 0%) |
| Client 1 | **28.0%** (2: 28%, 3: 28%) | **45.0%** (8: 52%, 9: 38%) | **40.5%** (14: 32%, 15: 49%) | **74.5%** (20: 82%, 21: 67%) | **39.0%** (26: 28%, 27: 50%) | **43.0%** (32: 34%, 33: 52%) | **38.0%** (38: 30%, 39: 46%) | **0.0%** (44: 0%, 45: 0%) |
| Client 2 | **48.0%** (4: 49%, 5: 47%) | **41.5%** (10: 36%, 11: 47%) | **61.0%** (16: 46%, 17: 76%) | **54.5%** (22: 32%, 23: 77%) | **44.5%** (28: 46%, 29: 43%) | **33.0%** (34: 36%, 35: 30%) | **53.0%** (40: 33%, 41: 73%) | **0.0%** (46: 0%, 47: 0%) |
