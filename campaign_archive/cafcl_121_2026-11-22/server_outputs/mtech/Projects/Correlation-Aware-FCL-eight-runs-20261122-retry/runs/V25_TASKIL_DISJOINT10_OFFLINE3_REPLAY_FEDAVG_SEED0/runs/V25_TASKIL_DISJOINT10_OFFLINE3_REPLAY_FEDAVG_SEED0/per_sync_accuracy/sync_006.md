# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.18% on 60 observed labels.
Validation accuracy: 0.5293333333333333; seen labels never predicted: 15.
Server-accepted classes: 48; pending private classes: 12.
Accepted-only TEST accuracy: 0.5320833333333334.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) | **53.5%** (10: 86%, 11: 21%) | **57.0%** (20: 15%, 21: 99%) | **50.0%** (30: 0%, 31: 100%) | **37.5%** (40: 19%, 41: 56%) | **50.0%** (50: 100%, 51: 0%) |
| Client 1 | **50.0%** (2: 0%, 3: 100%) | **52.0%** (12: 25%, 13: 79%) | **51.0%** (22: 100%, 23: 2%) | **51.5%** (32: 29%, 33: 74%) [PENDING upload] | **50.0%** (42: 100%, 43: 0%) [PENDING upload] | **41.5%** (52: 83%, 53: 0%) [PENDING upload] |
| Client 2 | **57.5%** (4: 100%, 5: 15%) | **50.0%** (14: 0%, 15: 100%) | **50.0%** (24: 0%, 25: 100%) | **50.0%** (34: 100%, 35: 0%) | **50.0%** (44: 100%, 45: 0%) | **50.0%** (54: 100%, 55: 0%) [PENDING upload] |
| Client 3 | **50.0%** (6: 0%, 7: 100%) | **52.5%** (16: 5%, 17: 100%) | **50.0%** (26: 0%, 27: 100%) | **50.0%** (36: 0%, 37: 100%) | **50.0%** (46: 100%, 47: 0%) [PENDING upload] | **45.5%** (56: 81%, 57: 10%) [PENDING upload] |
| Client 4 | **52.5%** (8: 100%, 9: 5%) | **60.0%** (18: 22%, 19: 98%) | **50.0%** (28: 0%, 29: 100%) | **64.5%** (38: 46%, 39: 83%) | **80.5%** (48: 67%, 49: 94%) | **58.5%** (58: 24%, 59: 93%) |
