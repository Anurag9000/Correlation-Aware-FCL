# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 53.71% on 80 observed labels.
Validation accuracy: 0.53075; seen labels never predicted: 3.
Server-accepted classes: 60; pending private classes: 20.
Accepted-only TEST accuracy: 0.6328333333333334.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **65.5%** (0: 26%, 1: 61%, 2: 78%, 3: 97%) | **66.8%** (20: 45%, 21: 77%, 22: 53%, 23: 92%) | **43.8%** (40: 13%, 41: 39%, 42: 98%, 43: 25%) | **74.2%** (60: 85%, 61: 42%, 62: 71%, 63: 99%) |
| Client 1 | **65.0%** (4: 95%, 5: 65%, 6: 49%, 7: 51%) | **60.2%** (24: 54%, 25: 34%, 26: 56%, 27: 97%) | **62.5%** (44: 94%, 45: 40%, 46: 45%, 47: 71%) | **20.8%** (64: 0%, 65: 34%, 66: 47%, 67: 2%) [PENDING upload] |
| Client 2 | **26.7%** (8: 0%, 9: 8%, 10: 29%, 11: 70%) [PENDING upload] | **30.0%** (28: 99%, 29: 12%, 30: 0%, 31: 9%) [PENDING upload] | **23.0%** (48: 3%, 49: 1%, 50: 11%, 51: 77%) [PENDING upload] | **24.5%** (68: 49%, 69: 21%, 70: 28%, 71: 0%) [PENDING upload] |
| Client 3 | **60.2%** (12: 94%, 13: 45%, 14: 47%, 15: 55%) | **57.8%** (32: 34%, 33: 94%, 34: 55%, 35: 48%) | **79.0%** (52: 95%, 53: 66%, 54: 56%, 55: 99%) | **57.8%** (72: 52%, 73: 65%, 74: 86%, 75: 28%) |
| Client 4 | **70.2%** (16: 45%, 17: 75%, 18: 72%, 19: 89%) | **66.0%** (36: 27%, 37: 73%, 38: 87%, 39: 77%) | **64.0%** (56: 68%, 57: 56%, 58: 58%, 59: 74%) | **56.2%** (76: 79%, 77: 6%, 78: 89%, 79: 51%) |
