# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 58.70% on 60 observed labels.
Validation accuracy: 0.587; seen labels never predicted: 4.
Server-accepted classes: 48; pending private classes: 12.
Accepted-only TEST accuracy: 0.6127083333333333.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **71.0%** (0: 88%, 1: 54%) | **55.0%** (10: 65%, 11: 45%) | **59.5%** (20: 73%, 21: 46%) | **50.0%** (30: 0%, 31: 100%) | **51.5%** (40: 18%, 41: 85%) | **46.5%** (50: 12%, 51: 81%) |
| Client 1 | **68.0%** (2: 37%, 3: 99%) | **54.5%** (12: 95%, 13: 14%) | **72.5%** (22: 61%, 23: 84%) | **42.5%** (32: 78%, 33: 7%) [PENDING upload] | **50.0%** (42: 100%, 43: 0%) [PENDING upload] | **50.0%** (52: 100%, 53: 0%) [PENDING upload] |
| Client 2 | **60.5%** (4: 88%, 5: 33%) | **56.5%** (14: 22%, 15: 91%) | **50.5%** (24: 1%, 25: 100%) | **64.5%** (34: 72%, 35: 57%) | **68.5%** (44: 88%, 45: 49%) | **49.5%** (54: 99%, 55: 0%) [PENDING upload] |
| Client 3 | **54.5%** (6: 27%, 7: 82%) | **68.5%** (16: 40%, 17: 97%) | **71.5%** (26: 46%, 27: 97%) | **63.0%** (36: 32%, 37: 94%) | **50.0%** (46: 100%, 47: 0%) [PENDING upload] | **48.5%** (56: 4%, 57: 93%) [PENDING upload] |
| Client 4 | **49.5%** (8: 99%, 9: 0%) | **71.0%** (18: 70%, 19: 72%) | **67.5%** (28: 81%, 29: 54%) | **66.0%** (38: 77%, 39: 55%) | **65.5%** (48: 36%, 49: 95%) | **64.5%** (58: 31%, 59: 98%) |
