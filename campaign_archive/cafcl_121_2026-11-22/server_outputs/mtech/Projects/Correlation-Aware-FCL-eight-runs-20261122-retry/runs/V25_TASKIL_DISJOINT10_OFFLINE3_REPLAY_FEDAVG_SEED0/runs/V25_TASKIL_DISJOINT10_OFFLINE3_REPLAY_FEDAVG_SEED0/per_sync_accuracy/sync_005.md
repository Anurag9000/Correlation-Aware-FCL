# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.90% on 50 observed labels.
Validation accuracy: 0.5352; seen labels never predicted: 11.
Server-accepted classes: 34; pending private classes: 16.
Accepted-only TEST accuracy: 0.5373529411764706.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **51.5%** (0: 5%, 1: 98%) [PENDING upload] | **57.5%** (10: 50%, 11: 65%) [PENDING upload] | **50.0%** (20: 100%, 21: 0%) [PENDING upload] | **50.0%** (30: 0%, 31: 100%) [PENDING upload] | **50.0%** (40: 0%, 41: 100%) [PENDING upload] |
| Client 1 | **53.5%** (2: 7%, 3: 100%) | **54.5%** (12: 77%, 13: 32%) | **50.0%** (22: 100%, 23: 0%) | **50.0%** (32: 54%, 33: 46%) [PENDING upload] | **50.0%** (42: 100%, 43: 0%) [PENDING upload] |
| Client 2 | **55.5%** (4: 100%, 5: 11%) | **50.5%** (14: 1%, 15: 100%) | **50.0%** (24: 0%, 25: 100%) | **50.0%** (34: 100%, 35: 0%) | **56.0%** (44: 97%, 45: 15%) |
| Client 3 | **50.0%** (6: 0%, 7: 100%) | **56.0%** (16: 12%, 17: 100%) | **50.0%** (26: 0%, 27: 100%) | **50.0%** (36: 0%, 37: 100%) | **50.0%** (46: 100%, 47: 0%) [PENDING upload] |
| Client 4 | **58.0%** (8: 100%, 9: 16%) | **66.0%** (18: 34%, 19: 98%) | **51.5%** (28: 3%, 29: 100%) | **51.0%** (38: 11%, 39: 91%) | **61.0%** (48: 23%, 49: 99%) |
