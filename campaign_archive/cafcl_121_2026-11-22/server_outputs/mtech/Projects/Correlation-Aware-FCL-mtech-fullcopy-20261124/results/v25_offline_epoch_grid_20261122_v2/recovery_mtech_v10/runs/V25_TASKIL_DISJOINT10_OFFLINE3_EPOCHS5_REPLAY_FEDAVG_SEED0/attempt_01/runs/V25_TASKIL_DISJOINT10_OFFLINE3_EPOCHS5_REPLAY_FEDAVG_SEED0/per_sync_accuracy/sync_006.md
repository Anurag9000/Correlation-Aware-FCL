# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 54.08% on 60 observed labels.
Validation accuracy: 0.5433333333333333; seen labels never predicted: 3.
Server-accepted classes: 48; pending private classes: 12.
Accepted-only TEST accuracy: 0.5564583333333334.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **50.0%** (0: 0%, 1: 100%) | **51.0%** (10: 98%, 11: 4%) | **59.0%** (20: 18%, 21: 100%) | **50.0%** (30: 0%, 31: 100%) | **49.5%** (40: 92%, 41: 7%) | **58.0%** (50: 21%, 51: 95%) |
| Client 1 | **52.5%** (2: 5%, 3: 100%) | **52.0%** (12: 94%, 13: 10%) | **83.5%** (22: 85%, 23: 82%) | **43.5%** (32: 10%, 33: 77%) [PENDING upload] | **50.5%** (42: 100%, 43: 1%) [PENDING upload] | **50.0%** (52: 99%, 53: 1%) [PENDING upload] |
| Client 2 | **66.0%** (4: 97%, 5: 35%) | **51.5%** (14: 3%, 15: 100%) | **49.5%** (24: 0%, 25: 99%) | **53.5%** (34: 99%, 35: 8%) | **58.5%** (44: 93%, 45: 24%) | **47.0%** (54: 93%, 55: 1%) [PENDING upload] |
| Client 3 | **55.0%** (6: 12%, 7: 98%) | **51.5%** (16: 3%, 17: 100%) | **54.0%** (26: 8%, 27: 100%) | **50.5%** (36: 1%, 37: 100%) | **50.0%** (46: 100%, 47: 0%) [PENDING upload] | **46.0%** (56: 81%, 57: 11%) [PENDING upload] |
| Client 4 | **51.5%** (8: 100%, 9: 3%) | **55.5%** (18: 11%, 19: 100%) | **50.5%** (28: 1%, 29: 100%) | **49.0%** (38: 2%, 39: 96%) | **65.5%** (48: 34%, 49: 97%) | **68.0%** (58: 93%, 59: 43%) |
