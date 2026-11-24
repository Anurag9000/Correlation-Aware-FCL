# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.48% on 50 observed labels.
Validation accuracy: 0.6072; seen labels never predicted: 4.
Server-accepted classes: 34; pending private classes: 16.
Accepted-only TEST accuracy: 0.65.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **49.5%** (0: 13%, 1: 86%) [PENDING upload] | **57.0%** (10: 62%, 11: 52%) [PENDING upload] | **55.0%** (20: 99%, 21: 11%) [PENDING upload] | **50.0%** (30: 0%, 31: 100%) [PENDING upload] | **53.5%** (40: 9%, 41: 98%) [PENDING upload] |
| Client 1 | **69.5%** (2: 40%, 3: 99%) | **58.5%** (12: 91%, 13: 26%) | **70.5%** (22: 59%, 23: 82%) | **42.0%** (32: 76%, 33: 8%) [PENDING upload] | **50.0%** (42: 100%, 43: 0%) [PENDING upload] |
| Client 2 | **77.0%** (4: 79%, 5: 75%) | **67.0%** (14: 76%, 15: 58%) | **50.0%** (24: 0%, 25: 100%) | **62.5%** (34: 57%, 35: 68%) | **64.5%** (44: 45%, 45: 84%) |
| Client 3 | **59.0%** (6: 54%, 7: 64%) | **69.5%** (16: 42%, 17: 97%) | **72.5%** (26: 48%, 27: 97%) | **73.0%** (36: 58%, 37: 88%) | **50.0%** (46: 100%, 47: 0%) [PENDING upload] |
| Client 4 | **49.0%** (8: 97%, 9: 1%) | **70.0%** (18: 72%, 19: 68%) | **66.5%** (28: 52%, 29: 81%) | **55.5%** (38: 91%, 39: 20%) | **70.5%** (48: 48%, 49: 93%) |
