# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 55.62% on 50 observed labels.
Validation accuracy: 0.5656; seen labels never predicted: 3.
Server-accepted classes: 34; pending private classes: 16.
Accepted-only TEST accuracy: 0.5697058823529412.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **59.0%** (0: 18%, 1: 100%) [PENDING upload] | **50.0%** (10: 64%, 11: 36%) [PENDING upload] | **58.0%** (20: 64%, 21: 52%) [PENDING upload] | **50.0%** (30: 0%, 31: 100%) [PENDING upload] | **58.5%** (40: 28%, 41: 89%) [PENDING upload] |
| Client 1 | **58.0%** (2: 16%, 3: 100%) | **49.0%** (12: 94%, 13: 4%) | **76.0%** (22: 99%, 23: 53%) | **45.5%** (32: 34%, 33: 57%) [PENDING upload] | **51.0%** (42: 100%, 43: 2%) [PENDING upload] |
| Client 2 | **69.0%** (4: 96%, 5: 42%) | **55.5%** (14: 11%, 15: 100%) | **50.0%** (24: 0%, 25: 100%) | **53.0%** (34: 100%, 35: 6%) | **52.5%** (44: 7%, 45: 98%) |
| Client 3 | **61.5%** (6: 30%, 7: 93%) | **53.5%** (16: 7%, 17: 100%) | **55.0%** (26: 10%, 27: 100%) | **50.5%** (36: 1%, 37: 100%) | **50.0%** (46: 100%, 47: 0%) [PENDING upload] |
| Client 4 | **53.5%** (8: 99%, 9: 8%) | **66.0%** (18: 32%, 19: 100%) | **50.5%** (28: 1%, 29: 100%) | **56.0%** (38: 19%, 39: 93%) | **59.0%** (48: 21%, 49: 97%) |
