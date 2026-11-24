# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 59.62% on 40 observed labels.
Validation accuracy: 0.612; seen labels never predicted: 3.
Server-accepted classes: 26; pending private classes: 14.
Accepted-only TEST accuracy: 0.6546153846153846.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **49.0%** (0: 5%, 1: 93%) [PENDING upload] | **59.0%** (10: 47%, 11: 71%) [PENDING upload] | **50.0%** (20: 100%, 21: 0%) [PENDING upload] | **50.0%** (30: 0%, 31: 100%) [PENDING upload] |
| Client 1 | **81.5%** (2: 73%, 3: 90%) | **64.5%** (12: 85%, 13: 44%) | **71.5%** (22: 95%, 23: 48%) | **41.0%** (32: 76%, 33: 6%) [PENDING upload] |
| Client 2 | **67.5%** (4: 68%, 5: 67%) | **53.0%** (14: 97%, 15: 9%) | **47.5%** (24: 31%, 25: 64%) [PENDING upload] | **45.0%** (34: 67%, 35: 23%) [PENDING upload] |
| Client 3 | **67.0%** (6: 92%, 7: 42%) | **74.5%** (16: 77%, 17: 72%) | **70.5%** (26: 68%, 27: 73%) | **79.5%** (36: 77%, 37: 82%) |
| Client 4 | **50.0%** (8: 100%, 9: 0%) | **61.0%** (18: 65%, 19: 57%) | **56.0%** (28: 94%, 29: 18%) | **54.5%** (38: 56%, 39: 53%) |
