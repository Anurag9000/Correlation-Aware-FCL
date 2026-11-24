# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 55.02% on 40 observed labels.
Validation accuracy: 0.552; seen labels never predicted: 2.
Server-accepted classes: 26; pending private classes: 14.
Accepted-only TEST accuracy: 0.5757692307692308.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **61.5%** (0: 23%, 1: 100%) [PENDING upload] | **47.5%** (10: 92%, 11: 3%) [PENDING upload] | **53.0%** (20: 9%, 21: 97%) [PENDING upload] | **50.0%** (30: 0%, 31: 100%) [PENDING upload] |
| Client 1 | **67.0%** (2: 34%, 3: 100%) | **48.5%** (12: 96%, 13: 1%) | **62.5%** (22: 100%, 23: 25%) | **40.5%** (32: 16%, 33: 65%) [PENDING upload] |
| Client 2 | **67.5%** (4: 99%, 5: 36%) | **50.0%** (14: 0%, 15: 100%) | **49.0%** (24: 0%, 25: 98%) [PENDING upload] | **50.5%** (34: 100%, 35: 1%) [PENDING upload] |
| Client 3 | **56.5%** (6: 19%, 7: 94%) | **57.0%** (16: 15%, 17: 99%) | **61.5%** (26: 24%, 27: 99%) | **51.0%** (36: 2%, 37: 100%) |
| Client 4 | **52.5%** (8: 99%, 9: 6%) | **52.5%** (18: 5%, 19: 100%) | **50.5%** (28: 1%, 29: 100%) | **71.5%** (38: 89%, 39: 54%) |
