# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 65.33% on 40 observed labels.
Validation accuracy: 0.6475; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 10.
Accepted-only TEST accuracy: 0.707.
Terminal flush: False. Selected clients: [0, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **69.5%** (0: 40%, 1: 99%) | **56.0%** (10: 83%, 11: 29%) | **56.5%** (20: 13%, 21: 100%) | **82.0%** (30: 66%, 31: 98%) |
| Client 1 | **74.0%** (2: 48%, 3: 100%) | **71.5%** (12: 51%, 13: 92%) | **85.0%** (22: 82%, 23: 88%) | **64.5%** (32: 87%, 33: 42%) [PENDING upload] |
| Client 2 | **44.5%** (4: 18%, 5: 71%) [PENDING upload] | **48.5%** (14: 95%, 15: 2%) [PENDING upload] | **35.0%** (24: 34%, 25: 36%) [PENDING upload] | **53.5%** (34: 99%, 35: 8%) [PENDING upload] |
| Client 3 | **56.5%** (6: 14%, 7: 99%) | **84.0%** (16: 91%, 17: 77%) | **67.0%** (26: 35%, 27: 99%) | **64.5%** (36: 31%, 37: 98%) |
| Client 4 | **70.0%** (8: 97%, 9: 43%) | **80.5%** (18: 65%, 19: 96%) | **64.0%** (28: 87%, 29: 41%) | **79.5%** (38: 85%, 39: 74%) |
