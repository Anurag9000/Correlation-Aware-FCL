# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 70.26% on 50 observed labels.
Validation accuracy: 0.7024; seen labels never predicted: 0.
Server-accepted classes: 38; pending private classes: 12.
Accepted-only TEST accuracy: 0.7663157894736842.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **64.0%** (0: 32%, 1: 96%) | **63.5%** (10: 90%, 11: 37%) | **57.5%** (20: 15%, 21: 100%) | **94.5%** (30: 96%, 31: 93%) | **52.0%** (40: 13%, 41: 91%) [PENDING upload] |
| Client 1 | **81.0%** (2: 62%, 3: 100%) | **70.0%** (12: 45%, 13: 95%) | **85.5%** (22: 80%, 23: 91%) | **85.0%** (32: 88%, 33: 82%) | **64.5%** (42: 99%, 43: 30%) |
| Client 2 | **45.5%** (4: 14%, 5: 77%) [PENDING upload] | **43.5%** (14: 83%, 15: 4%) [PENDING upload] | **37.5%** (24: 24%, 25: 51%) [PENDING upload] | **57.5%** (34: 89%, 35: 26%) [PENDING upload] | **64.5%** (44: 71%, 45: 58%) [PENDING upload] |
| Client 3 | **70.5%** (6: 46%, 7: 95%) | **84.0%** (16: 92%, 17: 76%) | **76.0%** (26: 54%, 27: 98%) | **81.0%** (36: 64%, 37: 98%) | **74.0%** (46: 100%, 47: 48%) |
| Client 4 | **76.0%** (8: 99%, 9: 53%) | **79.5%** (18: 63%, 19: 96%) | **75.0%** (28: 63%, 29: 87%) | **80.5%** (38: 86%, 39: 75%) | **94.0%** (48: 96%, 49: 92%) |
