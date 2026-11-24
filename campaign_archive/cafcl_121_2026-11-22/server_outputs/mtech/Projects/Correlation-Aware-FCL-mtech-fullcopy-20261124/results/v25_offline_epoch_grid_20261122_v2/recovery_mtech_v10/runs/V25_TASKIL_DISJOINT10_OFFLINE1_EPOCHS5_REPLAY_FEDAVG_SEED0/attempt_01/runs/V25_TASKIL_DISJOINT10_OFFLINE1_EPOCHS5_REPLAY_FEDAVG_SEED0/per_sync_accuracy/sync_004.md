# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 70.40% on 40 observed labels.
Validation accuracy: 0.6935; seen labels never predicted: 1.
Server-accepted classes: 38; pending private classes: 2.
Accepted-only TEST accuracy: 0.7147368421052631.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **76.5%** (0: 63%, 1: 90%) | **51.0%** (10: 92%, 11: 10%) | **91.5%** (20: 83%, 21: 100%) | **50.0%** (30: 0%, 31: 100%) [PENDING upload] |
| Client 1 | **77.0%** (2: 56%, 3: 98%) | **68.5%** (12: 91%, 13: 46%) | **83.0%** (22: 81%, 23: 85%) | **71.5%** (32: 49%, 33: 94%) |
| Client 2 | **75.0%** (4: 92%, 5: 58%) | **65.5%** (14: 38%, 15: 93%) | **76.5%** (24: 76%, 25: 77%) | **71.0%** (34: 83%, 35: 59%) |
| Client 3 | **63.5%** (6: 28%, 7: 99%) | **88.0%** (16: 82%, 17: 94%) | **57.0%** (26: 14%, 27: 100%) | **61.0%** (36: 24%, 37: 98%) |
| Client 4 | **61.5%** (8: 98%, 9: 25%) | **75.5%** (18: 77%, 19: 74%) | **71.0%** (28: 58%, 29: 84%) | **73.5%** (38: 75%, 39: 72%) |
